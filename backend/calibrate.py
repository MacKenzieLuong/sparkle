"""Measure what the car actually does, so steering can be computed in degrees.

    DRIVER=tb6612 CAMERA=rpicam .venv/bin/python calibrate.py

The steering math currently has no units: `dx` is a fraction of frame width and
TURN_GAIN converts it to a throttle difference by guesswork, so nothing can
work out how long to turn for. These experiments supply the missing constants.

Four things get measured:

  yaw rate    degrees per second at a given throttle difference
  coast       degrees the car keeps rotating after power is cut
  scale       degrees per pixel of horizontal image offset
  speed       metres per second forward at a given throttle

No compass or protractor is needed. The car turns while you press Enter at the
quarter and half turn; pressing at two marks makes your reaction time cancel
out of the arithmetic. Degrees per pixel is measured with optical flow: the car
turns a known amount and the image reports how far it moved.

THE CAR MUST BE ON THE FLOOR with clear space around it. On blocks the wheels
turn without the body rotating and every number comes out meaningless.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "calibration.json"
PIVOT_DIFFERENTIALS = (0.25, 0.35, 0.5)
PIVOT_TIMEOUT = 25.0  # the car is spinning; never wait on a press forever
FORWARD_THROTTLES = (0.2, 0.35)
FORWARD_SECONDS = 2.0


def ask_float(prompt: str) -> Optional[float]:
    """Read a measurement. Blank skips, 'q' aborts the whole run."""
    while True:
        try:
            raw = input(f"    {prompt}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw.lower() in ("q", "quit", "abort"):
            return None
        if not raw:
            return float("nan")
        try:
            return float(raw)
        except ValueError:
            print("    (a number, blank to skip, or q to abort)")


def countdown(message: str, seconds: int = 3) -> bool:
    print(f"  {message}")
    try:
        for remaining in range(seconds, 0, -1):
            print(f"    starting in {remaining}...", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  aborted")
        return False
    return True


def pulse(driver, left: float, right: float, seconds: float) -> None:
    driver.apply(left, right)
    time.sleep(seconds)
    driver.stop()


def watchdog(driver, seconds: float) -> threading.Timer:
    """Cut the motors if nobody presses anything. The car is spinning."""
    timer = threading.Timer(seconds, driver.stop)
    timer.daemon = True
    timer.start()
    return timer


def press(prompt: str) -> Optional[float]:
    """Wait for Enter and report when it came. None aborts."""
    try:
        if input(f"    {prompt}").strip().lower() in ("q", "quit", "abort"):
            return None
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return time.monotonic()


def measure_pivots(driver) -> dict:
    """Yaw rate and coast, timed by eye against a quarter and a half turn.

    No compass or protractor: perpendicular and fully-reversed are both easy to
    judge. Pressing at *two* marks is what makes it accurate — with a reaction
    delay t, 90 = rate*(t90 - t) and 180 = rate*(t180 - t), so subtracting one
    from the other cancels the delay and leaves rate = 90 / (t180 - t90).

    Coast then comes from a third press when the car stops moving: a body
    slowing to rest sweeps about rate * time / 2.
    """
    print("\n=== Yaw rate and coast ===")
    print("The car pivots in place, slowly. Press Enter twice while it turns:")
    print("  once at the QUARTER turn  (square to where it started)")
    print("  once at the HALF turn     (facing exactly backwards)")
    print("Then once more when it has completely stopped moving.")
    print("Line the car up against a wall or a tile edge to judge the marks.")
    print("\nWATCH BOTH WHEELS. They must turn in opposite directions. If one")
    print("sits still the car is swinging around it, not spinning, and the")
    print("numbers will describe an arc — answer 'n' and raise the minimums.")
    results: dict[str, dict] = {}

    for differential in PIVOT_DIFFERENTIALS:
        if not countdown(f"pivot at differential {differential}", 3):
            return results

        guard = watchdog(driver, PIVOT_TIMEOUT)
        driver.apply(differential, -differential)
        started = time.monotonic()

        quarter = press("press Enter at the QUARTER turn... ")
        half = press("press Enter at the HALF turn... ") if quarter else None
        driver.stop()
        guard.cancel()
        if quarter is None or half is None:
            return results

        cut = time.monotonic()
        settled = press("press Enter once it has STOPPED moving... ")
        if settled is None:
            return results

        try:
            both = input("    did BOTH wheels turn? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return results
        if both.startswith("n"):
            print("    -> discarded: that was a swing around one wheel, not a spin")
            continue

        t90, t180 = quarter - started, half - started
        if t180 - t90 < 0.15:
            print("    -> the two presses were too close together to use")
            continue

        rate = 90.0 / (t180 - t90)
        reaction = t90 - 90.0 / rate
        coast = rate * (settled - cut) / 2.0
        results[str(differential)] = {
            "deg_per_second": round(rate, 2),
            "coast_deg": round(max(0.0, coast), 2),
            "reaction_s": round(reaction, 3),
            "measured": {"t90": round(t90, 3), "t180": round(t180, 3),
                         "coast_s": round(settled - cut, 3)},
        }
        print(f"    -> {rate:.1f} deg/s, coasting {max(0.0, coast):.1f} deg "
              f"over {settled - cut:.2f}s")
        if not -0.1 < reaction < 1.0:
            print(f"       (implied reaction time {reaction:+.2f}s looks off — "
                  "if the marks were misjudged, redo this one)")
    return results


def measure_scale(driver, camera, pivots: dict) -> Optional[dict]:
    """Degrees per pixel, by rotating a known amount and watching the image."""
    print("\n=== Degrees per pixel ===")
    print("Point the camera at something with texture — not a blank wall — about")
    print("2-3 m away. The car turns briefly and the image says how far it moved.")

    usable = {
        float(d): v["deg_per_second"]
        for d, v in pivots.items()
        if v.get("deg_per_second")
    }
    if not usable:
        print("  needs a yaw rate first; skipping")
        return None
    differential = min(usable)
    rate = usable[differential]
    seconds = 0.35

    if not countdown(f"brief pivot at differential {differential}", 3):
        return None

    try:
        before = cv2.cvtColor(camera.read(), cv2.COLOR_BGR2GRAY)
    except Exception as exc:
        print(f"  camera unavailable: {exc}")
        return None
    points = cv2.goodFeaturesToTrack(before, 200, 0.01, 7)
    if points is None or len(points) < 20:
        print("  too few features — aim at something more textured")
        return None

    pulse(driver, differential, -differential, seconds)
    time.sleep(0.6)
    after = cv2.cvtColor(camera.read(), cv2.COLOR_BGR2GRAY)

    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        before, after, points, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if moved is None:
        print("  optical flow failed")
        return None
    keep = status.reshape(-1) == 1
    if int(keep.sum()) < 20:
        print(f"  only {int(keep.sum())} points survived the turn; too fast to measure")
        return None

    shift = float(np.median(
        (moved.reshape(-1, 2)[keep] - points.reshape(-1, 2)[keep])[:, 0]
    ))
    if abs(shift) < 2:
        print(f"  image barely moved ({shift:.1f}px); is the car actually turning?")
        return None

    turned = rate * seconds  # the coast lands after the frame is taken
    width = before.shape[1]
    deg_per_pixel = turned / abs(shift)
    print(f"    turned ~{turned:.1f} deg, image moved {abs(shift):.1f} px")
    print(f"    -> {deg_per_pixel:.3f} deg/pixel, so {deg_per_pixel * width:.0f} deg across the frame")
    return {
        "deg_per_pixel": round(deg_per_pixel, 4),
        "frame_width": width,
        "horizontal_fov_deg": round(deg_per_pixel * width, 1),
        "from": {"differential": differential, "seconds": seconds, "shift_px": round(shift, 1)},
    }


def measure_stiction(driver) -> dict:
    """The throttle at which each wheel actually starts turning.

    A more heavily loaded side needs more PWM to break away, so small commands
    move one wheel and not the other: the car swings instead of easing forward.
    """
    print("\n=== Minimum throttle per side ===")
    print("Each wheel is ramped up on its own. Say when it starts turning.")
    print("Lift that wheel clear of the floor so it is free to spin.")
    results: dict[str, float] = {}

    for side, (left_sign, right_sign) in (("left", (1, 0)), ("right", (0, 1))):
        print(f"\n  {side} wheel:")
        found: Optional[float] = None
        for step in range(5, 100, 5):
            throttle = step / 100.0
            # Straight to _drive: conditioning is what is being measured.
            driver._drive(left_sign * throttle, right_sign * throttle)
            time.sleep(0.6)
            try:
                answer = input(f"    {throttle:.2f} — turning? [y/N/q] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                driver.stop()
                print()
                return results
            if answer in ("q", "quit"):
                driver.stop()
                return results
            if answer.startswith("y"):
                found = throttle
                break
        driver.stop()
        time.sleep(0.3)
        if found is not None:
            results[side] = found
            print(f"    -> {side} breaks away at {found:.2f}")
        else:
            print(f"    -> {side} never moved; check wiring and battery")
    return results


def measure_trim(driver, pivots: dict) -> Optional[dict]:
    """How much one side outruns the other when both are commanded equally.

    An uneven chassis curves a commanded-straight drive. The drift is converted
    into the throttle difference that would have caused it, using the yaw rate
    already measured, and that difference becomes a per-side scale.
    """
    print("\n=== Straight-line trim ===")
    print("The car drives with both sides commanded equally. Measure the heading")
    print("it ends up on, compared with the one it started on.")

    usable = {
        float(d): v["deg_per_second"]
        for d, v in pivots.items()
        if v.get("deg_per_second")
    }
    if not usable:
        print("  needs a yaw rate first; skipping")
        return None
    pivot_diff = min(usable)
    # Pivoting at +d/-d is a difference of 2d between the sides.
    deg_per_second_per_unit = usable[pivot_diff] / (2 * pivot_diff)

    throttle = FORWARD_THROTTLES[0]
    seconds = FORWARD_SECONDS
    if not countdown(f"drive straight at {throttle} for {seconds}s", 3):
        return None
    pulse(driver, throttle, throttle, seconds)
    time.sleep(0.8)

    print("    positive = veered RIGHT, negative = veered LEFT")
    drift = ask_float("degrees of heading change")
    if drift is None or np.isnan(drift):
        return None
    if abs(drift) < 1.0:
        print("    -> already straight; no trim needed")
        return {"left_scale": 1.0, "right_scale": 1.0, "drift_deg": drift}

    # Magnitude only: which side is the fast one comes from the sign of the
    # drift, below. Deriving it from the signed difference gets the two
    # directions backwards.
    difference = abs((drift / seconds) / deg_per_second_per_unit)
    stronger = throttle + difference / 2
    weaker = throttle - difference / 2
    if weaker <= 0:
        print("    -> drift too large to trim; check for a mechanical fault")
        return None

    # Scale the faster side down rather than the slower side up, which has no
    # headroom left at full throttle anyway.
    ratio = round(max(0.3, min(1.0, weaker / stronger)), 3)
    trim = (
        {"left_scale": ratio, "right_scale": 1.0}
        if drift > 0  # veered right, so the left side is the faster one
        else {"left_scale": 1.0, "right_scale": ratio}
    )
    trim["drift_deg"] = drift
    weaker = "right" if drift > 0 else "left"
    print(f"    -> {weaker} side is slower; scale the other to {ratio}")
    return trim


def measure_forward(driver) -> dict:
    """Metres per second, so blind travel between replies can be reasoned about."""
    print("\n=== Forward speed ===")
    print("The car drives straight. Mark where it starts and measure how far it got.")
    results: dict[str, float] = {}
    for throttle in FORWARD_THROTTLES:
        if not countdown(f"drive forward at {throttle} for {FORWARD_SECONDS}s", 3):
            return results
        pulse(driver, throttle, throttle, FORWARD_SECONDS)
        time.sleep(0.5)
        metres = ask_float(f"metres travelled (at {throttle})")
        if metres is None:
            return results
        if not np.isnan(metres) and metres > 0:
            speed = metres / FORWARD_SECONDS
            results[str(throttle)] = round(speed, 3)
            print(f"    -> {speed:.2f} m/s")
    return results


def report(data: dict) -> None:
    print("\n" + "=" * 62)
    print("What this means for steering")
    print("=" * 62)

    scale = data.get("scale") or {}
    pivots = data.get("pivots") or {}
    forward = data.get("forward") or {}

    if scale.get("deg_per_pixel") and pivots:
        width = scale.get("frame_width", 320)
        half = scale["deg_per_pixel"] * width / 2
        print(f"  A target at the frame edge is {half:.0f} deg off centre.")
        for differential, values in sorted(pivots.items()):
            rate = values.get("deg_per_second")
            coast = values.get("coast_deg")
            if not rate:
                continue
            seconds = half / rate
            note = ""
            if coast:
                corrected = max(0.0, (half - coast) / rate)
                note = f", or {corrected:.2f}s allowing for {coast:.0f} deg of coast"
            print(f"    at differential {differential}: turn {seconds:.2f}s{note}")
        worst = max(
            (v["coast_deg"] for v in pivots.values() if v.get("coast_deg")), default=0.0
        )
        if worst:
            print(f"\n  Coast is {worst:.0f} deg at the fastest pivot. Any turn shorter")
            print("  than that overshoots no matter what the gain is — which is the")
            print("  overshoot seen on the floor.")

    if forward:
        fastest = max(forward.values())
        print(f"\n  At {fastest:.2f} m/s the car covers {fastest * 3.4:.1f} m during a 3.4s")
        print("  model call, and half that between confirmations at concurrency 2.")

    if not (scale.get("deg_per_pixel") and pivots):
        print("  Not enough measurements yet to derive turn durations.")

    trim = data.get("trim") or {}
    stiction = data.get("stiction") or {}
    if trim or stiction:
        print("\n  Put these in the environment (run-demo.sh reads them):")
        for key, name in (
            ("left_scale", "MOTOR_LEFT_SCALE"), ("right_scale", "MOTOR_RIGHT_SCALE"),
        ):
            if trim.get(key) is not None:
                print(f"    export {name}={trim[key]}")
        for side, name in (("left", "MOTOR_LEFT_MIN"), ("right", "MOTOR_RIGHT_MIN")):
            if stiction.get(side) is not None:
                print(f"    export {name}={stiction[side]}")
        if stiction.get("left") is not None and stiction.get("right") is not None:
            gap = abs(stiction["left"] - stiction["right"])
            if gap >= 0.1:
                heavier = "right" if stiction["right"] > stiction["left"] else "left"
                print(f"\n  The {heavier} side needs {gap:.2f} more throttle to break")
                print("  away. Below that the car swings instead of easing forward, so")
                print(f"  keep BASE_SPEED above {max(stiction.values()):.2f}.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pivots", action="store_true", help="yaw rate and coast only")
    parser.add_argument("--scale", action="store_true", help="degrees per pixel only")
    parser.add_argument("--forward", action="store_true", help="forward speed only")
    parser.add_argument("--trim", action="store_true", help="straight-line trim only")
    parser.add_argument("--stiction", action="store_true", help="minimum throttle only")
    args = parser.parse_args()
    chosen = args.pivots or args.scale or args.forward or args.trim or args.stiction

    from drive import make_driver

    driver = make_driver()
    if type(driver).__name__ == "FakeDriver":
        print("DRIVER is fake — nothing will move and every number would be invented.")
        print("Re-run with DRIVER=tb6612.", file=sys.stderr)
        return 2

    print(__doc__.strip().split("\n\n")[-1])
    print(f"\nDriver: {type(driver).__name__}   Camera: {os.environ.get('CAMERA', 'fake')}")
    print("Blank skips a measurement, q aborts.\n")

    existing = {}
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
        except ValueError:
            existing = {}
    data = dict(existing)

    try:
        # Stiction first, and applied straight away. A spin in place needs BOTH
        # wheels counter-rotating; if the loaded side never breaks away the car
        # swings around a stationary wheel instead, and every angle measured
        # after that describes an arc rather than a rotation.
        if not chosen or args.stiction or args.pivots or args.trim:
            stiction = data.get("stiction") or {}
            if not chosen or args.stiction or not stiction:
                measured = measure_stiction(driver)
                if measured:
                    stiction = measured
                    data["stiction"] = measured
            if stiction:
                driver.left_min = stiction.get("left", driver.left_min)
                driver.right_min = stiction.get("right", driver.right_min)
                print(f"\n  Using minimums left={driver.left_min:.2f} "
                      f"right={driver.right_min:.2f} for everything below, so both")
                print("  wheels turn and a pivot is a real pivot.")

        if not chosen or args.pivots or args.scale:
            pivots = measure_pivots(driver)
            if pivots:
                data["pivots"] = pivots
        if not chosen or args.scale:
            from camera import make_camera
            from scenarios import SCENARIOS

            camera = make_camera(SCENARIOS["center"])
            try:
                scale = measure_scale(driver, camera, data.get("pivots") or {})
                if scale:
                    data["scale"] = scale
            finally:
                release = getattr(camera, "release", None)
                if release:
                    release()
        if not chosen or args.trim:
            trim = measure_trim(driver, data.get("pivots") or {})
            if trim:
                data["trim"] = trim
        if not chosen or args.forward:
            speeds = measure_forward(driver)
            if speeds:
                data["forward"] = speeds
    finally:
        driver.stop()

    if not data:
        print("\nNothing measured.")
        return 1

    data["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    report(data)
    print(f"\nWritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
