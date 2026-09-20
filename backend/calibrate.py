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

Yaw rate and coast come from the same experiment: turning for two different
durations gives angle(t) = rate*t + coast, which solves for both. Scale is
measured with optical flow rather than a protractor — the car is rotated by a
known angle and the image is asked how far it moved.

THE CAR MUST BE ON THE FLOOR with clear space around it. On blocks the wheels
turn without the body rotating and every number comes out meaningless.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "calibration.json"
PIVOT_DIFFERENTIALS = (0.25, 0.35, 0.5)
PIVOT_DURATIONS = (1.0, 2.0)
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


def measure_pivots(driver) -> dict:
    """Yaw rate and coast, from angle(t) = rate*t + coast at two durations."""
    print("\n=== Yaw rate and coast ===")
    print("The car will pivot in place. After each one, measure how far the body")
    print("rotated in degrees — a phone compass is easiest; chalk marks work too.")
    results: dict[str, dict] = {}

    for differential in PIVOT_DIFFERENTIALS:
        angles: dict[float, float] = {}
        for seconds in PIVOT_DURATIONS:
            if not countdown(
                f"pivot right at differential {differential} for {seconds}s", 3
            ):
                return results
            pulse(driver, differential, -differential, seconds)
            time.sleep(1.0)  # let it settle before the measurement is taken
            angle = ask_float(f"degrees turned ({seconds}s at {differential})")
            if angle is None:
                return results
            if not np.isnan(angle):
                angles[seconds] = abs(angle)

        if len(angles) == 2:
            (short_t, short_a), (long_t, long_a) = sorted(angles.items())
            rate = (long_a - short_a) / (long_t - short_t)
            coast = short_a - rate * short_t
            results[str(differential)] = {
                "deg_per_second": round(rate, 2),
                "coast_deg": round(max(0.0, coast), 2),
                "measured": {str(k): v for k, v in angles.items()},
            }
            print(f"    -> {rate:.1f} deg/s, coasting {max(0.0, coast):.1f} deg after stop")
        elif angles:
            seconds, angle = next(iter(angles.items()))
            results[str(differential)] = {
                "deg_per_second": round(angle / seconds, 2),
                "coast_deg": None,
                "measured": {str(seconds): angle},
            }
            print(f"    -> {angle / seconds:.1f} deg/s (coast needs both durations)")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pivots", action="store_true", help="yaw rate and coast only")
    parser.add_argument("--scale", action="store_true", help="degrees per pixel only")
    parser.add_argument("--forward", action="store_true", help="forward speed only")
    args = parser.parse_args()
    chosen = args.pivots or args.scale or args.forward

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
