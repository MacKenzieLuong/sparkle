"""Ask the vision model where something is, without starting the car.

    python cli.py "the red ball"                  # one look through the camera
    python cli.py "the red ball" --image shot.jpg # ...or at a saved frame
    python cli.py "the red ball" -n 5 --save out  # sample latency, write overlays

Prints what the model returned, what the parser made of it, and what the
steering math would have done — so a bad run can be pinned on the model, the
parse, or the controller without moving a wheel. Honours MOCK, the camera and
model env vars, and the MAX_COST_USD cap.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

from camera import make_camera
from controller import command
from scenarios import SCENARIOS
from vision import is_mock, make_vision


def annotate(frame: np.ndarray, box, label: str) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]
    ymin, xmin, ymax, xmax = box
    p1 = (int(xmin / 1000 * width), int(ymin / 1000 * height))
    p2 = (int(xmax / 1000 * width), int(ymax / 1000 * height))
    cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
    cv2.putText(
        out, label, (p1[0] + 6, p1[1] + 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return out


def describe(detection, elapsed: float, raw: Optional[str], show_raw: bool) -> None:
    print(f"  {elapsed * 1000:7.0f} ms", end="  ")
    if show_raw and raw is not None:
        print(f"raw={raw.strip()!r}", end="  ")
    if detection is None:
        print("not found")
        return
    cmd = command(detection.box_2d)
    print(
        f"{detection.label!r} box={list(detection.box_2d)} "
        f"area={cmd.area_fraction:.3f} -> left={cmd.left:+.2f} "
        f"right={cmd.right:+.2f} ({cmd.status})"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-shot object detection against the configured vision model."
    )
    parser.add_argument("target", help='what to look for, e.g. "the red ball"')
    parser.add_argument("--image", help="detect against this file instead of the camera")
    parser.add_argument("-n", "--repeat", type=int, default=1, help="number of looks")
    parser.add_argument(
        "--interval", type=float, default=0.0, help="seconds to wait between looks"
    )
    parser.add_argument("--save", help="write annotated frames to SAVE-1.jpg, ...")
    parser.add_argument("--raw", action="store_true", help="print the model's raw reply")
    args = parser.parse_args(argv)

    if args.image is not None:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"cannot read image: {args.image}", file=sys.stderr)
            return 2
        camera = None
    else:
        try:
            camera = make_camera(SCENARIOS["center"])
        except Exception as exc:
            print(f"camera unavailable: {exc}", file=sys.stderr)
            return 2

    try:
        vision = make_vision(SCENARIOS["center"])
    except Exception as exc:
        print(f"vision unavailable: {exc}", file=sys.stderr)
        return 2

    source = args.image or f"camera ({type(camera).__name__})"
    print(f"target {args.target!r} via {source}, mock={is_mock()}")

    vision.start(args.target)
    found = 0
    elapsed: List[float] = []
    try:
        for attempt in range(1, args.repeat + 1):
            if camera is not None:
                try:
                    frame = camera.read()
                except Exception as exc:
                    print(f"  camera read failed: {exc}", file=sys.stderr)
                    return 2
            started = time.monotonic()
            try:
                detection = vision.detect(args.target, frame)
            except Exception as exc:
                print(f"  detect failed: {exc}", file=sys.stderr)
                return 2
            took = time.monotonic() - started
            elapsed.append(took)
            describe(detection, took, getattr(vision, "last_raw", None), args.raw)

            if detection is not None:
                found += 1
                if args.save:
                    path = f"{args.save}-{attempt}.jpg"
                    cv2.imwrite(path, annotate(frame, detection.box_2d, detection.label))
                    print(f"           wrote {path}")
            if attempt < args.repeat and args.interval:
                time.sleep(args.interval)
    finally:
        vision.stop()
        release = getattr(camera, "release", None)
        if release:
            release()

    if args.repeat > 1:
        order = sorted(elapsed)
        print(
            f"\n{found}/{args.repeat} found | "
            f"latency min {order[0] * 1000:.0f} ms "
            f"median {order[len(order) // 2] * 1000:.0f} ms "
            f"max {order[-1] * 1000:.0f} ms"
        )
    spend = getattr(vision, "estimated_spend_usd", None)
    if spend:
        cap = getattr(vision, "cost_cap_usd", 0.0)
        print(f"estimated spend ${spend:.4f} of ${cap:g} cap")

    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())
