"""Measure whether this machine can track between model calls, and at what rate.

    .venv/bin/python bench_tracking.py

Reports the per-frame cost of everything a tracking loop would add, so the
decision to adopt it is made on this hardware's numbers rather than on an
assumption about what a Pi can do. Run it on the Pi itself.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

WIDTH, HEIGHT = 320, 240
POINTS = 80
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def textured_frame(width: int, height: int, shift: int = 0) -> np.ndarray:
    """Noise plus a shape, so there are real features to track."""
    rng = np.random.default_rng(7)
    frame = rng.integers(60, 190, (height, width, 3), dtype=np.uint8)
    cv2.circle(frame, (int(width * 0.4) + shift, height // 2), height // 5, (40, 40, 220), -1)
    cv2.rectangle(frame, (20 + shift, 20), (70 + shift, 70), (240, 240, 40), -1)
    return frame


def timed(label: str, function, runs: int = 60) -> float:
    function()  # warm caches and any lazy init
    started = time.perf_counter()
    for _ in range(runs):
        function()
    each = (time.perf_counter() - started) / runs
    print(f"  {label:38} {each * 1000:7.2f} ms")
    return each


def main() -> int:
    print(f"OpenCV {cv2.__version__}, frames {WIDTH}x{HEIGHT}\n")
    print("Per-frame costs:")

    big = textured_frame(640, 480)
    small = textured_frame(WIDTH, HEIGHT)
    ok, buf = cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    jpeg_big = buf.tobytes()
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok
    jpeg_small = buf.tobytes()

    decode_big = timed(
        "imdecode 640x480 (camera read)",
        lambda: cv2.imdecode(np.frombuffer(jpeg_big, np.uint8), cv2.IMREAD_COLOR),
    )
    timed(
        "imdecode 320x240",
        lambda: cv2.imdecode(np.frombuffer(jpeg_small, np.uint8), cv2.IMREAD_COLOR),
    )
    timed("resize 640x480 -> 320x240", lambda: cv2.resize(big, (WIDTH, HEIGHT)))
    encode = timed(
        "imencode 320x240 q70 (to the model)",
        lambda: cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70]),
    )

    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grey_next = cv2.cvtColor(textured_frame(WIDTH, HEIGHT, shift=3), cv2.COLOR_BGR2GRAY)
    to_grey = timed("cvtColor to grey", lambda: cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

    seed = timed(
        f"goodFeaturesToTrack ({POINTS} pts, on a new box)",
        lambda: cv2.goodFeaturesToTrack(grey, POINTS, 0.01, 7),
    )
    points = cv2.goodFeaturesToTrack(grey, POINTS, 0.01, 7)
    assert points is not None, "no features found"

    def flow():
        forward, status, _ = cv2.calcOpticalFlowPyrLK(grey, grey_next, points, None, **LK_PARAMS)
        # Forward-backward check: the honest way to know tracking is still good.
        back, _, _ = cv2.calcOpticalFlowPyrLK(grey_next, grey, forward, None, **LK_PARAMS)
        return forward, status, back

    track = timed("optical flow + back-check (per frame)", flow)

    print("\nWhat that adds up to:")
    per_tracked_frame = to_grey + track
    print(f"  tracking work per frame               {per_tracked_frame * 1000:7.2f} ms")
    print(f"  seeding, once per model detection     {seed * 1000:7.2f} ms")
    for hz in (10, 20, 30):
        load = per_tracked_frame * hz * 100
        print(f"  at {hz:2d} Hz that is {load:5.1f}% of one core")
    print(f"\n  current camera read (decode 640x480)  {decode_big * 1000:7.2f} ms")
    print(f"  current send path (resize + encode)   {encode * 1000:7.2f} ms")
    print(f"  cores available: {cv2.getNumberOfCPUs()}")
    print("\nTracking is worth adopting while it stays a small fraction of a core;")
    print("the camera decode is the larger cost and is already being paid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
