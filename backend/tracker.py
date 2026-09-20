"""Carry a box forward between model calls with optical flow.

The model answers every second or three. Steering on a box that old is what
makes the car weave and overshoot. So the box the model returns is used to seed
feature points, and those points are followed frame to frame locally, which
costs well under a millisecond and keeps the box roughly current.

This is only ever a short bridge. Flow drifts, and it fails outright on
occlusion, fast rotation and untextured targets, so every update reports a
confidence and the caller is expected to fall back to the model's own box
rather than steer on a tracker that has quietly lost its target.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

Box2D = Tuple[int, int, int, int]  # ymin, xmin, ymax, xmax, normalised 0-1000

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def to_pixels(box: Box2D, width: int, height: int) -> Tuple[float, float, float, float]:
    ymin, xmin, ymax, xmax = box
    return (
        xmin / 1000.0 * width, ymin / 1000.0 * height,
        xmax / 1000.0 * width, ymax / 1000.0 * height,
    )


def to_normalised(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Box2D:
    clamp = lambda value: int(max(0, min(1000, round(value))))  # noqa: E731
    return (
        clamp(y1 / height * 1000), clamp(x1 / width * 1000),
        clamp(y2 / height * 1000), clamp(x2 / width * 1000),
    )


class BoxTracker:
    """Follows one box between detections. Not a re-detector."""

    def __init__(self) -> None:
        self.max_points = int(os.environ.get("TRACK_POINTS", "80"))
        self.min_points = int(os.environ.get("TRACK_MIN_POINTS", "8"))
        # A point whose round trip lands far from where it started was not
        # really tracked; this is the standard forward-backward sanity check.
        self.fb_tolerance = _env_float("TRACK_FB_TOLERANCE", 2.0)
        self._previous: Optional[np.ndarray] = None
        self._points: Optional[np.ndarray] = None
        self._seeded_count = 0
        self._box: Optional[Tuple[float, float, float, float]] = None

    @property
    def tracking(self) -> bool:
        return self._points is not None and self._box is not None

    def reset(self) -> None:
        self._previous = None
        self._points = None
        self._seeded_count = 0
        self._box = None

    def seed(self, grey: np.ndarray, box: Box2D) -> bool:
        """Anchor to a fresh box from the model. True if it found features."""
        height, width = grey.shape[:2]
        x1, y1, x2, y2 = to_pixels(box, width, height)
        mask = np.zeros_like(grey)
        top, bottom = int(max(0, y1)), int(min(height, y2))
        left, right = int(max(0, x1)), int(min(width, x2))
        if bottom - top < 4 or right - left < 4:
            self.reset()
            return False
        mask[top:bottom, left:right] = 255
        points = cv2.goodFeaturesToTrack(
            grey, self.max_points, 0.01, 7, mask=mask
        )
        if points is None or len(points) < self.min_points:
            self.reset()
            return False
        self._previous = grey
        self._points = points
        self._seeded_count = len(points)
        self._box = (x1, y1, x2, y2)
        return True

    def update(self, grey: np.ndarray) -> Optional[Tuple[Box2D, float]]:
        """Advance the box to this frame. None once tracking is not credible."""
        if self._previous is None or self._points is None or self._box is None:
            return None

        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            self._previous, grey, self._points, None, **LK_PARAMS
        )
        if forward is None:
            self.reset()
            return None
        backward, _, _ = cv2.calcOpticalFlowPyrLK(
            grey, self._previous, forward, None, **LK_PARAMS
        )
        if backward is None:
            self.reset()
            return None

        round_trip = np.linalg.norm(
            self._points.reshape(-1, 2) - backward.reshape(-1, 2), axis=1
        )
        keep = (status.reshape(-1) == 1) & (round_trip < self.fb_tolerance)
        if int(keep.sum()) < self.min_points:
            self.reset()
            return None

        before = self._points.reshape(-1, 2)[keep]
        after = forward.reshape(-1, 2)[keep]
        shift = np.median(after - before, axis=0)

        # Scale from how the points spread about their own centre: the box has
        # to grow as the car closes on the target, or "arrived" never fires.
        spread_before = np.median(np.linalg.norm(before - before.mean(axis=0), axis=1))
        spread_after = np.median(np.linalg.norm(after - after.mean(axis=0), axis=1))
        scale = 1.0
        if spread_before > 1e-3:
            scale = float(np.clip(spread_after / spread_before, 0.8, 1.25))

        x1, y1, x2, y2 = self._box
        centre_x = (x1 + x2) / 2 + shift[0]
        centre_y = (y1 + y2) / 2 + shift[1]
        half_w = (x2 - x1) / 2 * scale
        half_h = (y2 - y1) / 2 * scale
        self._box = (centre_x - half_w, centre_y - half_h,
                     centre_x + half_w, centre_y + half_h)
        self._previous = grey
        self._points = after.reshape(-1, 1, 2)

        height, width = grey.shape[:2]
        confidence = float(keep.sum()) / max(1, self._seeded_count)
        return to_normalised(*self._box, width, height), confidence
