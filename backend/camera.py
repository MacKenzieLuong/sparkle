from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from scenarios import FakeScene


class CameraProvider:
    def read(self) -> np.ndarray:
        raise NotImplementedError


class PiCamera(CameraProvider):
    def __init__(self, width: int = 640, height: int = 480):
        from picamera2 import Picamera2

        self._lock = threading.Lock()
        self._picam2 = Picamera2()
        config = self._picam2.create_still_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._picam2.configure(config)
        self._picam2.start()

    def read(self) -> np.ndarray:
        with self._lock:
            rgb = self._picam2.capture_array()
        return rgb[:, :, ::-1].copy()


class FakeCamera(CameraProvider):
    def __init__(self, scene: FakeScene, width: int = 640, height: int = 480):
        self._scene = scene
        self._width = width
        self._height = height

    def read(self) -> np.ndarray:
        frame = np.full((self._height, self._width, 3), 40, dtype=np.uint8)
        cv2.line(
            frame,
            (self._width // 2, 0),
            (self._width // 2, self._height),
            (0, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            "SIMULATED CAMERA",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        box = self._scene.peek()
        if box is not None:
            ymin, xmin, ymax, xmax = box
            x1 = int(xmin / 1000 * self._width)
            y1 = int(ymin / 1000 * self._height)
            x2 = int(xmax / 1000 * self._width)
            y2 = int(ymax / 1000 * self._height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                frame,
                "TARGET",
                (x1 + 6, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "NO TARGET",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )
        return frame


def make_camera(scene: FakeScene) -> CameraProvider:
    provider = os.environ.get("CAMERA", "fake").lower()
    width = int(os.environ.get("CAMERA_WIDTH", "640"))
    height = int(os.environ.get("CAMERA_HEIGHT", "480"))
    if provider == "picamera2":
        return PiCamera(width=width, height=height)
    return FakeCamera(scene=scene, width=width, height=height)