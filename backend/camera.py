from __future__ import annotations

import os
import subprocess
import sys
import threading

from typing import Optional

import cv2
import numpy as np

from scenarios import FakeScene


class CameraProvider:
    def __init__(self):
        self._frame_no = 0

    @property
    def frame_no(self) -> int:
        return self._frame_no

    def read(self) -> np.ndarray:
        raise NotImplementedError

    def _tag_frame(self, frame: np.ndarray) -> np.ndarray:
        self._frame_no += 1
        return frame


class PiCamera(CameraProvider):
    def __init__(self, width: int = 640, height: int = 480):
        super().__init__()
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
        return self._tag_frame(rgb[:, :, ::-1].copy())


class RpiCamCamera(CameraProvider):
    """Read MJPEG frames from one rpicam-vid process shared by all consumers."""

    def __init__(self, width: int = 640, height: int = 480, framerate: int = 30):
        super().__init__()
        self._lock = threading.Lock()
        self._camera = subprocess.Popen(
            [
                "rpicam-vid", "-t", "0", "--codec", "mjpeg",
                "--width", str(width), "--height", str(height),
                "--framerate", str(framerate), "--nopreview", "-o", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def read(self) -> np.ndarray:
        with self._lock:
            raw = self._read_jpeg()
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("rpicam-vid produced an invalid JPEG frame")
        return self._tag_frame(frame)

    def _read_jpeg(self) -> bytes:
        if self._camera.stdout is None:
            raise RuntimeError("rpicam-vid stdout is unavailable")
        stream = self._camera.stdout
        while True:
            if stream.read(2) == b"\xff\xd8":
                break
            if self._camera.poll() is not None:
                raise RuntimeError("rpicam-vid exited while waiting for a frame")
        frame = bytearray(b"\xff\xd8")
        while True:
            byte = stream.read(1)
            if not byte:
                raise RuntimeError("rpicam-vid ended while reading a frame")
            frame += byte
            if frame[-2:] == b"\xff\xd9":
                return bytes(frame)

    def release(self) -> None:
        if self._camera.poll() is None:
            self._camera.terminate()


class WebcamCamera(CameraProvider):
    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        api_preference: Optional[int] = cv2.CAP_ANY,
    ):
        super().__init__()
        self._lock = threading.Lock()
        self._cap = cv2.VideoCapture(index, api_preference)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"webcam {index} not opened — check CAMERA_INDEX and that "
                "the terminal has camera permission (macOS: System Settings → "
                "Privacy & Security → Camera)"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray:
        with self._lock:
            ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("webcam read failed")
        return self._tag_frame(frame)

    def release(self) -> None:
        with self._lock:
            self._cap.release()


class FakeCamera(CameraProvider):
    def __init__(self, scene: FakeScene, width: int = 640, height: int = 480):
        super().__init__()
        self._scene = scene
        self._width = width
        self._height = height
        self._lock = threading.Lock()

    def read(self) -> np.ndarray:
        with self._lock:
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
            return self._tag_frame(frame)


def make_camera(scene: FakeScene) -> CameraProvider:
    provider = os.environ.get("CAMERA", "fake").lower()
    width = int(os.environ.get("CAMERA_WIDTH", "640"))
    height = int(os.environ.get("CAMERA_HEIGHT", "480"))
    if provider == "picamera2":
        return PiCamera(width=width, height=height)
    if provider in ("rpicam", "rpicam-vid"):
        return RpiCamCamera(
            width=width,
            height=height,
            framerate=int(os.environ.get("CAMERA_FRAMERATE", "30")),
        )
    if provider in ("webcam", "usb", "v4l2"):
        index = int(os.environ.get("CAMERA_INDEX", "0"))
        return WebcamCamera(
            index=index, width=width, height=height,
            api_preference=cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY,
        )
    return FakeCamera(scene=scene, width=width, height=height)
