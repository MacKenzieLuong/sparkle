from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time

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


SOI = b"\xff\xd8"  # JPEG start-of-image
EOI = b"\xff\xd9"  # JPEG end-of-image


def newest_jpeg(buf: bytearray) -> Optional[bytes]:
    """Pull the newest complete JPEG out of buf, discarding older ones.

    Frames that queued up while a consumer was busy are worthless to a moving
    car, so only the last complete one survives. Consumed bytes and any junk
    ahead of a start marker are dropped; a trailing partial frame is kept.
    """
    frame = None
    while True:
        start = buf.find(SOI)
        if start < 0:
            # A lone trailing 0xff may be the first half of the next marker.
            del buf[: max(0, len(buf) - 1)]
            return frame
        end = buf.find(EOI, start + 2)
        if end < 0:
            del buf[:start]
            return frame
        frame = bytes(buf[start : end + 2])
        del buf[: end + 2]


class RpiCamCamera(CameraProvider):
    """Read MJPEG frames from one rpicam-vid process shared by all consumers.

    A reader thread drains the pipe continuously and keeps only the newest
    frame. Reading the pipe on demand instead lets it back up behind whichever
    consumer is slowest, and the car would then steer off a frame describing
    where it used to be.
    """

    def __init__(self, width: int = 640, height: int = 480, framerate: int = 30):
        super().__init__()
        self._lock = threading.Lock()
        self._latest: Optional[bytes] = None
        self._error: Optional[str] = None
        self._closed = False
        self._command = [
            "rpicam-vid", "-t", "0", "--codec", "mjpeg",
            "--width", str(width), "--height", str(height),
            "--framerate", str(framerate), "--nopreview", "-o", "-",
        ]
        # stderr goes to a file, not a pipe: nothing drains a pipe here, and a
        # full one would wedge the camera. Without it a failure is unreadable.
        self._log = tempfile.TemporaryFile()
        try:
            self._camera = subprocess.Popen(
                self._command,
                stdout=subprocess.PIPE,
                stderr=self._log,
                bufsize=65536,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "rpicam-vid not found — install rpicam-apps, or set CAMERA to "
                "picamera2, webcam or fake"
            ) from exc
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _exit_reason(self) -> str:
        """Why the camera stopped, including what it printed on the way out."""
        code = self._camera.poll()
        head = (
            f"rpicam-vid exited with code {code}"
            if code is not None
            else "rpicam-vid stopped producing frames"
        )
        try:
            self._log.seek(0)
            output = self._log.read().decode("utf-8", "replace").strip()
        except (ValueError, OSError):
            output = ""
        if not output:
            return f"{head} (no stderr). Reproduce with: {' '.join(self._command)}"
        tail = " | ".join(line.strip() for line in output.splitlines()[-4:])
        return f"{head}: {tail}"

    def _drain(self) -> None:
        buf = bytearray()
        stream = self._camera.stdout
        if stream is None:
            with self._lock:
                self._error = "rpicam-vid stdout is unavailable"
            return
        while not self._closed:
            chunk = stream.read(65536)
            if not chunk:
                reason = self._exit_reason()
                with self._lock:
                    if not self._closed:
                        self._error = reason
                return
            buf += chunk
            frame = newest_jpeg(buf)
            if frame is not None:
                with self._lock:
                    self._latest = frame

    def read(self, timeout: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout
        while True:
            if self._closed:
                # Never hand back the last cached frame: a caller steering off
                # a frozen image would have no way to tell it had stopped.
                raise RuntimeError("rpicam-vid camera is released")
            with self._lock:
                if self._error:
                    raise RuntimeError(self._error)
                raw = self._latest
            if raw is not None:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"rpicam-vid produced no frame within {timeout}s")
            time.sleep(0.005)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("rpicam-vid produced an invalid JPEG frame")
        return self._tag_frame(frame)

    def release(self) -> None:
        self._closed = True
        if self._camera.poll() is None:
            self._camera.terminate()
        self._log.close()


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
        # Shortest queue the backend allows: a queued frame shows the car where
        # it used to be. Not every backend honours this, hence no check.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
