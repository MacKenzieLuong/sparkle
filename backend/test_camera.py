import os
import sys
import time

import numpy as np
import pytest

from camera import EOI, SOI, FakeCamera, RpiCamCamera, newest_jpeg
from scenarios import FakeScene

FAKE_RPICAM = '''\
#!{python}
"""Emits a 30fps MJPEG stream; each frame is a solid fill encoding its index."""
import sys, time
import cv2, numpy as np

n = 0
while True:
    frame = np.full((480, 640, 3), (n % 200) + 20, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    try:
        sys.stdout.buffer.write(buf.tobytes())
        sys.stdout.buffer.flush()
    except BrokenPipeError:
        break
    n += 1
    time.sleep(1 / 30)
'''


def _jpeg(payload: bytes) -> bytes:
    return SOI + payload + EOI


@pytest.fixture
def fake_rpicam(tmp_path, monkeypatch):
    script = tmp_path / "rpicam-vid"
    script.write_text(FAKE_RPICAM.format(python=sys.executable))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    return tmp_path


def test_frame_no_increments():
    cam = FakeCamera(FakeScene(boxes=[(300, 300, 700, 700)]))
    assert cam.frame_no == 0
    cam.read()
    cam.read()
    cam.read()
    assert cam.frame_no == 3


def _corner_marked_camera(monkeypatch, rotation):
    """A camera whose frames carry a bright mark in the top-left corner."""
    monkeypatch.setenv("CAMERA_ROTATION", str(rotation))

    class Marked(FakeCamera):
        def read(self):
            frame = np.zeros((40, 60, 3), dtype=np.uint8)
            frame[0:10, 0:10] = 255
            return self._tag_frame(frame)

    return Marked(FakeScene(boxes=[]))


def test_rotation_180_moves_the_mark_to_the_opposite_corner(monkeypatch):
    frame = _corner_marked_camera(monkeypatch, 180).read()
    assert frame[-1, -1].max() == 255, "mark should land bottom-right"
    assert frame[0, 0].max() == 0
    assert frame.shape == (40, 60, 3), "180 keeps the dimensions"


def test_rotation_zero_leaves_the_frame_alone(monkeypatch):
    frame = _corner_marked_camera(monkeypatch, 0).read()
    assert frame[0, 0].max() == 255


def test_rotation_90_swaps_dimensions(monkeypatch):
    frame = _corner_marked_camera(monkeypatch, 90).read()
    assert frame.shape == (60, 40, 3)


def test_newest_jpeg_discards_backlog():
    buf = bytearray(_jpeg(b"old") + _jpeg(b"newer") + _jpeg(b"newest"))
    assert newest_jpeg(buf) == _jpeg(b"newest")
    assert bytes(buf) == b"", "consumed frames should leave nothing behind"


def test_newest_jpeg_keeps_partial_tail():
    tail = SOI + b"half a frame"
    buf = bytearray(_jpeg(b"done") + tail)
    assert newest_jpeg(buf) == _jpeg(b"done")
    assert bytes(buf) == tail, "an incomplete frame must survive for the next chunk"


def test_newest_jpeg_waits_without_complete_frame():
    buf = bytearray(SOI + b"still arriving")
    assert newest_jpeg(buf) is None


def test_newest_jpeg_resyncs_past_junk():
    buf = bytearray(b"\x00\x01garbage" + _jpeg(b"good"))
    assert newest_jpeg(buf) == _jpeg(b"good")


def test_newest_jpeg_keeps_split_marker():
    """A chunk ending mid-marker must not lose the 0xff."""
    buf = bytearray(b"junk\xff")
    assert newest_jpeg(buf) is None
    buf += b"\xd8body" + EOI
    assert newest_jpeg(buf) == _jpeg(b"body")


def _frame_index(frame):
    """Recover the counter the fake stream encodes as a solid fill."""
    return int(round(float(frame.mean()))) - 20


def test_rpicam_serves_the_newest_frame_not_a_backlog(fake_rpicam):
    cam = RpiCamCamera()
    try:
        time.sleep(0.4)
        first = _frame_index(cam.read())
        time.sleep(1.0)  # ~30 frames pile up while nobody reads
        started = time.monotonic()
        second = _frame_index(cam.read())
        elapsed = time.monotonic() - started

        # ~30 frames are produced during the pause; a backlogged reader would
        # hand back the very next one. The gap between those is what matters,
        # so the bound stays loose enough for a loaded machine.
        assert (second - first) % 200 > 8, "reader is draining a stale backlog"
        assert elapsed < 0.5, "read() blocked instead of serving the newest frame"
    finally:
        cam.release()


def test_rpicam_read_fails_after_release(fake_rpicam):
    cam = RpiCamCamera()
    time.sleep(0.4)
    cam.read()
    cam.release()
    with pytest.raises(RuntimeError, match="released"):
        cam.read(timeout=0.5)