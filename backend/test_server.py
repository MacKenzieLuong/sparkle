import os
import time

import pytest

from scenarios import FakeScene
from server import ControlLoop


def _loop(**env):
    old = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        from camera import FakeCamera
        from drive import FakeDriver
        from vision import FakeVision

        scene = FakeScene(boxes=[])
        return ControlLoop(FakeCamera(scene), FakeVision(scene), FakeDriver())
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_pause_uses_short_interval_when_near():
    loop = _loop(CONTROL_INTERVAL="5", SHORT_INTERVAL="1", SHORT_INTERVAL_AREA="0.15")
    assert loop._pause(0.2) == 1.0
    assert loop._pause(0.15) == 1.0
    assert loop._pause(0.14) == 5.0
    assert loop._pause(None) == 5.0


def test_pause_defaults():
    loop = _loop()
    assert loop._pause(None) == 5.0
    assert loop._pause(0.0) == 5.0
    assert loop._pause(0.5) == 1.0


def test_infer_recorded_with_frame_no():
    old = dict(os.environ)
    os.environ.clear()
    os.environ.update(
        MOCK="true",
        CAMERA="fake",
        DRIVER="fake",
        CONTROL_INTERVAL="0.01",
    )
    try:
        from camera import FakeCamera
        from drive import FakeDriver
        from vision import FakeVision

        scene = FakeScene(boxes=[(300, 300, 700, 700)])
        cam = FakeCamera(scene)
        loop = ControlLoop(cam, FakeVision(scene), FakeDriver())
        loop.start("ball")
        deadline = time.time() + 2.0
        while loop.snapshot()["infer"]["count"] < 2:
            assert time.time() < deadline, "loop did not run inference"
            time.sleep(0.01)
        loop.stop()
        stats = loop.snapshot()
        assert stats["infer"]["count"] >= 2
        assert stats["infer"]["frame_no"] is not None
        assert stats["infer"]["frame_no"] > 0
        assert cam.frame_no >= stats["infer"]["frame_no"]
        assert stats["infer"]["label"] == "ball"
        assert stats["infer"]["box_2d"] == [300, 300, 700, 700]
    finally:
        os.environ.clear()
        os.environ.update(old)