import os
import time
from contextlib import contextmanager

import pytest

from camera import FakeCamera
from drive import FakeDriver
from scenarios import FakeScene
from server import ControlLoop
from vision import DetectedObject, VisionProvider

MOVING_BOX = (300, 300, 700, 700)  # area 0.16 — steers, never "arrived"
ARRIVED_BOX = (100, 100, 900, 900)  # area 0.64 — over the arrival threshold


@contextmanager
def _env(**values):
    old = dict(os.environ)
    os.environ.clear()
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


class SlowVision(VisionProvider):
    def __init__(self, box, delay=0.0):
        self._box = box
        self._delay = delay

    def detect(self, target, frame):
        time.sleep(self._delay)
        return DetectedObject(box_2d=self._box, label=target)


class HangingVision(VisionProvider):
    """Answers once, then hangs — the failure the staleness deadman exists for."""

    def __init__(self, box):
        self._box = box
        self.calls = 0

    def detect(self, target, frame):
        self.calls += 1
        if self.calls > 1:
            time.sleep(30)
        return DetectedObject(box_2d=self._box, label=target)


def _wait_until(predicate, message, timeout=2.0):
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, message
        time.sleep(0.01)


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


def test_steering_outpaces_slow_inference():
    with _env(
        MOCK="true",
        CONTROL_INTERVAL="0",
        SHORT_INTERVAL="0",
        CONTROL_HZ="100",
        STALE_MIN="5",
    ):
        driver = FakeDriver()
        loop = ControlLoop(
            FakeCamera(FakeScene(boxes=[])), SlowVision(MOVING_BOX, delay=0.2), driver
        )
        loop.start("ball")
        time.sleep(0.6)
        snap = loop.snapshot()
        driving = driver.last
        loop.stop()

    assert snap["cycle"] <= 6, "inference ran faster than the mocked model latency"
    assert snap["control_cycle"] > snap["cycle"] * 5
    assert driving != (0.0, 0.0)


def test_motors_cut_when_detection_goes_stale():
    with _env(
        MOCK="true",
        CONTROL_INTERVAL="0",
        SHORT_INTERVAL="0",
        CONTROL_HZ="100",
        STALE_MIN="0.15",
        STALE_FACTOR="1",
    ):
        driver = FakeDriver()
        loop = ControlLoop(
            FakeCamera(FakeScene(boxes=[])), HangingVision(MOVING_BOX), driver
        )
        loop.start("ball")
        _wait_until(lambda: driver.last != (0.0, 0.0), "car never started driving")
        _wait_until(
            lambda: driver.last == (0.0, 0.0), "motors never cut on a stale detection"
        )
        snap = loop.snapshot()
        loop.stop()

    assert snap["status"] == "stale"
    assert snap["running"] is True, "a stale box stops the motors, not the navigation"


class ScriptedVision(VisionProvider):
    """Replays fixed boxes, then repeats the last one."""

    def __init__(self, boxes):
        self._boxes = list(boxes)
        self.calls = 0

    def detect(self, target, frame):
        box = self._boxes[min(self.calls, len(self._boxes) - 1)]
        self.calls += 1
        return None if box is None else DetectedObject(box_2d=box, label=target)


def test_one_spurious_arrival_does_not_end_the_drive():
    """The sequence a live camera produced: a full-frame box between good ones."""
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0",
        CONTROL_HZ="100", STALE_MIN="5", ARRIVE_CONFIRM="2",
    ):
        driver = FakeDriver()
        vision = ScriptedVision([MOVING_BOX, ARRIVED_BOX, MOVING_BOX, MOVING_BOX])
        loop = ControlLoop(FakeCamera(FakeScene(boxes=[])), vision, driver)
        loop.start("a chair")
        _wait_until(lambda: vision.calls >= 4, "loop stalled")
        time.sleep(0.1)
        snap = loop.snapshot()
        driving = driver.last
        loop.stop()

    assert snap["running"] is True, "a single bad frame ended the run"
    assert snap["status"] == "moving"
    assert driving != (0.0, 0.0), "car should have resumed after the bad frame"


def test_repeated_arrival_still_ends_the_drive():
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0",
        CONTROL_HZ="100", STALE_MIN="5", ARRIVE_CONFIRM="2",
    ):
        driver = FakeDriver()
        vision = ScriptedVision([MOVING_BOX, ARRIVED_BOX, ARRIVED_BOX])
        loop = ControlLoop(FakeCamera(FakeScene(boxes=[])), vision, driver)
        loop.start("a chair")
        _wait_until(lambda: not loop.snapshot()["running"], "never confirmed arrival")
        snap = loop.snapshot()

    assert snap["status"] == "arrived"
    assert driver.last == (0.0, 0.0)


def test_arrival_stops_navigation():
    with _env(MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0", CONTROL_HZ="100"):
        driver = FakeDriver()
        loop = ControlLoop(
            FakeCamera(FakeScene(boxes=[])), SlowVision(ARRIVED_BOX), driver
        )
        loop.start("ball")
        _wait_until(lambda: not loop.snapshot()["running"], "never arrived")
        snap = loop.snapshot()

    assert snap["status"] == "arrived"
    assert driver.last == (0.0, 0.0)


def test_no_motor_writes_after_stop():
    with _env(
        MOCK="true",
        CONTROL_INTERVAL="0",
        SHORT_INTERVAL="0",
        CONTROL_HZ="100",
        STALE_MIN="5",
    ):
        driver = FakeDriver()
        loop = ControlLoop(
            FakeCamera(FakeScene(boxes=[])), SlowVision(MOVING_BOX), driver
        )
        loop.start("ball")
        _wait_until(lambda: driver.last != (0.0, 0.0), "car never started driving")
        loop.stop()
        settled = len(driver.commands)
        time.sleep(0.3)

    assert driver.last == (0.0, 0.0)
    assert len(driver.commands) == settled, "a worker kept driving after stop"


def test_restart_switches_target():
    with _env(
        MOCK="true",
        CONTROL_INTERVAL="0",
        SHORT_INTERVAL="0",
        CONTROL_HZ="100",
        STALE_MIN="5",
    ):
        driver = FakeDriver()
        loop = ControlLoop(
            FakeCamera(FakeScene(boxes=[])), SlowVision(MOVING_BOX), driver
        )
        loop.start("first")
        _wait_until(lambda: driver.last != (0.0, 0.0), "car never started driving")
        loop.start("second")
        _wait_until(lambda: loop.snapshot()["infer"]["count"] > 0, "restart stalled")
        snap = loop.snapshot()
        loop.stop()

    assert snap["target"] == "second"
    assert snap["running"] is True