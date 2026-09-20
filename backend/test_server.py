import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import cv2
import numpy as np
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


class OneDecisionVision(VisionProvider):
    """One decision of the given action, then silence (never goes stale)."""

    def __init__(self, action, box=None):
        self._action = action
        self._box = box
        self.calls = 0

    def detect(self, target, frame):
        self.calls += 1
        if self.calls > 1:
            time.sleep(30)
        return DetectedObject(box_2d=self._box, label=target, action=self._action)


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


def test_app_is_built_once_per_process():
    """Each build opens a camera, and a real one cannot be acquired twice."""
    import server

    assert server.get_app() is server.app
    assert server.get_app() is server.get_app()


def test_server_startup_opens_one_camera():
    """Importing server and asking for the app must not open a second camera.

    A real camera cannot be acquired twice: the second rpicam-vid dies with
    "Pipeline handler in use by another process".
    """
    code = (
        "import camera\n"
        "opened = []\n"
        "_real = camera.make_camera\n"
        "def counting(scene):\n"
        "    opened.append(1)\n"
        "    return _real(scene)\n"
        "camera.make_camera = counting\n"
        "import server\n"          # builds app at import
        "server.get_app()\n"       # what __main__ does
        "print('cameras=', len(opened))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env={**os.environ, "MOCK": "true", "CAMERA": "fake", "DRIVER": "fake"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "cameras= 1" in result.stdout, result.stdout


def test_steering_outpaces_slow_inference():
    with _env(
        MOCK="true",
        CONTROL_INTERVAL="0",
        SHORT_INTERVAL="0",
        CONTROL_HZ="100",
        STALE_AFTER="30",
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


class RecordingVision(VisionProvider):
    """Notes when each request's frame was taken, and replies unevenly.

    Uneven on purpose: with equal latencies a startup stagger survives by
    accident, so the drift only shows when the workers' cycles differ — which
    live they always do, by the 1.8s between a fast reply and a slow one.
    """

    def __init__(self, box, delays):
        self._box = box
        self._delays = list(delays)
        self._lock = threading.Lock()
        self.captured_at = []

    def detect(self, target, frame):
        with self._lock:
            index = len(self.captured_at)
            self.captured_at.append(time.monotonic())
        time.sleep(self._delays[index % len(self._delays)])
        return DetectedObject(box_2d=self._box, label=target)


def test_inference_frames_stay_spread_out_across_the_cycle():
    """Frames must sample the whole cycle, not bunch into one instant.

    The workers were staggered once at startup and then paced by a fixed sleep
    after each reply, so a slow call pushed that worker's next frame late and
    within a few cycles they converged — two requests firing together and
    nothing looked at in between.
    """
    spacing = 0.3
    with _env(
        MOCK="true",
        CONTROL_INTERVAL="0",
        SHORT_INTERVAL="0",
        CONTROL_HZ="50",
        STALE_AFTER="30",
        TRACK="false",
        VISION_CONCURRENCY="2",
        VISION_SPACING=str(spacing),
    ):
        vision = RecordingVision(MOVING_BOX, delays=[0.12, 0.5])
        loop = ControlLoop(FakeCamera(FakeScene(boxes=[])), vision, FakeDriver())
        loop.start("ball")
        time.sleep(3.0)
        loop.stop()
        taken = sorted(vision.captured_at)

    gaps = [later - earlier for earlier, later in zip(taken, taken[1:])]
    assert len(gaps) >= 5, f"too few frames to judge spread: {len(taken)}"
    assert min(gaps) >= spacing / 2, (
        f"frames bunched together: {[round(g, 3) for g in gaps]}"
    )


class SceneVision(VisionProvider):
    """Replies with wherever the scene's box currently is, slowly.

    The camera renders the same box, so image and detection move together and
    the tracker has something real to follow between replies.
    """

    def __init__(self, scene, delay=0.0):
        self._scene = scene
        self._delay = delay

    def detect(self, target, frame):
        time.sleep(self._delay)
        return DetectedObject(box_2d=self._scene.peek(), label=target)


def test_closing_error_reads_as_a_negative_turn_rate():
    """The lead term is silent unless the damper survives between replies.

    Its whole failure mode is quiet: reset the damper too eagerly — on every
    reply, or every control tick — and the rate reads zero forever, the lead
    contributes nothing, and the car overshoots exactly as it did before, with
    nothing in the logs to say so. This asserts the rate really is measured,
    end to end, with the sign that damps rather than amplifies.
    """
    scene = FakeScene(boxes=[(400, 810, 600, 950)])
    with _env(
        MOCK="true",
        CONTROL_INTERVAL="0",
        SHORT_INTERVAL="0",
        CONTROL_HZ="50",
        STALE_AFTER="30",
        TRACK="true",
        TRACK_HZ="50",
        TRACK_MAX_AGE="10",
    ):
        loop = ControlLoop(FakeCamera(scene), SceneVision(scene, delay=0.5), FakeDriver())
        loop.start("ball")
        time.sleep(0.4)
        rates = []
        for centre in range(880, 620, -20):
            scene.set_override((400, centre - 70, 600, centre + 70))
            time.sleep(0.05)
            cmd = loop.snapshot().get("last_command") or {}
            if cmd.get("turn_rate") is not None:
                rates.append(cmd["turn_rate"])
        loop.stop()

    assert rates, "the loop never issued a command to read a rate from"
    assert any(rate < 0 for rate in rates), (
        f"a target closing on the centre must read as a negative rate: {rates}"
    )


class JitteryVision(VisionProvider):
    """Alternates fast and slow replies, like the live model does."""

    def __init__(self, box, delays):
        self._box = box
        self._delays = list(delays)
        self.calls = 0

    def detect(self, target, frame):
        time.sleep(self._delays[min(self.calls, len(self._delays) - 1)])
        self.calls += 1
        return DetectedObject(box_2d=self._box, label=target)


def test_a_slow_call_after_fast_ones_does_not_cut_the_motors():
    """Live latency swung 2.5-4.3s; the window must tolerate that jitter."""
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0",
        CONTROL_HZ="100", STALE_AFTER="1.0",
    ):
        driver = FakeDriver()
        vision = JitteryVision(MOVING_BOX, [0.05, 0.05, 0.05, 0.4, 0.05])
        loop = ControlLoop(FakeCamera(FakeScene(boxes=[])), vision, driver)
        loop.start("a chair")
        _wait_until(lambda: vision.calls >= 5, "loop stalled", timeout=5.0)
        commands = list(driver.commands)
        loop.stop()

    # Zeros before the first box are the acquiring phase, not the deadman.
    driving = next(i for i, c in enumerate(commands) if c != (0.0, 0.0))
    cuts = [c for c in commands[driving:] if c == (0.0, 0.0)]
    assert not cuts, f"deadman fired on latency jitter: {len(cuts)} cuts"


def test_motors_cut_when_detection_goes_stale():
    with _env(
        MOCK="true",
        CONTROL_INTERVAL="0",
        SHORT_INTERVAL="0",
        CONTROL_HZ="100",
        STALE_AFTER="0.15",
    ):
        driver = FakeDriver()
        loop = ControlLoop(
            FakeCamera(FakeScene(boxes=[])), HangingVision(MOVING_BOX), driver
        )
        loop.start("ball")
        _wait_until(lambda: driver.last != (0.0, 0.0), "car never started driving")
        # Wait on the status, not the throttle: the control thread stops the
        # motors first and records why second, so the throttle reaching zero
        # does not yet mean the state has caught up.
        _wait_until(
            lambda: loop.snapshot()["status"] == "stale",
            "motors never cut on a stale detection",
        )
        snap = loop.snapshot()
        loop.stop()

    assert driver.last == (0.0, 0.0)
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
        CONTROL_HZ="100", STALE_AFTER="30", ARRIVE_CONFIRM="2",
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
        CONTROL_HZ="100", STALE_AFTER="30", ARRIVE_CONFIRM="2",
    ):
        driver = FakeDriver()
        vision = ScriptedVision([MOVING_BOX, ARRIVED_BOX, ARRIVED_BOX])
        loop = ControlLoop(FakeCamera(FakeScene(boxes=[])), vision, driver)
        loop.start("a chair")
        _wait_until(lambda: not loop.snapshot()["running"], "never confirmed arrival")
        snap = loop.snapshot()

    assert snap["status"] == "arrived"
    assert driver.last == (0.0, 0.0)


class PlanVision(VisionProvider):
    """Replays a scripted sequence of model decisions.

    The delay stands in for model latency; without it perception outruns the
    control thread and a decision can expire before it is ever executed.
    """

    def __init__(self, plans, delay=0.05):
        self._plans = list(plans)
        self._delay = delay
        self.calls = 0

    def detect(self, target, frame):
        time.sleep(self._delay)
        action, box = self._plans[min(self.calls, len(self._plans) - 1)]
        self.calls += 1
        return DetectedObject(box_2d=box, label=target, action=action)


def test_car_searches_then_drives_when_the_model_finds_it():
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0",
        CONTROL_HZ="100", STALE_AFTER="30", SEARCH_LIMIT="8",
        SEARCH_SPEED="0.25", BASE_SPEED="0.5",
    ):
        driver = FakeDriver()
        vision = PlanVision(
            [("search_right", None), ("search_right", None), ("approach", MOVING_BOX)]
        )
        loop = ControlLoop(FakeCamera(FakeScene(boxes=[])), vision, driver)
        loop.start("a chair")
        _wait_until(
            lambda: driver.last[0] > 0 > driver.last[1], "never rotated to search"
        )
        _wait_until(
            lambda: driver.last[0] > 0 and driver.last[1] > 0, "never drove forward"
        )
        snap = loop.snapshot()
        loop.stop()

    assert snap["running"] is True
    assert snap["action"] == "approach"
    assert snap["status"] == "moving"


def test_endless_search_gives_up():
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0",
        CONTROL_HZ="100", STALE_AFTER="30", SEARCH_LIMIT="4",
    ):
        driver = FakeDriver()
        vision = PlanVision([("search_left", None)])
        loop = ControlLoop(FakeCamera(FakeScene(boxes=[])), vision, driver)
        loop.start("a chair")
        _wait_until(lambda: not loop.snapshot()["running"], "search never gave up")
        snap = loop.snapshot()

    assert snap["status"] == "target_lost"
    assert snap["search_streak"] >= 4
    assert driver.last == (0.0, 0.0)


def test_model_stop_needs_confirming_too():
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0",
        CONTROL_HZ="100", STALE_AFTER="30", ARRIVE_CONFIRM="2",
    ):
        driver = FakeDriver()
        vision = PlanVision(
            [("approach", MOVING_BOX), ("stop", None), ("approach", MOVING_BOX),
             ("approach", MOVING_BOX)]
        )
        loop = ControlLoop(FakeCamera(FakeScene(boxes=[])), vision, driver)
        loop.start("a chair")
        _wait_until(lambda: vision.calls >= 4, "loop stalled")
        snap = loop.snapshot()
        loop.stop()

    assert snap["running"] is True, "a lone stop ended the run"


def test_pipelining_raises_the_decision_rate():
    """Three requests in flight should deliver decisions ~3x as often."""
    def rate(concurrency):
        with _env(
            MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0", CONTROL_HZ="100",
            STALE_AFTER="30", VISION_CONCURRENCY=str(concurrency), VISION_STAGGER="0.1",
        ):
            loop = ControlLoop(
                FakeCamera(FakeScene(boxes=[])),
                SlowVision(MOVING_BOX, delay=0.3),
                FakeDriver(),
            )
            loop.start("a chair")
            time.sleep(1.2)
            cycles = loop.snapshot()["cycle"]
            loop.stop()
            return cycles

    single = rate(1)
    triple = rate(3)
    assert triple > single * 1.8, f"pipelining gained nothing: {single} -> {triple}"


def test_out_of_order_replies_never_overwrite_a_newer_one():
    """Concurrent requests finish out of order; the older one must be dropped."""
    with _env(MOCK="true", STALE_AFTER="30"):
        loop = ControlLoop(
            FakeCamera(FakeScene(boxes=[])), SlowVision(MOVING_BOX), FakeDriver()
        )
        loop.start("a chair")
        newer = DetectedObject(box_2d=MOVING_BOX, label="newer", action="approach")
        older = DetectedObject(box_2d=MOVING_BOX, label="older", action="approach")
        epoch = loop._epoch

        assert loop._publish(epoch, newer, 100.0, 1, None) is None
        assert loop._publish(epoch, older, 90.0, 2, None) is None
        snap = loop.snapshot()
        loop.stop()

    assert snap["infer"]["label"] == "newer", "a stale reply replaced a fresher one"
    assert snap["out_of_order"] == 1
    assert snap["cycle"] == 1, "the dropped reply should not count as a cycle"


class TrackableCamera(FakeCamera):
    """A textured scene whose target slides right, so flow has work to do."""

    def __init__(self, scene):
        super().__init__(scene)
        self.shift = 0

    def read(self):
        rng = np.random.default_rng(5)
        frame = rng.integers(60, 190, (240, 320, 3), dtype=np.uint8)
        centre = (110 + self.shift, 120)
        cv2.circle(frame, centre, 40, (30, 30, 30), -1)
        for offset in range(-34, 34, 7):
            cv2.line(frame, (centre[0] + offset, 86), (centre[0] + offset, 154),
                     (220, 220, 220), 2)
        self.shift = min(self.shift + 2, 120)
        return self._tag_frame(frame)


def test_tracking_keeps_the_steering_box_current():
    """Between model replies the box should age in ms, not seconds."""
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0", CONTROL_HZ="50",
        STALE_AFTER="30", TRACK="true", TRACK_HZ="30",
    ):
        camera = TrackableCamera(FakeScene(boxes=[]))
        loop = ControlLoop(camera, SlowVision((330, 220, 670, 470), 1.0), FakeDriver())
        loop.start("a chair")
        _wait_until(
            lambda: loop.snapshot()["detection_age"] is not None, "never steered"
        )
        time.sleep(0.8)  # well inside one 1.0s model call
        snap = loop.snapshot()
        loop.stop()

    assert snap["tracking"] not in (None, "lost"), f"tracker never held: {snap['tracking']}"
    assert snap["detection_age"] < 0.3, (
        f"steering on a {snap['detection_age']}s-old box; tracking is not being used"
    )
    assert snap["model_age"] > snap["detection_age"], "model box should be the older one"


def test_tracking_is_not_trusted_past_its_horizon():
    """The failure seen on the floor: it kept driving at a box nothing confirmed.

    Flow cannot report that a target has gone, so a seed is only followed for
    TRACK_MAX_AGE before the model has to say the target is still there.
    """
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0", CONTROL_HZ="50",
        STALE_AFTER="30", TRACK="true", TRACK_HZ="30", TRACK_MAX_AGE="0.3",
    ):
        loop = ControlLoop(
            TrackableCamera(FakeScene(boxes=[])),
            SlowVision((330, 220, 670, 470), 2.0),  # one slow reply, then silence
            FakeDriver(),
        )
        loop.start("a chair")
        _wait_until(
            lambda: loop.snapshot()["detection_age"] is not None, "never steered",
            timeout=5.0,
        )
        time.sleep(0.6)  # past the 0.3s horizon, still inside the 2s model call
        snap = loop.snapshot()
        loop.stop()

    assert snap["detection_age"] >= 0.3, (
        "still steering on a tracked box past its horizon: "
        f"age {snap['detection_age']}s"
    )


def test_control_falls_back_to_the_model_box_when_tracking_is_off():
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0", CONTROL_HZ="50",
        STALE_AFTER="30", TRACK="false",
    ):
        loop = ControlLoop(
            FakeCamera(FakeScene(boxes=[])), SlowVision(MOVING_BOX, 0.5), FakeDriver()
        )
        loop.start("a chair")
        _wait_until(lambda: loop.snapshot()["detection_age"] is not None, "never steered")
        snap = loop.snapshot()
        loop.stop()

    assert snap["tracking"] is None, "tracker should not run when TRACK=false"
    assert snap["detection_age"] == snap["model_age"]


RIGHT_BOX = (300, 750, 700, 950)  # dx 0.7 — a hard turn the budget must bound
_DT = 0.01  # CONTROL_HZ=100


def test_one_decision_turns_only_inside_its_rotation_budget():
    """A single reply keeps the box; the car must not keep rotating forever.

    Budget in throttle-seconds; once it is spent the wheels equalise and the
    car coasts straight at the same forward speed until the next reply.
    """
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0", CONTROL_HZ="100",
        STALE_AFTER="30", VISION_CONCURRENCY="1", ROTATION_BUDGET="0.1",
    ):
        driver = FakeDriver()
        loop = ControlLoop(
            FakeCamera(FakeScene(boxes=[])),
            OneDecisionVision("approach", RIGHT_BOX), driver,
        )
        loop.start("a chair")
        _wait_until(lambda: driver.last != (0.0, 0.0), "car never started turning")
        time.sleep(1.0)  # well past the 0.21s it takes to exhaust the budget
        snap = loop.snapshot()
        commands = list(driver.commands)
        loop.stop()

    turned = sum(abs((l - r) / 2) * _DT for l, r in commands)
    assert 0.08 <= turned <= 0.11, f"turned {turned:.3f} throttle-seconds, budget 0.1"
    assert snap["rotation_spent"] <= 0.11
    assert snap["rotation_unit"] == "throttle-seconds", "no calibration -> fallback unit"
    assert snap["deg_per_turn_second"] is None, "no calibration -> no degree rate"
    coasts = [c for c in commands if c[0] == c[1] and c[0] > 0]
    assert coasts, "never coasted straight after the budget ran out"
    assert coasts[-1] == pytest.approx((0.42, 0.42)), (
        "coast must keep the forward speed rather than stopping"
    )


def test_new_decision_refills_the_rotation_budget():
    """A fresh reply re-arms the budget: the car may turn again after a coast."""
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0", CONTROL_HZ="100",
        STALE_AFTER="30", VISION_CONCURRENCY="1", ROTATION_BUDGET="0.1",
    ):
        driver = FakeDriver()
        vision = PlanVision(
            [("search_right", None), ("approach", RIGHT_BOX)], delay=0.9
        )
        loop = ControlLoop(FakeCamera(FakeScene(boxes=[])), vision, driver)
        loop.start("a chair")

        def has_second_turn():
            cmds = driver.commands
            try:
                first = next(i for i, c in enumerate(cmds) if c[0] != c[1])
                coast = next(
                    i for i in range(first, len(cmds)) if cmds[i][0] == cmds[i][1]
                )
                next(i for i in range(coast, len(cmds)) if cmds[i][0] != cmds[i][1])
                return True
            except StopIteration:
                return False

        _wait_until(has_second_turn, "second decision never turned again", timeout=6.0)
        time.sleep(0.5)  # the second turn should also coast once its budget is spent
        commands = list(driver.commands)
        loop.stop()

    assert any(
        c[0] == c[1] and c[0] > 0 for c in commands
    ), "the second decision over-turned: it never coasted straight again"


def test_calibrated_loop_limits_rotation_in_degrees(tmp_path):
    """With calibration.json the budget is an angle, not throttle-time.

    k = 25 deg/s at differential 0.25 -> 100 deg/s per turn-throttle. A hard
    turn then spends 6 deg and stops; the same test without calibration would
    spend throttle-seconds instead and report a different unit.
    """
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        '{"pivots": {"0.25": {"deg_per_second": 25.0}}}'
    )
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0", CONTROL_HZ="100",
        STALE_AFTER="30", VISION_CONCURRENCY="1",
        CALIBRATION_FILE=str(calibration), ROTATION_BUDGET_DEG="6",
    ):
        driver = FakeDriver()
        loop = ControlLoop(
            FakeCamera(FakeScene(boxes=[])),
            OneDecisionVision("approach", RIGHT_BOX), driver,
        )
        assert loop._rotation_unit == "deg"
        assert loop._deg_per_turn_second == pytest.approx(100.0)
        loop.start("a chair")
        _wait_until(lambda: driver.last != (0.0, 0.0), "car never started turning")
        time.sleep(1.0)  # well past the ~0.13s it takes to spend 6 deg
        snap = loop.snapshot()
        commands = list(driver.commands)
        loop.stop()

    turned = sum(abs((l - r) / 2) * _DT * 100.0 for l, r in commands)
    assert 5.5 <= turned <= 6.5, f"rotated {turned:.2f} deg, budget 6"
    assert snap["rotation_spent"] <= 6.5
    assert snap["rotation_unit"] == "deg"
    assert snap["deg_per_turn_second"] == pytest.approx(100.0)
    coasts = [c for c in commands if c[0] == c[1] and c[0] > 0]
    assert coasts, "never coasted straight after the degree budget ran out"


def test_run_stops_at_the_time_limit():
    """The backstop for a drive nobody can reach to stop."""
    with _env(
        MOCK="true", CONTROL_INTERVAL="0", SHORT_INTERVAL="0",
        CONTROL_HZ="100", STALE_AFTER="30", MAX_RUN_SECONDS="0.4",
    ):
        driver = FakeDriver()
        loop = ControlLoop(
            FakeCamera(FakeScene(boxes=[])), SlowVision(MOVING_BOX, delay=0.02), driver
        )
        loop.start("a chair")
        _wait_until(lambda: driver.last != (0.0, 0.0), "car never started driving")
        _wait_until(
            lambda: not loop.snapshot()["running"], "ran past the limit", timeout=3.0
        )
        snap = loop.snapshot()

    assert snap["status"] == "time_limit"
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
        STALE_AFTER="30",
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
        STALE_AFTER="30",
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