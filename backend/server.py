from __future__ import annotations

import os
import asyncio
import re
import uuid
from contextlib import asynccontextmanager
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2

from fastapi import Request
from pydantic import BaseModel, Field
from speech import SpeechService, validate_wav, MAX_BYTES, REJECTION

import controller
from camera import CameraProvider, FakeCamera, make_camera
from controller import (
    ROTATION_BUDGET,
    ROTATION_BUDGET_DEG,
    TurnDamper,
    act,
    calibration_deg_per_turn_second,
    command,
    dx_of,
    enforce_turn_budget,
    enforce_turn_budget_deg,
    turn_authority,
)
from drive import Driver, FakeDriver, make_driver
from scenarios import SCENARIOS, FakeScene
from tracker import BoxTracker
from vision import (
    DetectedObject,
    FakeVision,
    LocalVision,
    OmniVision,
    VisionProvider,
    is_mock,
    make_vision,
)

BASE_DIR = Path(__file__).resolve().parent


class DirectRequest(BaseModel):
    target: str = Field(min_length=1, max_length=200)
    commandId: Optional[str] = Field(default=None, max_length=100)
    sessionId: Optional[str] = Field(default=None, max_length=100)
    expectedRevision: Optional[int] = None
    resume: bool = False


class ResumeRequest(BaseModel):
    expectedRevision: int


class SessionRequest(BaseModel):
    sessionId: str = Field(min_length=1, max_length=100)


class ScenarioRequest(BaseModel):
    name: str


class BoxRequest(BaseModel):
    box: Optional[list] = None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _blank_infer() -> dict:
    return {
        "count": 0, "frame_no": None, "label": None,
        "box_2d": None, "action": None, "reason": None,
    }


class ControlLoop:
    """Perception and steering run on separate threads.

    A model call takes seconds; steering must not. The perception thread asks
    the model what to do and publishes its latest decision; the control thread
    executes that decision at a fixed rate, cutting the motors once it is older
    than STALE_AFTER. The model chooses the action, this file chooses when to
    stop obeying it.
    """

    def __init__(self, camera: CameraProvider, vision: VisionProvider, driver: Driver):
        self._camera = camera
        self._vision = vision
        self._driver = driver
        self._interval = _env_float("CONTROL_INTERVAL", 5.0)
        self._short_interval = _env_float("SHORT_INTERVAL", 1.0)
        self._short_interval_area = _env_float("SHORT_INTERVAL_AREA", 0.15)
        self._control_period = 1.0 / max(_env_float("CONTROL_HZ", 10.0), 0.1)
        # The longest the car may drive on a single box. Deliberately absolute:
        # scaling it off the measured cycle cut the motors on every slow call,
        # because a call is only known to be slow once it has already returned.
        self._stale_after = _env_float("STALE_AFTER", 8.0)
        self._miss_limit = 3
        self._arrive_confirm = int(os.environ.get("ARRIVE_CONFIRM", "2"))
        self._search_limit = int(os.environ.get("SEARCH_LIMIT", "8"))
        # Requests in flight at once. One model call takes seconds, so the only
        # way to hear from it more often over HTTP is to have several running.
        # Costs one full call per worker per cycle.
        self._concurrency = max(1, int(os.environ.get("VISION_CONCURRENCY", "1")))
        self._stagger = _env_float("VISION_STAGGER", 1.2)
        # Minimum gap between successive inference frames, across all workers.
        # Defaults to the old startup stagger, but is now held for the whole
        # run rather than decaying after the first few cycles.
        self._frame_spacing = _env_float("VISION_SPACING", self._stagger)
        self._slot_at: Optional[float] = None
        # Local tracking bridges the gap between model replies, so the box the
        # car steers on is tens of milliseconds old rather than seconds.
        self._track_hz = _env_float("TRACK_HZ", 20.0)
        self._track_enabled = os.environ.get("TRACK", "true").lower() in (
            "1", "true", "yes",
        )
        self._track_min_confidence = _env_float("TRACK_MIN_CONFIDENCE", 0.4)
        # How long a seed may be followed before the model has to confirm the
        # target again. Flow cannot tell you the target is gone, so without a
        # horizon the car drives at whatever the points drifted onto.
        self._track_max_age = _env_float("TRACK_MAX_AGE", 1.5)
        # Last resort: if the network drops, /stop is unreachable and nothing
        # else bounds a drive that never arrives.
        self._max_run_seconds = _env_float("MAX_RUN_SECONDS", 120.0)
        self._lease_seconds = max(1.0, _env_float("HEARTBEAT_TIMEOUT", 3.0))
        self._lock = threading.RLock()
        self._epoch = 0
        self._heartbeat = time.monotonic()
        self._receipts = {}
        self._latest: Optional[DetectedObject] = None
        self._latest_at: Optional[float] = None
        self._cycle_time: Optional[float] = None
        self._started_at = 0.0
        self._published_at: Optional[float] = None
        self._previous_publish: Optional[float] = None
        self._arrived_streak = 0
        self._search_streak = 0
        self._last_terminal: Optional[str] = None
        self._tracker = BoxTracker()
        self._tracked: Optional[DetectedObject] = None
        self._tracked_at: Optional[float] = None
        self._tracked_seed: Optional[float] = None
        # One decision may command only so much rotation before the car coasts
        # straight; re-armed on every accepted reply. With calibration.json the
        # budget is an angle; without it, throttle-seconds is a best-effort
        # stand-in that never claims to know degrees.
        self._deg_per_turn_second = calibration_deg_per_turn_second()
        self._rotation_unit = "deg" if self._deg_per_turn_second else "throttle-seconds"
        self._rotation_budget = (
            _env_float("ROTATION_BUDGET_DEG", ROTATION_BUDGET_DEG)
            if self._deg_per_turn_second
            else _env_float("ROTATION_BUDGET", ROTATION_BUDGET)
        )
        self._rotation_spent = 0.0
        self._budget_seed_at: Optional[float] = None
        # The lead term's derivative source. Touched only by the control
        # thread, and fed from whichever box that thread actually steers on.
        self._damper = TurnDamper()
        self._damper_stream: Optional[Tuple[bool, Optional[float]]] = None
        # Pulse mode: move only in response to an inference, one short burst
        # per decision, then stop until the next one arrives. Trades speed
        # for never travelling on a guess -- see the comment in _control.
        self._pulse_mode = os.environ.get("PULSE_MODE", "false").lower() in (
            "1", "true", "yes",
        )
        self._pulse_seconds = _env_float("PULSE_SECONDS", 0.25)
        self._pulse_until: Optional[float] = None
        self._pulse_for: Optional[float] = None
        self.state = {
            "running": False,
            "target": None,
            "status": "idle",
            "cycle": 0,
            "control_cycle": 0,
            "missed": 0,
            "arrived_streak": 0,
            "search_streak": 0,
            "out_of_order": 0,
            "action": None,
            "tracking": None,
            "error": None,
            "detection_age": None,
            "model_age": None,
            "last_command": None,
            "infer": _blank_infer(),
            "command_id": None,
            "session_id": None,
            "revision": 0,
            "paused": False,
        }

    def start(self, target: str, command_id=None, session_id=None,
              expected_revision=None, resume=False) -> bool:
        with self._lock:
            if command_id and command_id in self._receipts:
                receipt = self._receipts[command_id]
                if receipt["target"] != target or receipt["session_id"] != session_id:
                    raise ValueError("command_id_conflict")
                return True
            if session_id and self.state["running"]:
                raise ValueError("robot_busy")
            if expected_revision is not None and expected_revision != self.state["revision"]:
                raise ValueError("stale_command")
            if session_id and self.state["paused"] and not resume:
                raise ValueError("resume_required")
            self._epoch += 1
            epoch = self._epoch
            self._driver.stop()
            self._vision.stop()
            self._vision.start(target)
            self._heartbeat = time.monotonic()
            self._latest = None
            self._latest_at = None
            self._cycle_time = None
            self._started_at = time.monotonic()
            self._published_at = None
            self._previous_publish = None
            self._arrived_streak = 0
            self._search_streak = 0
            self._last_terminal = None
            self._tracker.reset()
            self._tracked = None
            self._tracked_at = None
            self._tracked_seed = None
            self._rotation_spent = 0.0
            self._budget_seed_at = None
            self._damper.reset()
            self._damper_stream = None
            self._slot_at = None
            self._pulse_until = None
            self._pulse_for = None
            self.state.update(
                running=True,
                target=target,
                status="acquiring",
                cycle=0,
                control_cycle=0,
                missed=0,
                arrived_streak=0,
                search_streak=0,
                out_of_order=0,
                action=None,
                tracking=None,
                error=None,
                detection_age=None,
                model_age=None,
                last_command=None,
                infer=_blank_infer(),
                command_id=command_id,
                session_id=session_id,
                revision=self.state["revision"] + 1,
                paused=False,
            )
            self._save_receipt()
        for index in range(self._concurrency):
            threading.Thread(
                target=self._perceive, args=(epoch, index), daemon=True
            ).start()
        threading.Thread(target=self._control, args=(epoch,), daemon=True).start()
        if self._track_enabled and not self._pulse_mode:
            threading.Thread(target=self._track, args=(epoch,), daemon=True).start()
        return False

    def _save_receipt(self):
        cid = self.state["command_id"]
        if cid:
            self._receipts[cid] = {key: self.state[key] for key in
                                   ("command_id", "target", "session_id", "status", "running")}
            if len(self._receipts) > 200:
                del self._receipts[next(iter(self._receipts))]

    def receipt(self, command_id):
        with self._lock:
            result = self._receipts.get(command_id)
            return dict(result) if result else None

    def stop(self, status="stopped") -> None:
        with self._lock:
            self._epoch += 1
            self.state["running"] = False
            self.state["status"] = status
            self.state["paused"] = True
            self.state["revision"] += 1
            self._driver.stop()
            self._vision.stop()
            self._save_receipt()

    def resume(self, expected_revision):
        with self._lock:
            if self.state["running"] or self.state["revision"] != expected_revision:
                raise ValueError("stale_command")
            self.state.update(paused=False, status="idle", revision=self.state["revision"] + 1)

    def heartbeat(self, session_id):
        with self._lock:
            if self.state["session_id"] == session_id:
                self._heartbeat = time.monotonic()

    def check_watchdog(self):
        with self._lock:
            if (self.state["running"] and self.state["session_id"]
                    and time.monotonic() - self._heartbeat > self._lease_seconds):
                self.stop("connection_lost")

    def snapshot(self) -> dict:
        with self._lock:
            snap = dict(self.state)
            snap["stale_after"] = self._stale_after
            snap["rotation_budget"] = self._rotation_budget
            snap["rotation_spent"] = round(self._rotation_spent, 3)
            snap["rotation_unit"] = self._rotation_unit
            snap["deg_per_turn_second"] = self._deg_per_turn_second
            snap["cycle_time"] = (
                round(self._cycle_time, 3) if self._cycle_time is not None else None
            )
            return snap

    @property
    def pulse_mode(self) -> bool:
        return self._pulse_mode

    @property
    def pulse_seconds(self) -> float:
        return self._pulse_seconds

    def budget_banner(self) -> str:
        if self._deg_per_turn_second:
            return (
                f"  rotation : budget {self._rotation_budget:g} deg per decision "
                f"({self._deg_per_turn_second:.0f} deg/s per turn-throttle)"
            )
        return (
            f"  rotation : budget {self._rotation_budget:g} throttle-seconds per "
            "decision -- UNcalibrated: run calibrate.py for a degree budget"
        )

    def _pause(self, area_fraction: Optional[float]) -> float:
        if area_fraction is not None and area_fraction >= self._short_interval_area:
            return self._short_interval
        return self._interval

    def _spacing(self, area_fraction: Optional[float]) -> float:
        """How far apart successive inference frames should be taken.

        With one worker there is nothing to spread, so this is just its pause
        and the cadence is unchanged. With several, frames should land evenly
        across the cycle rather than together, so the floor is VISION_SPACING.
        """
        pause = self._pause(area_fraction)
        if self._concurrency <= 1:
            return pause
        return max(pause / self._concurrency, self._frame_spacing)

    def _claim_slot(self, spacing: float) -> float:
        """The next capture time, from a grid shared by all the workers.

        A one-off stagger at startup decays. Each worker's cycle is its own
        model latency plus its pause, those differ by up to the second and a
        half that separates a fast reply from a slow one, so within a few
        cycles the workers drift into each other and fire together — the
        frames bunch at one instant and nothing is sampled across the rest of
        the second. Handing out capture times from one grid keeps successive
        frames `spacing` apart however the latencies wander, and whichever
        worker happens to take them.

        A worker that is already late gets `now`, so this never holds back
        throughput; it only pushes apart workers that have converged.
        """
        with self._lock:
            now = time.monotonic()
            at = now if self._slot_at is None else max(now, self._slot_at + spacing)
            self._slot_at = at
            return at

    def _owns(self, epoch: int) -> bool:
        """Whether this worker still drives the car. Caller holds the lock."""
        return self.state["running"] and self._epoch == epoch

    def _active(self, epoch: int) -> bool:
        with self._lock:
            return self._owns(epoch)

    def _finish(self, status: str, epoch: int) -> None:
        with self._lock:
            if not self._owns(epoch):
                return
            self.state["running"] = False
            self.state["status"] = status
            self.state["paused"] = status != "arrived"
            self.state["revision"] += 1
            self._driver.stop()
            self._vision.stop()
            self._save_receipt()

    def _publish(
        self,
        epoch: int,
        detection: Optional[DetectedObject],
        captured_at: float,
        frame_no: Optional[int],
        error: Optional[str],
    ) -> Optional[str]:
        """Record one result and say whether the run should end.

        Shared by every perception worker, so the streak counters stay in one
        place. Results are keyed on when the frame was *captured*, not when the
        reply arrived: with several requests in flight they finish out of order,
        and a decision about an older frame must never replace a newer one.
        """
        with self._lock:
            if not self._owns(epoch):
                return None
            if self._published_at is not None and captured_at <= self._published_at:
                self.state["out_of_order"] += 1
                return None
            self._published_at = captured_at
            if self._previous_publish is not None:
                self._cycle_time = captured_at - self._previous_publish
            self._previous_publish = captured_at

            self.state["cycle"] += 1
            self.state["error"] = error
            infer = self.state["infer"]
            infer["count"] += 1
            infer["frame_no"] = frame_no
            infer["label"] = detection.label if detection else None
            infer["box_2d"] = (
                list(detection.box_2d) if detection and detection.box_2d else None
            )
            infer["action"] = detection.action if detection else None
            infer["reason"] = detection.reason if detection else None

            if detection is None:
                self.state["missed"] += 1
                if self.state["missed"] >= self._miss_limit:
                    return "target_lost"
                return None

            self.state["missed"] = 0
            self._latest = detection
            self._latest_at = time.monotonic()
            self.state["action"] = detection.action

            searching = detection.action in ("search_left", "search_right")
            self._search_streak = self._search_streak + 1 if searching else 0
            self.state["search_streak"] = self._search_streak

            terminal = (
                "arrived" if act(detection.action, detection.box_2d).status == "arrived"
                else "halted" if detection.action == "stop"
                else None
            )
            if terminal is None:
                self._arrived_streak = 0
            elif terminal == self._last_terminal:
                self._arrived_streak += 1
            else:
                self._arrived_streak = 1
            self._last_terminal = terminal
            self.state["arrived_streak"] = self._arrived_streak

            if terminal is not None and self._arrived_streak >= self._arrive_confirm:
                return terminal
            if self._search_streak >= self._search_limit:
                return "target_lost"
            return None

    def _perceive(self, epoch: int, index: int = 0) -> None:
        """One request in flight. Several of these run when pipelining."""
        area: Optional[float] = None
        while self._active(epoch):
            # Wait for this worker's turn on the shared grid. Claiming the slot
            # before the capture is what spreads the frames: the pause used to
            # be taken after the reply, so a slow call pushed the next frame
            # late and the workers converged.
            at = self._claim_slot(self._spacing(area))
            delay = at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if not self._active(epoch):
                return
            frame_no = None
            error = None
            detection = None
            captured_at = time.monotonic()
            try:
                frame = self._camera.read()
                frame_no = self._camera.frame_no
                captured_at = time.monotonic()
                with self._lock:
                    target = self.state["target"]
                detection = self._vision.detect(target or "", frame)
            except Exception as exc:
                error = str(exc)

            terminal = self._publish(epoch, detection, captured_at, frame_no, error)
            if terminal is not None:
                self._finish(terminal, epoch)
                return

            if detection is not None:
                result = act(detection.action, detection.box_2d)
                area = result.area_fraction
            # No pause here: the wait is taken before the next capture, against
            # the grid, so it cannot accumulate into a drift.

    def _track(self, epoch: int) -> None:
        """Follow the model's box locally between replies.

        Only ever a bridge: it re-seeds on every new detection and reports its
        own failure, so the control thread can go back to the model's box
        rather than steer on a tracker that has lost the target.
        """
        period = 1.0 / max(self._track_hz, 0.1)
        while self._active(epoch):
            started = time.monotonic()
            with self._lock:
                detection = self._latest
                seeded_from = self._latest_at
                already = self._tracked_seed
            if detection is None or detection.box_2d is None:
                time.sleep(period)
                continue

            try:
                grey = cv2.cvtColor(self._camera.read(), cv2.COLOR_BGR2GRAY)
            except Exception:
                # The perception workers report camera failures; tracking just
                # stops contributing rather than ending the run twice over.
                time.sleep(period)
                continue

            if seeded_from != already:
                ok = self._tracker.seed(grey, tuple(detection.box_2d))
                with self._lock:
                    self._tracked_seed = seeded_from
                    self._tracked = detection if ok else None
                    self._tracked_at = time.monotonic() if ok else None
                    self.state["tracking"] = "seeded" if ok else "no features"
            else:
                result = self._tracker.update(grey)
                with self._lock:
                    if not self._owns(epoch):
                        return
                    if result is None:
                        self._tracked = None
                        self._tracked_at = None
                        self.state["tracking"] = "lost"
                    else:
                        box, confidence = result
                        if confidence >= self._track_min_confidence:
                            self._tracked = DetectedObject(
                                box_2d=box,
                                label=detection.label,
                                action=detection.action,
                                reason=detection.reason,
                            )
                            self._tracked_at = time.monotonic()
                            self.state["tracking"] = f"{confidence:.2f}"
                        else:
                            self._tracked = None
                            self._tracked_at = None
                            self.state["tracking"] = f"weak {confidence:.2f}"

            time.sleep(max(0.0, period - (time.monotonic() - started)))

    def _control(self, epoch: int) -> None:
        while self._active(epoch):
            now = time.monotonic()
            with self._lock:
                if not self._owns(epoch):
                    return
                overrun = (
                    self._max_run_seconds > 0
                    and now - self._started_at >= self._max_run_seconds
                )
            if overrun:
                self._finish("time_limit", epoch)
                return
            with self._lock:
                if not self._owns(epoch):
                    return
                # Prefer the tracked box: same decision, but current rather
                # than seconds old, so the turn is still the right turn.
                seed_age = (
                    None if self._tracked_seed is None else now - self._tracked_seed
                )
                trackable = (
                    not self._pulse_mode
                    and self._tracked is not None
                    and self._tracked_at is not None
                    and seed_age is not None
                    and seed_age <= self._track_max_age
                )
                if trackable:
                    detection, source_at = self._tracked, self._tracked_at
                else:
                    detection, source_at = self._latest, self._latest_at
                # Pulse mode: one short burst per model decision, then stop and
                # wait for the next. The car never moves on an extrapolation --
                # not on a tracked box, not on a decision held past its arrival
                # -- so every millimetre it travels is one the model asked for
                # while looking at a frame. Slower by construction, and the
                # overshoot that the lead term and the rotation budget exist to
                # correct simply cannot accumulate.
                if self._pulse_mode and source_at is not None and source_at != self._pulse_for:
                    self._pulse_for = source_at
                    self._pulse_until = now + self._pulse_seconds
                pulsing = (
                    not self._pulse_mode
                    or (self._pulse_until is not None and now < self._pulse_until)
                )
                # The damper differences successive boxes, so it has to know
                # when the box stops being the same running measurement:
                # falling back to the model's box, or re-seeding onto a newly
                # arrived one, is a jump rather than the target moving. Within
                # a stream the boxes are successive samples of one quantity.
                damper_stream = (
                    trackable, self._tracked_seed if trackable else None
                )
                age = None if source_at is None else now - source_at
                model_age = None if self._latest_at is None else now - self._latest_at
                # Staleness stays measured against the model: a tracker will
                # happily follow a box long after the model stopped confirming
                # that the target is really there.
                fresh = (
                    age is not None
                    and model_age is not None
                    and model_age <= self._stale_after
                )
                self.state["control_cycle"] += 1
                self.state["detection_age"] = None if age is None else round(age, 3)
                self.state["model_age"] = None if model_age is None else round(model_age, 3)

            if damper_stream != self._damper_stream:
                self._damper.reset()
                self._damper_stream = damper_stream
            turn_rate = self._damper.update(
                dx_of(detection.box_2d) if detection is not None else None, now
            )

            if detection is not None and fresh and pulsing:
                authority = turn_authority(age or 0.0)
                cmd = act(detection.action, detection.box_2d, authority, turn_rate)
                with self._lock:
                    if not self._owns(epoch):
                        return
                    # A new reply re-arms the budget: what the car may turn on
                    # this decision is spent against it, and once gone it
                    # coasts straight until the model confirms the target again.
                    if self._latest_at != self._budget_seed_at:
                        self._rotation_spent = 0.0
                        self._budget_seed_at = self._latest_at
                    remaining = self._rotation_budget - self._rotation_spent
                    if self._deg_per_turn_second:
                        cmd, spent = enforce_turn_budget_deg(
                            cmd, remaining, self._control_period,
                            self._deg_per_turn_second,
                        )
                    else:
                        cmd, spent = enforce_turn_budget(
                            cmd, remaining, self._control_period
                        )
                        # The uncalibrated helper reports turn throttle, not the
                        # throttle-seconds the budget is measured in.
                        spent *= self._control_period
                    self._rotation_spent += spent
                    self._driver.apply(cmd.left, cmd.right)
                    self.state["status"] = cmd.status
                    self.state["last_command"] = {
                        "left": round(cmd.left, 3),
                        "right": round(cmd.right, 3),
                        "status": cmd.status,
                        "note": cmd.note,
                        "label": detection.label,
                        "action": detection.action,
                        "reason": detection.reason,
                        "turn_authority": round(authority, 3),
                        "turn_rate": round(turn_rate, 3),
                        "turn_budget_used": round(self._rotation_spent, 3),
                        "box_2d": list(detection.box_2d) if detection.box_2d else None,
                        "area_fraction": round(cmd.area_fraction, 3),
                    }
            else:
                with self._lock:
                    if not self._owns(epoch):
                        return
                    self._driver.stop()
                    self.state["status"] = (
                        "acquiring" if age is None
                        else "waiting" if self._pulse_mode and fresh
                        else "stale"
                    )
                    self.state["last_command"] = None

            time.sleep(self._control_period)


def _banner(camera, vision, driver, loop) -> str:
    """Say plainly what is real and what is pretend.

    A fake driver looks identical to a working one from the web UI: the car
    simply never moves, and the logs fill with commands that went nowhere.
    """
    fake_driver = isinstance(driver, FakeDriver)
    lines = [
        "",
        "=" * 62,
        f"  vision   : {'FAKE (scripted boxes)' if is_mock() else ('LOCAL (OpenCV red blob)' if isinstance(vision, LocalVision) else 'LIVE ' + os.environ.get('HUAWEI_MODEL', 'qwen3.8-omni-flash'))}",
        f"  camera   : {type(camera).__name__}"
        + (f"  rotated {os.environ['CAMERA_ROTATION']}deg" if os.environ.get("CAMERA_ROTATION", "0") != "0" else ""),
        f"  driver   : {type(driver).__name__}"
        + ("   <-- NOTHING WILL MOVE" if fake_driver else "   <-- REAL MOTORS"),
        f"  speed    : BASE_SPEED={controller.BASE_SPEED} TURN_GAIN={controller.TURN_GAIN} "
        f"SEARCH_SPEED={controller.SEARCH_SPEED}",
        (f"  control  : PULSE every inference, {loop.pulse_seconds:g}s per decision"
         "  (tracking off, nothing moves between replies)"
         if loop.pulse_mode else
         "  control  : CONTINUOUS, driving between replies on the tracked box"),
        f"  steering : TURN_LEAD={controller.TURN_LEAD:g}s"
        + (
            "  -- no lead: a coasting chassis will overshoot"
            if not controller.TURN_LEAD
            else ""
        ),
        loop.budget_banner(),
        f"  limits   : STALE_AFTER={_env_float('STALE_AFTER', 8.0)}s "
        f"MAX_RUN_SECONDS={_env_float('MAX_RUN_SECONDS', 120.0)}s",
    ]
    if not is_mock() and hasattr(vision, "cost_cap_usd"):
        lines.append(
            f"  spend cap: ${getattr(vision, 'cost_cap_usd', 0.0):g} for this process"
        )
    if fake_driver:
        lines.append("  ** set DRIVER=tb6612 to drive the real motors **")
    lines += ["=" * 62, ""]
    return "\n".join(lines)


def _build_app(env: Optional[dict] = None):
    env = env or os.environ
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, StreamingResponse

    scene = FakeScene(boxes=list(SCENARIOS[os.getenv("FAKE_SCENARIO", "center")].boxes))
    camera = make_camera(scene)
    vision = make_vision(scene)
    driver = make_driver()
    loop = ControlLoop(camera, vision, driver)
    speech = SpeechService()
    instance_id = str(uuid.uuid4())
    print(_banner(camera, vision, driver, loop), flush=True)

    async def watchdog():
        while True:
            loop.check_watchdog()
            await asyncio.sleep(0.1)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(watchdog())
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            loop.stop()
            await speech.close()
            release = getattr(camera, "release", None)
            if release:
                release()

    app = FastAPI(title="RC Car Pilot", lifespan=lifespan)
    app.state.loop = loop
    app.state.speech = speech

    @app.get("/capabilities")
    def capabilities():
        return {**speech.capabilities(), "commandReceipts": True, "heartbeatSeconds": 1}

    @app.post("/voice/interpret")
    async def interpret(request: Request):
        request_id = request.headers.get("x-request-id", "")
        if not request_id or len(request_id) > 100:
            raise HTTPException(422, {"code": "invalid_request_id"})
        if not speech.configured:
            raise HTTPException(503, {"code": "speech_unavailable", "requestId": request_id})
        if request.headers.get("content-type", "").split(";")[0] not in ("audio/wav", "audio/x-wav"):
            raise HTTPException(415, {"code": "unsupported_audio", "requestId": request_id})
        if speech._busy.locked():
            raise HTTPException(409, {"code": "speech_busy", "requestId": request_id})
        async with speech._busy:
            data = bytearray()
            try:
                async def read_audio():
                    async for part in request.stream():
                        data.extend(part)
                        if len(data) > MAX_BYTES:
                            raise OverflowError()
                await asyncio.wait_for(read_audio(), timeout=15)
            except OverflowError:
                raise HTTPException(413, {"code": "audio_too_large", "requestId": request_id})
            except (asyncio.TimeoutError, TimeoutError):
                raise HTTPException(408, {"code": "upload_timeout", "requestId": request_id})
            try:
                validate_wav(bytes(data))
            except OverflowError:
                raise HTTPException(413, {"code": "audio_too_long", "requestId": request_id})
            except ValueError:
                raise HTTPException(422, {"code": "invalid_audio", "requestId": request_id})
            try:
                result = await asyncio.wait_for(speech.interpret(bytes(data)), timeout=30)
            except (asyncio.TimeoutError, TimeoutError):
                raise HTTPException(504, {"code": "speech_timeout", "requestId": request_id})
            except Exception as exc:
                detail = {"code": "speech_provider_error", "requestId": request_id}
                upstream_status = getattr(exc, "status_code", None)
                if isinstance(upstream_status, int) and 400 <= upstream_status <= 599:
                    detail["upstreamStatus"] = upstream_status
                upstream_code = getattr(exc, "code", None)
                if isinstance(upstream_code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", upstream_code):
                    detail["upstreamCode"] = upstream_code
                raise HTTPException(503, detail) from None
        return {"requestId": request_id, **result.model_dump(),
                "message": REJECTION if result.intent == "reject" else None,
                "simulated": speech.provider == "fake"}

    @app.post("/heartbeat")
    def heartbeat(req: SessionRequest):
        loop.heartbeat(req.sessionId)
        return {"ok": True}

    @app.get("/commands/{command_id}")
    def receipt(command_id: str):
        result = loop.receipt(command_id)
        if result is None:
            raise HTTPException(404, "unknown_command")
        return result

    def _ensure_fake() -> FakeScene:
        if not isinstance(camera, FakeCamera) or not isinstance(vision, FakeVision):
            raise HTTPException(400, "debug endpoints require fake providers")
        return scene

    @app.get("/")
    def index():
        return FileResponse(BASE_DIR / "static" / "index.html")

    @app.post("/direct")
    def direct(req: DirectRequest):
        if not req.target.strip():
            raise HTTPException(400, "target is required")
        if req.sessionId and (not req.commandId or req.expectedRevision is None):
            raise HTTPException(422, "session commands require commandId and expectedRevision")
        try:
            duplicate = loop.start(req.target.strip(), req.commandId, req.sessionId,
                                   req.expectedRevision, req.resume)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        return {"started": True, "target": req.target.strip(), "commandId": req.commandId,
                "duplicate": duplicate}

    @app.post("/stop")
    def stop():
        loop.stop()
        return {"stopped": True}

    @app.post("/resume")
    def resume(req: ResumeRequest):
        try:
            loop.resume(req.expectedRevision)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        return {"resumed": True}

    @app.get("/status")
    def status():
        snap = loop.snapshot()
        snap["driver"] = (
            getattr(driver, "last", (0.0, 0.0))
            if isinstance(driver, FakeDriver)
            else None
        )
        snap["mock"] = is_mock()
        snap["instance_id"] = instance_id
        snap["vision_mode"] = "fake" if is_mock() else ("local" if isinstance(vision, LocalVision) else "http")
        if hasattr(vision, "estimated_spend_usd"):
            snap["spend"] = {
                "estimated_usd": round(vision.estimated_spend_usd, 4),
                "cap_usd": vision.cost_cap_usd,
            }
        return snap

    @app.get("/video")
    def video():
        def gen():
            while True:
                frame = camera.read()
                frame_no = camera.frame_no
                h, w = frame.shape[:2]
                snap = loop.snapshot()
                infer = snap.get("infer") or {}
                infer_no = infer.get("frame_no")
                box = infer.get("box_2d")

                cv2.putText(
                    frame,
                    f"FRAME {frame_no}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if infer_no == frame_no:
                    label = infer.get("label") or ""
                    cv2.putText(
                        frame,
                        f"INFER RAN HERE (#{infer.get('count', 0)}) {label}",
                        (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        frame,
                        f"last infer #{infer.get('count', 0)} @ frame {infer_no}",
                        (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )
                if box is not None:
                    ymin, xmin, ymax, xmax = [int(v) for v in box]
                    x1 = int(xmin / 1000 * w)
                    y1 = int(ymin / 1000 * h)
                    x2 = int(xmax / 1000 * w)
                    y2 = int(ymax / 1000 * h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        str(infer.get("label") or "?"),
                        (x1 + 6, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ok:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )
                time.sleep(0.1)

        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/debug/scenarios")
    def list_scenarios():
        return {"scenarios": list(SCENARIOS.keys())}

    @app.post("/debug/scenario")
    def set_scenario(req: ScenarioRequest):
        _ensure_fake()
        if req.name not in SCENARIOS:
            raise HTTPException(404, "unknown scenario")
        scene.boxes = list(SCENARIOS[req.name].boxes)
        scene.reset()
        scene.clear_override()
        return {"scenario": req.name}

    @app.post("/debug/box")
    def set_box(req: BoxRequest):
        _ensure_fake()
        if req.box is None:
            scene.set_override(None)
        else:
            if not isinstance(req.box, list) or len(req.box) != 4:
                raise HTTPException(400, "box must be a list of 4 or null")
            scene.set_override(tuple(int(v) for v in req.box))
        return {"box": req.box}

    @app.post("/debug/box/clear")
    def clear_box():
        _ensure_fake()
        scene.clear_override()
        return {"cleared": True}

    return app


_app = None


def get_app():
    """The one app for this process.

    Building twice opens the camera twice, and the second rpicam-vid cannot
    acquire a sensor the first already holds — so both `python server.py` and
    `uvicorn server:app` have to land on the same instance.
    """
    global _app
    if _app is None:
        _app = _build_app()
    return _app


app = get_app()

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
