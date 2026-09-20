from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from pydantic import BaseModel

import controller
from camera import CameraProvider, FakeCamera, make_camera
from controller import act, command, turn_authority
from drive import Driver, FakeDriver, make_driver
from scenarios import SCENARIOS, FakeScene
from vision import (
    DetectedObject,
    FakeVision,
    OmniVision,
    VisionProvider,
    is_mock,
    make_vision,
)

BASE_DIR = Path(__file__).resolve().parent


class DirectRequest(BaseModel):
    target: str


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
        # Last resort: if the network drops, /stop is unreachable and nothing
        # else bounds a drive that never arrives.
        self._max_run_seconds = _env_float("MAX_RUN_SECONDS", 120.0)
        self._lock = threading.Lock()
        self._epoch = 0
        self._latest: Optional[DetectedObject] = None
        self._latest_at: Optional[float] = None
        self._cycle_time: Optional[float] = None
        self._started_at = 0.0
        self._published_at: Optional[float] = None
        self._previous_publish: Optional[float] = None
        self._arrived_streak = 0
        self._search_streak = 0
        self._last_terminal: Optional[str] = None
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
            "error": None,
            "detection_age": None,
            "last_command": None,
            "infer": _blank_infer(),
        }

    def start(self, target: str) -> None:
        self._driver.stop()
        self._vision.start(target)
        with self._lock:
            self._epoch += 1
            epoch = self._epoch
            self._latest = None
            self._latest_at = None
            self._cycle_time = None
            self._started_at = time.monotonic()
            self._published_at = None
            self._previous_publish = None
            self._arrived_streak = 0
            self._search_streak = 0
            self._last_terminal = None
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
                error=None,
                detection_age=None,
                last_command=None,
                infer=_blank_infer(),
            )
        for index in range(self._concurrency):
            threading.Thread(
                target=self._perceive, args=(epoch, index), daemon=True
            ).start()
        threading.Thread(target=self._control, args=(epoch,), daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            # Orphans the current workers so a restart never races them.
            self._epoch += 1
            self.state["running"] = False
            self.state["status"] = "stopped"
        self._driver.stop()
        self._vision.stop()

    def snapshot(self) -> dict:
        with self._lock:
            snap = dict(self.state)
            snap["stale_after"] = self._stale_after
            snap["cycle_time"] = (
                round(self._cycle_time, 3) if self._cycle_time is not None else None
            )
            return snap

    def _pause(self, area_fraction: Optional[float]) -> float:
        if area_fraction is not None and area_fraction >= self._short_interval_area:
            return self._short_interval
        return self._interval

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
        self._driver.stop()
        self._vision.stop()

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
        if index:
            # Spread the workers out, or they bunch up and the gap between
            # decisions is no better than with one.
            time.sleep(self._stagger * index)
        area: Optional[float] = None
        while self._active(epoch):
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
            time.sleep(self._pause(area))

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
                detection = self._latest
                age = None if self._latest_at is None else now - self._latest_at
                fresh = age is not None and age <= self._stale_after
                self.state["control_cycle"] += 1
                self.state["detection_age"] = None if age is None else round(age, 3)

            if detection is not None and fresh:
                authority = turn_authority(age or 0.0)
                cmd = act(detection.action, detection.box_2d, authority)
                self._driver.apply(cmd.left, cmd.right)
                with self._lock:
                    if not self._owns(epoch):
                        return
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
                        "box_2d": list(detection.box_2d) if detection.box_2d else None,
                        "area_fraction": round(cmd.area_fraction, 3),
                    }
            else:
                self._driver.stop()
                with self._lock:
                    if not self._owns(epoch):
                        return
                    self.state["status"] = "acquiring" if age is None else "stale"
                    self.state["last_command"] = None

            time.sleep(self._control_period)


def _banner(camera, vision, driver) -> str:
    """Say plainly what is real and what is pretend.

    A fake driver looks identical to a working one from the web UI: the car
    simply never moves, and the logs fill with commands that went nowhere.
    """
    fake_driver = isinstance(driver, FakeDriver)
    lines = [
        "",
        "=" * 62,
        f"  vision   : {'FAKE (scripted boxes)' if is_mock() else 'LIVE ' + os.environ.get('HUAWEI_MODEL', 'qwen3.8-omni-flash')}",
        f"  camera   : {type(camera).__name__}"
        + (f"  rotated {os.environ['CAMERA_ROTATION']}deg" if os.environ.get("CAMERA_ROTATION", "0") != "0" else ""),
        f"  driver   : {type(driver).__name__}"
        + ("   <-- NOTHING WILL MOVE" if fake_driver else "   <-- REAL MOTORS"),
        f"  speed    : BASE_SPEED={controller.BASE_SPEED} TURN_GAIN={controller.TURN_GAIN} "
        f"SEARCH_SPEED={controller.SEARCH_SPEED}",
        f"  limits   : STALE_AFTER={_env_float('STALE_AFTER', 8.0)}s "
        f"MAX_RUN_SECONDS={_env_float('MAX_RUN_SECONDS', 120.0)}s",
    ]
    if not is_mock():
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

    scene = SCENARIOS["center"]
    camera = make_camera(scene)
    vision = make_vision(scene)
    driver = make_driver()
    loop = ControlLoop(camera, vision, driver)
    print(_banner(camera, vision, driver), flush=True)

    app = FastAPI(title="RC Car Pilot")

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
        loop.start(req.target.strip())
        return {"started": True, "target": req.target.strip()}

    @app.on_event("shutdown")
    def shutdown() -> None:
        loop.stop()
        release = getattr(camera, "release", None)
        if release:
            release()

    @app.post("/stop")
    def stop():
        loop.stop()
        return {"stopped": True}

    @app.get("/status")
    def status():
        snap = loop.snapshot()
        snap["driver"] = (
            getattr(driver, "last", (0.0, 0.0))
            if isinstance(driver, FakeDriver)
            else None
        )
        snap["mock"] = is_mock()
        snap["vision_mode"] = "fake" if is_mock() else "http"
        if isinstance(vision, OmniVision):
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
