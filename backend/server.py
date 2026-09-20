from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from pydantic import BaseModel

from camera import CameraProvider, FakeCamera, make_camera
from controller import command
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
    return {"count": 0, "frame_no": None, "label": None, "box_2d": None}


class ControlLoop:
    """Perception and steering run on separate threads.

    A model call takes seconds; steering must not. The perception thread
    publishes the freshest box it can get, and the control thread steers off
    that box at a fixed rate, cutting the motors whenever the box is older
    than the observed perception cycle can account for.
    """

    def __init__(self, camera: CameraProvider, vision: VisionProvider, driver: Driver):
        self._camera = camera
        self._vision = vision
        self._driver = driver
        self._interval = _env_float("CONTROL_INTERVAL", 5.0)
        self._short_interval = _env_float("SHORT_INTERVAL", 1.0)
        self._short_interval_area = _env_float("SHORT_INTERVAL_AREA", 0.15)
        self._control_period = 1.0 / max(_env_float("CONTROL_HZ", 10.0), 0.1)
        self._stale_factor = _env_float("STALE_FACTOR", 1.5)
        self._stale_min = _env_float("STALE_MIN", 1.0)
        self._miss_limit = 3
        self._lock = threading.Lock()
        self._epoch = 0
        self._latest: Optional[DetectedObject] = None
        self._latest_at: Optional[float] = None
        self._cycle_time: Optional[float] = None
        self.state = {
            "running": False,
            "target": None,
            "status": "idle",
            "cycle": 0,
            "control_cycle": 0,
            "missed": 0,
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
            self.state.update(
                running=True,
                target=target,
                status="acquiring",
                cycle=0,
                control_cycle=0,
                missed=0,
                error=None,
                detection_age=None,
                last_command=None,
                infer=_blank_infer(),
            )
        for worker in (self._perceive, self._control):
            threading.Thread(target=worker, args=(epoch,), daemon=True).start()

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
            snap["stale_after"] = round(self._stale_after(), 3)
            return snap

    def _pause(self, area_fraction: Optional[float]) -> float:
        if area_fraction is not None and area_fraction >= self._short_interval_area:
            return self._short_interval
        return self._interval

    def _stale_after(self) -> float:
        """Seconds a box stays usable. Caller holds the lock.

        Scaled off the measured perception cycle so the deadman catches a hung
        model call without firing on a cadence the operator chose deliberately.
        """
        cycle = self._cycle_time if self._cycle_time is not None else self._interval
        return max(self._stale_min, self._stale_factor * cycle)

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

    def _perceive(self, epoch: int) -> None:
        area: Optional[float] = None
        previous_finish: Optional[float] = None
        while self._active(epoch):
            frame_no = None
            error = None
            detection = None
            try:
                frame = self._camera.read()
                frame_no = self._camera.frame_no
                with self._lock:
                    target = self.state["target"]
                detection = self._vision.detect(target or "", frame)
            except Exception as exc:
                error = str(exc)

            now = time.monotonic()
            with self._lock:
                if not self._owns(epoch):
                    return
                if previous_finish is not None:
                    self._cycle_time = now - previous_finish
                self.state["cycle"] += 1
                self.state["error"] = error
                infer = self.state["infer"]
                infer["count"] += 1
                infer["frame_no"] = frame_no
                infer["label"] = detection.label if detection else None
                infer["box_2d"] = list(detection.box_2d) if detection else None
                if detection is None:
                    self.state["missed"] += 1
                else:
                    self.state["missed"] = 0
                    self._latest = detection
                    self._latest_at = now
                lost = self.state["missed"] >= self._miss_limit
            previous_finish = now

            if lost:
                self._finish("target_lost", epoch)
                return
            if detection is not None:
                area = command(detection.box_2d).area_fraction
            time.sleep(self._pause(area))

    def _control(self, epoch: int) -> None:
        while self._active(epoch):
            now = time.monotonic()
            with self._lock:
                if not self._owns(epoch):
                    return
                detection = self._latest
                age = None if self._latest_at is None else now - self._latest_at
                fresh = age is not None and age <= self._stale_after()
                self.state["control_cycle"] += 1
                self.state["detection_age"] = None if age is None else round(age, 3)

            if detection is not None and fresh:
                cmd = command(detection.box_2d)
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
                        "box_2d": list(detection.box_2d),
                        "area_fraction": round(cmd.area_fraction, 3),
                    }
                if cmd.status == "arrived":
                    self._finish("arrived", epoch)
                    return
            else:
                self._driver.stop()
                with self._lock:
                    if not self._owns(epoch):
                        return
                    self.state["status"] = "acquiring" if age is None else "stale"
                    self.state["last_command"] = None

            time.sleep(self._control_period)


def _build_app(env: Optional[dict] = None):
    env = env or os.environ
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, StreamingResponse

    scene = SCENARIOS["center"]
    camera = make_camera(scene)
    vision = make_vision(scene)
    driver = make_driver()
    loop = ControlLoop(camera, vision, driver)

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


def get_app():
    return _build_app()


app = get_app()

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(get_app(), host=host, port=port)
