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
from vision import FakeVision, VisionProvider, is_mock, make_vision

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


class ControlLoop:
    def __init__(self, camera: CameraProvider, vision: VisionProvider, driver: Driver):
        self._camera = camera
        self._vision = vision
        self._driver = driver
        self._interval = _env_float("CONTROL_INTERVAL", 5.0)
        self._short_interval = _env_float("SHORT_INTERVAL", 1.0)
        self._short_interval_area = _env_float("SHORT_INTERVAL_AREA", 0.15)
        self._miss_limit = 3
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.state = {
            "running": False,
            "target": None,
            "status": "idle",
            "cycle": 0,
            "missed": 0,
            "last_command": None,
            "infer": {"count": 0, "frame_no": None, "label": None, "box_2d": None},
        }

    def start(self, target: str) -> None:
        with self._lock:
            self.state["target"] = target
            self.state["running"] = True
            self.state["status"] = "starting"
            self.state["cycle"] = 0
            self.state["missed"] = 0
            self.state["last_command"] = None
            self.state["infer"] = {
                "count": 0,
                "frame_no": None,
                "label": None,
                "box_2d": None,
            }
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self.state["running"] = False
            self.state["status"] = "stopped"
        self._driver.stop()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state)

    def _pause(self, area_fraction: Optional[float]) -> float:
        if area_fraction is not None and area_fraction >= self._short_interval_area:
            return self._short_interval
        return self._interval

    def _run(self) -> None:
        last_area: Optional[float] = None
        while self._is_running():
            frame = self._camera.read()
            frame_no = self._camera.frame_no
            with self._lock:
                target = self.state["target"]

            try:
                detection = self._vision.detect(target or "", frame)
            except Exception as exc:
                detection = None
                with self._lock:
                    self.state["status"] = f"error: {exc}"

            with self._lock:
                self.state["cycle"] += 1
                infer = self.state["infer"]
                infer["count"] += 1
                infer["frame_no"] = frame_no
                infer["label"] = detection.label if detection else None
                infer["box_2d"] = (
                    list(detection.box_2d) if detection else None
                )

            if detection is None:
                with self._lock:
                    self.state["missed"] += 1
                    if self.state["missed"] >= self._miss_limit:
                        self.state["running"] = False
                        self.state["status"] = "target_lost"
                if not self._is_running():
                    self._driver.stop()
                    break
            else:
                with self._lock:
                    self.state["missed"] = 0
                cmd = command(detection.box_2d)
                last_area = cmd.area_fraction
                self._driver.apply(cmd.left, cmd.right)
                with self._lock:
                    self.state["last_command"] = {
                        "left": round(cmd.left, 3),
                        "right": round(cmd.right, 3),
                        "status": cmd.status,
                        "note": cmd.note,
                        "label": detection.label,
                        "box_2d": list(detection.box_2d),
                        "area_fraction": round(cmd.area_fraction, 3),
                    }
                    self.state["status"] = cmd.status
                    if cmd.status == "arrived":
                        self.state["running"] = False
                if cmd.status == "arrived":
                    self._driver.stop()
                    break

            time.sleep(self._pause(last_area))

    def _is_running(self) -> bool:
        with self._lock:
            return self.state["running"]


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
