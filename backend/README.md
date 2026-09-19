# Sparkle Backend — RC Car Autopilot

Real-time "drive to the thing I named" API for a Raspberry Pi RC car. A web
page takes a natural-language target (e.g. "go to the red ball"), a vision
model finds the object in the camera frame, steering math turns that into left/
right motor throttles, and the car drives until it arrives or loses the target.

## Architecture

```
User types "go to the red ball" on web page
        │  POST /direct {"target": "..."}
        ▼
   FastAPI server (on the Pi)
        │  starts ControlLoop (background thread)
        │  every CONTROL_INTERVAL (default 5s), immediately on Go:
        │    grab frame from camera ───────────► Pi Camera (picamera2) or fake
        │    vision.detect(target, frame) ─────► OmniVision (yibuapi/OMNI) or fake
        │    JSON: [{"box_2d":[ymin,xmin,ymax,xmax],"label":...}] or not found
        ▼
   Steering math (controller.command)
        │  box center vs frame center → turn left/right
        │  box size → approach speed; box ~50%+ of frame → arrived
        ▼
   Motor driver ──► left/right throttle in [-1, 1]
        │  H-bridge (L298N) on GPIO or fake (logs)
        └─ holds last command during the pause between inference calls
```

## Layout

| File | Purpose |
| --- | --- |
| `server.py` | FastAPI app, MJPEG stream, `ControlLoop` background thread, debug endpoints |
| `vision.py` | `VisionProvider` interface, `FakeVision` (scripted), `OmniVision` (real inference via yibuapi) |
| `controller.py` | Pure steering math: bounding box → `DriveCommand(left, right, status)` |
| `camera.py` | `PiCamera` (picamera2 CSI), `FakeCamera` (synthetic frames) |
| `drive.py` | `L298NDriver` (gpiozero), `FakeDriver` (logs) |
| `scenarios.py` | Scripted bounding-box scenarios shared by `FakeVision` and `FakeCamera` |
| `static/index.html` | Web UI: camera preview, target input, scenario/box debug controls |
| `test_controller.py`, `test_vision.py` | pytest suites |

Every hardware piece has a fake counterpart, so the whole server runs on a
laptop with `MOCK=true` (the default) — no Pi, no camera, no motors, no API
calls.

## Running

Python 3.9+. Create a venv and install:

```sh
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Start the server (mock mode, no config needed):

```sh
.venv/bin/python server.py
# -> http://0.0.0.0:8000  (open the web UI at http://<host>:8000)
```

Environment variables are read from the shell only. The file `.env` is **not**
read by the backend.

## Mock mode vs real inference

All inference is gated behind `MOCK`:

| `MOCK` | What happens |
| --- | --- |
| `true` (default) | `FakeVision` replays scripted boxes. No key, no network, no credits spent. |
| `false` | Calls the real inference endpoint via `OmniVision`. Requires `HUAWEI_API_KEY` or startup fails fast with a clear error. |

To go live, export the sponsor-key env vars and restart:

```sh
export MOCK=false
export HUAWEI_API_KEY=<key from yibuapi>
export HUAWEI_BASE_URL=https://yibuapi.com/v1   # optional, this is the default
export HUAWEI_MODEL=qwen3.5-omni-flash          # optional, this is the default
```

## The inference model

One model does object detection: `qwen3.5-omni-flash` (a Qwen3.5-Omni
multimodal model) reached through the sponsor's OpenAI-compatible gateway
(yibuapi). It is a general multimodal LLM, not a purpose-trained detector like
YOLO — it is prompted to return bounding-box JSON and boxes are parsed
leniently (`vision._parse_boxes`), so approximate output degrades to "not
found" rather than crashing the loop. To keep cost/latency down, only one
downscaled JPEG frame is sent per cycle and calls run at the 5s cadence.

Detection contract (same shape the steering math consumes):

```json
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}]
```

Coordinates are normalized to `0–1000` for a 1000×1000 reference frame
(`box_2d` order is `ymin, xmin, ymax, xmax`). An empty array means the target
is not in frame.

## Control loop behavior

On `POST /direct`:

1. Detection runs **immediately** (starts on "go", no initial pause).
2. During the pause between calls (`CONTROL_INTERVAL`, default 5s) the car
   keeps driving the **last commanded** direction.
3. Exit conditions (checked on detection cycles):
   - **arrived** — bounding box covers ≥ 50% of the frame → stop.
   - **target_lost** — no detection for 3 consecutive cycles → stop.
   - **manual stop** — `POST /stop`.

## Steering math

`controller.command(box_2d)` returns `DriveCommand(left, right, status, note)`:

- `left` / `right` are motor throttles in `[-1, 1]`.
- `dx = (box_center_x − 500) / 500`; a dead zone (`|dx| < 0.08`) drives
  straight, otherwise the car turns proportionally (target right → left wheel
  faster).
- Approach speed scales down as the box grows; `area_fraction ≥ 0.5` →
  `status="arrived"` with both throttles `0`.
- `box_2d=None` → `status="target_lost"`, both throttles `0`.

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Web UI (MJPEG preview + controls) |
| POST | `/direct` | Body `{"target": "the red ball"}` — start the control loop |
| POST | `/stop` | Stop the car and the loop |
| GET | `/status` | Loop state: running/target/status/cycle/last command/mock |
| GET | `/video` | MJPEG camera stream (free preview, no inference) |
| GET | `/debug/scenarios` | List scripted fake scenarios |
| POST | `/debug/scenario` | Body `{"name": "approach"}` — load a scripted scenario (mock only) |
| POST | `/debug/box` | Body `{"box": [ymin,xmin,ymax,xmax]}` or `{"box": null}` — override the fake box (mock only) |
| POST | `/debug/box/clear` | Clear the override (mock only) |

Debug endpoints return `400` when the providers are not fake.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `MOCK` | `true` | `true` = scripted fake inference; `false` = real API calls |
| `HUAWEI_API_KEY` | — | Sponsor key; required when `MOCK=false` |
| `HUAWEI_BASE_URL` | `https://yibuapi.com/v1` | OpenAI-compatible gateway base URL |
| `HUAWEI_MODEL` | `qwen3.5-omni-flash` | Model used for detection |
| `CAMERA` | `fake` | `fake` or `picamera2` |
| `DRIVER` | `fake` | `fake` or `l298n` |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `640` / `480` | Frame resolution |
| `CONTROL_INTERVAL` | `5` | Seconds between inference calls (the pause) |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Uvicorn bind address |
| `L298N_LEFT_FWD`…`L298N_STBY` | see `.env.example` | GPIO pins for the motor driver (SparkFun) |
| `L298N_STBY` | `21` | Driver standby pin (held high to drive, low when stopped) |

## Tests

```sh
.venv/bin/python -m pytest
```

Covers the steering math (`test_controller.py`) and the lenient JSON box
parsing used in real mode (`test_vision.py`). No network, no API usage.

## Running on the Pi

```sh
export MOCK=false
export HUAWEI_API_KEY=<key>
export CAMERA=picamera2
export DRIVER=l298n
export L298N_LEFT_FWD=17 L298N_LEFT_REV=27 L298N_LEFT_EN=13
export L298N_RIGHT_FWD=16 L298N_RIGHT_REV=20 L298N_RIGHT_EN=12
export L298N_STBY=21
.venv/bin/python server.py
```

`picamera2`, `gpiozero`, and `openai` require `pip install -r requirements.txt`
(picamera2 is arm/Linux-only). The mapping follows the SparkFun handoff:
**Motor A = left wheel** (AI1=17, AI2=27, PWMA=13), **Motor B = right wheel**
(BI1=16, BI2=20, PWMB=12), **STBY=21**. The code never swaps A/B for
forward/reverse/turn behavior; if a wheel spins backwards, swap that motor's
two output wires physically. The car drives slowly (the loop is paced by API
latency + `CONTROL_INTERVAL`), so allow plenty of room. Point the web UI at
the Pi's LAN address and the video preview shows the car's view.