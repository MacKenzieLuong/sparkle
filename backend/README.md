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
        │  H-bridge (dual TB6612FNG) on GPIO or fake (logs)
        └─ holds last command during the pause between inference calls
```

## Layout

| File | Purpose |
| --- | --- |
| `server.py` | FastAPI app, MJPEG stream, `ControlLoop` background thread, debug endpoints |
| `vision.py` | `VisionProvider` interface, `FakeVision` (scripted), `OmniVision` (real inference via yibuapi) |
| `controller.py` | Pure steering math: bounding box → `DriveCommand(left, right, status)` |
| `camera.py` | `PiCamera` (picamera2 CSI), `FakeCamera` (synthetic frames) |
| `drive.py` | `TB6612Driver` (dual TB6612FNG, gpiozero), `FakeDriver` (logs) |
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
export HUAWEI_MODEL=qwen3.8-omni-flash          # optional, this is the default
```

Only one live vision path exists: request/response HTTP calls to
`OmniVision` on the adaptive cadence described below (`CONTROL_INTERVAL` /
`SHORT_INTERVAL`). Yibu's documented Realtime WebSocket endpoint does not
accept image events, so there is no streaming/WebSocket vision mode — an
earlier experimental adapter for it was removed. To drive the cadence faster
for a live demo, lower `CONTROL_INTERVAL`/`SHORT_INTERVAL` directly; actual
cadence is still latency-bound (model round trip + the configured pause).

## The inference model

One model does object detection: `qwen3.8-omni-flash` (a Qwen Omni
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
2. During the pause between calls the car keeps driving the **last commanded**
   direction.
3. The pause is adaptive: while the target is far, calls run every
   `CONTROL_INTERVAL` (default 5s). Once the target's bounding box covers ≥
   `SHORT_INTERVAL_AREA` (default 0.15) of the frame, the cadence tightens to
   `SHORT_INTERVAL` (default 1s) for precise final alignment — same per-call
   cost, ~5× more steering updates near the target.
4. Exit conditions (checked on detection cycles):
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
| GET | `/video` | MJPEG camera stream (free preview, no inference) — overlays the live frame number, and highlights `INFER RAN HERE` on the exact frame the control loop used, plus the last inferred box |
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
| `HUAWEI_MODEL` | `qwen3.8-omni-flash` | Model used for detection |
| `CAMERA` | `fake` | `fake`, `picamera2`, or `rpicam` (one `rpicam-vid` MJPEG process) |
| `CAMERA_FRAMERATE` | `30` | Capture rate when `CAMERA=rpicam` |
| `DRIVER` | `fake` | `fake` or `tb6612` (`l298n` accepted as an alias) |
| `VISION_MAX_TOKENS` | `128` | Small response limit for bounding-box JSON |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `640` / `480` | Frame resolution |
| `CONTROL_INTERVAL` | `5` | Seconds between inference calls while the target is far |
| `SHORT_INTERVAL` | `1` | Seconds between inference calls once the target is near (≥ `SHORT_INTERVAL_AREA`) |
| `SHORT_INTERVAL_AREA` | `0.15` | Box area fraction (of the 1000×1000 frame) that triggers the fast cadence |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Uvicorn bind address |
| `TB6612_AIN1`…`TB6612_STBY` | see `.env.example` | GPIO pins for the dual TB6612FNG (SparkFun) |
| `TB6612_STBY` | `21` | Driver standby pin (held high to drive, low when stopped) |

## Tests

```sh
.venv/bin/python -m pytest
```

Covers the steering math (`test_controller.py`), the lenient JSON box
parsing used in real mode (`test_vision.py`), the adaptive pause and
inference-overlay state (`test_server.py`), and per-frame camera counters
(`test_camera.py`). No network, no API usage.

## Running on the Pi

```sh
export MOCK=false
export HUAWEI_API_KEY=<key>
export CAMERA=picamera2
export DRIVER=tb6612
export TB6612_AIN1=17 TB6612_AIN2=27 TB6612_PWMA=13
export TB6612_BIN1=16 TB6612_BIN2=20 TB6612_PWMB=12
export TB6612_STBY=21
.venv/bin/python server.py
```

`picamera2`, `gpiozero`, and `openai` require `pip install -r requirements.txt`
(picamera2 is arm/Linux-only). The mapping follows the SparkFun handoff:
**Motor A = left wheel** (AIN1=17, AIN2=27, PWMA=13; forward=A IN2/GPIO 27),
**Motor B = right wheel** (BIN1=16, BIN2=20, PWMB=12; forward=B IN1/GPIO 16),
and **STBY=21**. The car drives slowly (the loop is paced by API
latency + `CONTROL_INTERVAL`), so allow plenty of room. Point the web UI at
the Pi's LAN address and the video preview shows the car's view.

## Benchmarks

Measured on macOS (2026 M-series) with the mock stack and `CONTROL_INTERVAL=5`;
real hardware keeps the same structure, only the absolute per-cell ms values
move.

**Per-stage local cost** (mean / 99th percentile):

| stage | mean | 99th |
| --- | --- | --- |
| `camera.read()` fake 640×480 render | 0.038 ms | 0.048 ms |
| JPEG encode q85 | 0.566 ms | 0.662 ms |
| base64 of that JPEG | 0.581 ms | 0.659 ms |
| `controller.command()` (center/right/None) | ≤ 0.002 ms | — |
| `driver.apply()` (fake) | ~0.000 ms | — |
| full local detect→compute→apply cycle | 0.003 ms | 0.008 ms |

Local cost is a rounding error: the control loop's latency is entirely
`api_latency + pause`. First command lands `api_latency + ~25 ms` after `go`
(the ~25 ms is read+encode+parse+compute+apply on the Pi).

**Cadence with and without near-target optimization** (fixed mimic API latencies):

| api mimic | first cmd latency | far cadence `5s` | near cadence `1s` | cycles/min near |
| --- | --- | --- | --- | --- |
| 0.0 (mock) | ~25 ms | 5.000 s | 1.000 s | 60 |
| 0.5 s | ~528 ms | 5.500 s | 1.500 s | 40 |
| 1.0 s | ~1030 ms | 6.000 s | 2.000 s | 30 |
| 2.0 s | ~2029 ms | 7.000 s | 3.000 s | 20 |

Steady-state cadence is exactly `pause + api` (no jitter), so with a real
~1–2 s API round trip the car gets 20–40 steering updates per minute during
final approach instead of ~10.
