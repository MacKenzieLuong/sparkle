# Sparkle Backend — RC Car Autopilot

Real-time "drive to the thing I named" API for a Raspberry Pi RC car. A web
page takes a natural-language target (e.g. "go to the red ball"), a vision
model finds the object in the camera frame, steering math turns that into left/
right motor throttles, and the car drives until it arrives or loses the target.

## Architecture

A model call takes seconds; steering must not. So `POST /direct` starts two
threads that share one "freshest box" record:

```
User types "go to the red ball" on web page
        │  POST /direct {"target": "..."}
        ▼
   FastAPI server (on the Pi)
        │
        ├── perception thread ── every CONTROL_INTERVAL, immediately on Go:
        │     grab frame from camera ──────────► Pi Camera / rpicam-vid / fake
        │     vision.detect(target, frame) ────► OmniVision (yibuapi) or fake
        │     JSON: [{"box_2d":[ymin,xmin,ymax,xmax],"label":...}] or not found
        │            │
        │            ▼
        │     ┌──────────────────────┐
        │     │ latest box + arrival │  ◄── shared, lock-guarded
        │     │ timestamp            │
        │     └──────────────────────┘
        │            │
        └── control thread ── every 1/CONTROL_HZ (default 10 Hz):
              read freshest box; if older than the deadman window → motors off
              else controller.command(box) ────► left/right throttle in [-1, 1]
                     box center vs frame center → turn
                     box size → approach speed; ≥50% of frame → arrived
                     │
                     ▼
              Motor driver: H-bridge (dual TB6612FNG) on GPIO or fake (logs)
```

The car therefore keeps steering at a steady 10 Hz off the last box it saw,
instead of freezing for the whole duration of each model round trip. If
perception stalls — hung request, dead camera, dropped network — the box goes
stale and the control thread cuts the motors without waiting for the request
to time out.

## Layout

| File | Purpose |
| --- | --- |
| `server.py` | FastAPI app, MJPEG stream, `ControlLoop` (perception + control threads), debug endpoints |
| `cli.py` | One-shot detection from the terminal — model check without the car |
| `vision.py` | `VisionProvider` interface, `FakeVision` (scripted), `OmniVision` (real inference via yibuapi) |
| `controller.py` | Pure steering math: bounding box → `DriveCommand(left, right, status)` |
| `camera.py` | `PiCamera` (picamera2 CSI), `RpiCamCamera` (rpicam-vid MJPEG), `WebcamCamera`, `FakeCamera` (synthetic frames) |
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

### Spend guard

Live mode estimates the cost of every call and refuses to make one that would
push the total past `MAX_COST_USD` (default **$1.00**). The count is **per
process**, not per navigation, so a restart is what resets it — a cap that
reset on every press of Go would not cap a test session. `/status` reports the
running total:

```json
"spend": {"estimated_usd": 0.0134, "cap_usd": 1.0}
```

The estimate bills one input token per 32×32 pixel block plus the full output
token cap, so it runs slightly ahead of reality rather than behind it. It is a
local guard, not an accounting record — **set a hard spending limit at the
provider too.** Hitting the cap raises on each attempt, which counts as a miss,
so navigation stops within `miss_limit` cycles with the reason in
`status.error`. No call is sent once the cap is reached.

Requests use a `VISION_TIMEOUT` (default 10s) with **retries disabled**: a
retry would re-send a frame describing where the car used to be, so failing
fast and sending a fresh frame next cycle is both cheaper and more correct.

## Talking to the model from the terminal

`cli.py` runs one detection and prints what came back — no server, no motors.
It reads the same env vars as the server, so it is the cheapest way to confirm
a key, a model, or a camera actually works before the car moves.

```sh
.venv/bin/python cli.py "the red ball"                 # one look through the camera
.venv/bin/python cli.py "the red ball" --image shot.jpg # ...or at a saved frame
.venv/bin/python cli.py "the red ball" -n 5 --interval 1  # sample latency
.venv/bin/python cli.py "the red ball" --raw --save look   # raw reply + overlays
```

Each look prints the round trip, the parsed box, and **what the steering math
would have done with it**:

```
target 'the red ball' via camera (RpiCamCamera), mock=False
     1430 ms  'red ball' box=[350, 400, 650, 600] area=0.060 -> left=+0.44 right=+0.44 (moving)
```

So a bad run can be pinned on the model, the parse, or the controller without
moving a wheel. `--raw` prints the model's reply verbatim, which is what you
want the first time a new model returns something `_parse_boxes` rejects.
`--save NAME` writes `NAME-1.jpg` with the box drawn on the frame, so you can
check the box is actually *on* the object. Exit status is `0` when something
was found, `1` when nothing was, `2` on error — usable in a shell loop.

Latency from `-n` is what you tune `CONTROL_INTERVAL` against, and the run ends
by reporting estimated spend against the cap.

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
2. The **perception thread** paces model calls. Its pause is adaptive: while
   the target is far, calls run every `CONTROL_INTERVAL` (default 5s). Once the
   box covers ≥ `SHORT_INTERVAL_AREA` (default 0.15) of the frame, the cadence
   tightens to `SHORT_INTERVAL` (default 1s) for final alignment. This pause is
   purely a **cost governor** — it no longer limits how often the car steers.
3. The **control thread** re-steers every `1/CONTROL_HZ` (default 10 Hz) from
   the freshest box, regardless of what perception is doing.
4. **Staleness deadman.** If the freshest box is older than
   `max(STALE_MIN, STALE_FACTOR × measured perception cycle)`, the control
   thread cuts the motors and reports `status="stale"`, leaving navigation
   running so it resumes the moment a box arrives. The window is scaled off the
   *measured* cycle so a deliberately slow cadence never trips it, while a hung
   request trips it within a couple of cycles. `/status` reports the live value
   as `stale_after`.
5. Exit conditions:
   - **arrived** — bounding box covers ≥ 50% of the frame → stop.
   - **target_lost** — no detection for 3 consecutive perception cycles → stop.
     Camera and API errors count as misses, so a persistent failure ends the
     run rather than looping forever; the message lands in `status.error`.
   - **manual stop** — `POST /stop`.

Restarting with a new target bumps an epoch counter that orphans the previous
pair of threads, so a slow in-flight call from the old run can never write a
throttle for the new one.

For live driving you likely want a much tighter perception cadence than the
cost-conservative default — set `CONTROL_INTERVAL=0` to poll as fast as model
latency allows (this is what the old `http-poll` mode did), and budget
accordingly.

## Steering math

`controller.command(box_2d)` returns `DriveCommand(left, right, status, note)`:

- `left` / `right` are motor throttles in `[-1, 1]`.
- `dx = (box_center_x − 500) / 500`; a dead zone (`DEAD_ZONE`, default 0.08)
  drives straight, otherwise the car turns proportionally by `TURN_GAIN`
  (target right → left wheel faster).
- Approach speed starts at `BASE_SPEED` and scales down as the box grows;
  `area_fraction ≥ ARRIVED_AREA_FRACTION` → `status="arrived"`, throttles `0`.
- `box_2d=None` → `status="target_lost"`, both throttles `0`.

### Picking a speed

**Speed has to be chosen against model latency, not by feel.** The car drives
on the last box for the whole round trip, so at a measured ~3s latency it
covers three seconds of ground blind between corrections — and the staleness
deadman cannot react faster than one perception cycle either. Measure first:

```sh
.venv/bin/python cli.py "<something in frame>" -n 5 --interval 1
```

Then set `BASE_SPEED` so that *latency × speed* is a distance you are willing
to let the car travel uncorrected. Start around `0.2` on a first floor test and
raise it only once you have watched it steer. Lower `TURN_GAIN` with it —
turning is not scaled by `BASE_SPEED`, so a slow car with the default gain
spins in place instead of driving toward the target.

These are read at import, so set them before launching the server.

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Web UI (MJPEG preview + controls) |
| POST | `/direct` | Body `{"target": "the red ball"}` — start the control loop |
| POST | `/stop` | Stop the car and the loop |
| GET | `/status` | Loop state: running/target/status/error, `cycle` (perception) vs `control_cycle` (steering), `detection_age`, `stale_after`, last command, mock |
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
| `CAMERA` | `fake` | `fake`, `picamera2`, `webcam`, or `rpicam` (one `rpicam-vid` MJPEG process, drained by a reader thread that keeps only the newest frame) |
| `CAMERA_FRAMERATE` | `30` | Capture rate when `CAMERA=rpicam` |
| `DRIVER` | `fake` | `fake` or `tb6612` (`l298n` accepted as an alias) |
| `VISION_MAX_TOKENS` | `128` | Small response limit for bounding-box JSON |
| `VISION_TIMEOUT` | `10` | Per-request timeout in seconds; retries are disabled |
| `MAX_COST_USD` | `1.00` | Estimated spend cap for the process; `0` disables it |
| `VISION_INPUT_USD_PER_MILLION` | `0.55` | Image-token input rate used by the estimate |
| `VISION_OUTPUT_USD_PER_MILLION` | `2.20` | Output-token rate used by the estimate |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `640` / `480` | Frame resolution |
| `CONTROL_INTERVAL` | `5` | Seconds between inference calls while the target is far; `0` polls as fast as model latency allows |
| `SHORT_INTERVAL` | `1` | Seconds between inference calls once the target is near (≥ `SHORT_INTERVAL_AREA`) |
| `SHORT_INTERVAL_AREA` | `0.15` | Box area fraction (of the 1000×1000 frame) that triggers the fast cadence |
| `CONTROL_HZ` | `10` | Steering updates per second, independent of inference cadence |
| `STALE_FACTOR` | `1.5` | Motors cut once the freshest box is this many measured perception cycles old |
| `STALE_MIN` | `1.0` | Floor for the staleness window, in seconds |
| `BASE_SPEED` | `0.5` | Forward throttle before area scaling — **lower this for a slow first test** |
| `TURN_GAIN` | `0.8` | Turn aggressiveness; lower it alongside `BASE_SPEED` |
| `DEAD_ZONE` | `0.08` | Horizontal offset below which the car drives straight |
| `ARRIVED_AREA_FRACTION` | `0.5` | Box area fraction that counts as arrived |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Uvicorn bind address |
| `TB6612_AIN1`…`TB6612_STBY` | see `.env.example` | GPIO pins for the dual TB6612FNG (SparkFun) |
| `TB6612_STBY` | `21` | Driver standby pin (held high to drive, low when stopped) |

## Tests

```sh
.venv/bin/python -m pytest
```

Covers the steering math (`test_controller.py`) and the lenient JSON box
parsing plus the spend guard (`test_vision.py`). `test_camera.py` covers frame
counters, the MJPEG framing rules, and — against a stand-in `rpicam-vid` — that
a slow consumer is served the newest frame rather than a backlog.
`test_server.py` covers the adaptive pause, the inference-overlay state, and
the perception/control split: that steering outpaces a slow model, that the
deadman cuts the motors when a call hangs, that arrival ends the run, and that
nothing drives the motors after a stop.

No network and no API usage: the spend-guard tests assert the cap refuses the
call *before* it reaches the client.

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
and **STBY=21**. Perception is paced by API latency + `CONTROL_INTERVAL` while
steering runs at `CONTROL_HZ`, so the car keeps moving between model calls —
allow plenty of room and start slow. Point the web UI at the Pi's LAN address
and the video preview shows the car's view.

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

Steady-state cadence is exactly `pause + api` (no jitter). These numbers
predate the perception/control split and now describe the **perception**
cadence — i.e. how often a *new box* arrives, and therefore what you spend.
Steering updates are no longer tied to it: they run at `CONTROL_HZ` (default
10 Hz = 600/min) off whichever box is freshest.
