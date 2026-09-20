# Sparkle Backend — RC Car Autopilot

Give a Raspberry Pi RC car a goal in plain language — "go to the red ball" —
and it works out how to get there. A multimodal model looks through the camera
and decides what the car should do next; local code executes that decision
smoothly and decides when to stop trusting it.

## Architecture

A model call takes seconds; steering must not. So `POST /direct` starts two
threads that share one "latest decision" record:

```
User types "go to the red ball" on web page
        │  POST /direct {"target": "..."}
        ▼
   FastAPI server (on the Pi)
        │
        ├── perception thread ── every CONTROL_INTERVAL, immediately on Go:
        │     grab frame from camera ──────────► Pi Camera / rpicam-vid / fake
        │     ask the model what to do ────────► OmniVision (yibuapi) or fake
        │       sees: frame + goal + its own recent actions
        │       answers: {"action": "approach", "box_2d": [...], "reason": ...}
        │            │
        │            ▼
        │     ┌────────────────────────┐
        │     │ latest decision + when │  ◄── shared, lock-guarded
        │     └────────────────────────┘
        │            │
        └── control thread ── every 1/CONTROL_HZ (default 10 Hz):
              decision older than STALE_AFTER? → motors off
              else controller.act(action, box) ──► left/right throttle in [-1, 1]
                     approach → steer at the box, slow as it fills the frame
                     search_* → rotate in place to look around
                     back_off → reverse;  stop → hold still
                     │
                     ▼
              Motor driver: H-bridge (dual TB6612FNG) on GPIO or fake (logs)
```

The car keeps acting at a steady 10 Hz on the last decision it received,
instead of freezing for the whole duration of each model round trip. If
perception stalls — hung request, dead camera, dropped network — the decision
goes stale and the control thread cuts the motors without waiting for the
request to time out.

## Layout

| File | Purpose |
| --- | --- |
| `server.py` | FastAPI app, MJPEG stream, `ControlLoop` (perception + control threads), debug endpoints |
| `cli.py` | One-shot detection, camera listing and motor test from the terminal |
| `run-demo.sh` | The full live configuration in one command |
| `vision.py` | `VisionProvider` interface, `FakeVision` (scripted), `OmniVision` (asks the model what to do) |
| `controller.py` | Executes the model's action under local speed limits; `command()` is the pure steering math |
| `tracker.py` | Optical-flow box tracking that bridges the gap between model replies |
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

## Tracking between model calls

The model answers every few seconds. Steering on a box that old is the real
cause of weaving and overshoot, so the box it returns seeds feature points that
are followed locally frame to frame with optical flow. Measured on a Pi 5, with
a mocked 3s model:

| | box the car steers on |
| --- | --- |
| `TRACK=false` | 1.95 s old on average, 2.50 s at worst |
| `TRACK=true` | **0.04 s old** |

That costs 0.83 ms per frame — under 2% of one core at 20 Hz — against a
640x480 JPEG decode at 2.75 ms which the camera read already pays.

Tracking is only ever a bridge, and it is treated as one. Optical flow cannot
tell you a target has *gone* — points that drift onto the background keep
tracking beautifully, and a confident, centred box the car drives at is exactly
what that looks like from the outside. It was observed on the floor: the car
turned past the target and kept going, steering at scenery. So:

- a seed is followed for at most `TRACK_MAX_AGE`, after which the model must
  confirm the target again — the only check that catches background-following,
  since surviving-point count stays high throughout
- the box may not wander more than `TRACK_MAX_DRIFT` of the frame from where
  the model put it, nor end up mostly outside the frame
- every new detection re-seeds it, so drift never accumulates across replies
- each update reports confidence from surviving points and a forward-backward
  error check; below `TRACK_MIN_CONFIDENCE` it reports failure instead of a
  plausible-looking box, and the control thread falls back to the model's
- staleness is still measured against the **model**, never the tracker: flow
  will happily follow a box long after the model stopped confirming that the
  target is there at all

`/status` shows `tracking` (confidence, `lost`, or `seeded`), `detection_age`
for the box being steered on, and `model_age` for the last real reply.

## Why calls can overlap

One model call takes 2.5-4.3s and that cannot be reduced: Qwen's Realtime
WebSocket was probed against yibuapi's own documented protocol and will not
accept a local frame in any of nine content shapes, verified by input-token
count rather than by the absence of an error (see `realtime_probe.py`). Its
`image_url` field wants a fetchable http(s) URL, which would mean uploading
every frame — slower, not faster.

So the only way to hear from the model more often is to have several requests
in flight. `VISION_CONCURRENCY` workers each hold one, staggered by
`VISION_STAGGER`, and the decision rate rises almost linearly:

| workers | decision every | cost |
| --- | --- | --- |
| 1 | ~3.4 s | 1x |
| 2 | ~1.7 s | 2x |
| 3 | ~1.1 s | 3x |

With tracking on, this is rarely worth paying for: the model only has to
confirm the target and correct drift, which one request in flight does fine.

Replies then finish out of order. Every result is keyed on when its frame was
*captured*, and one describing an older frame is dropped rather than allowed to
replace a newer decision; `/status` counts those as `out_of_order`.

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

One model does both the seeing and the deciding: `qwen3.8-omni-flash` (a Qwen
Omni multimodal model) reached through the sponsor's OpenAI-compatible gateway
(yibuapi). It is a general multimodal LLM, not a purpose-trained detector like
YOLO — it is prompted to return a decision as JSON, parsed leniently
(`vision._parse_plan`), so approximate output degrades to a halt rather than
crashing the loop. Measured round trip on a Pi 5 with an imx708: **2.5–4.3 s**,
median ~3.4 s. To keep cost/latency down, only one
downscaled JPEG frame is sent per cycle and calls run at the 5s cadence.

Detection contract (same shape the steering math consumes):

```json
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}]
```

Coordinates are normalized to `0–1000` for a 1000×1000 reference frame
(`box_2d` order is `ymin, xmin, ymax, xmax`). An empty array means the target
is not in frame.

## What the model decides

The model is not a box detector wired to a formula. Each cycle it is shown the
camera frame, the goal, and its own recent actions, and it answers with a
decision:

```json
{"action": "approach", "box_2d": [607, 106, 999, 843],
 "label": "a chair", "reason": "chair ahead slightly left"}
```

| action | what the car does |
| --- | --- |
| `approach` | steer toward `box_2d` using the math below |
| `search_left` / `search_right` | rotate in place at `SEARCH_SPEED` to look for the goal |
| `back_off` | reverse at `BACK_OFF_SPEED` — blocked, or far too close |
| `stop` | hold still; ends the run once `ARRIVE_CONFIRM` replies agree |

The split matters because a model call takes seconds. **The model chooses what
to do; this code chooses how fast and for how long.** Speeds live in
`controller.act`, so no reply can make the car move faster than the machine was
configured to allow, an unrecognised action halts rather than being guessed at,
and an `approach` with no box becomes a `stop`. Sending its own recent actions
back to it is what stops a search rocking left-right-left; `SEARCH_LIMIT`
(default 8) bounds how long it may hunt before giving up as `target_lost`.

Replies in the older bare `[{"box_2d": ...}]` shape are still accepted and read
as `approach`.

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
4. **Staleness deadman.** If the freshest box is older than `STALE_AFTER`
   seconds (default 8), the control thread cuts the motors and reports
   `status="stale"`, leaving navigation running so it resumes the moment a box
   arrives.

   The window is deliberately an absolute number rather than a multiple of the
   measured cycle. Scaling it off the cycle was tried and cut the motors on
   every slow call: a call is only known to be slow once it has returned, so
   the box ages past a window sized on the previous, faster cycle while the
   slow one is still in flight. Live latency swung 2.5–4.3 s call to call,
   which made that stutter constant.

   Set it from measured latency — `cli.py -n 5` reports the spread — and treat
   it as *the longest the car may drive on one box*. `/status` reports both
   `stale_after` and the observed `cycle_time` for tuning.
5. Exit conditions, both owned by the perception thread so one bad frame
   cannot end a drive:
   - **arrived** — the box covers ≥ `ARRIVED_AREA_FRACTION` of the frame on
     `ARRIVE_CONFIRM` (default 2) successive detections. The car halts on the
     *first* one, because the steering math returns zero throttle for it, but
     the run only ends once another detection agrees. Live testing produced a
     hallucinated full-frame box between two good ones — under a single-frame
     rule that stops the car for good, mid-drive.
   - **target_lost** — no usable reply for 3 consecutive perception cycles,
     or `SEARCH_LIMIT` consecutive search actions without finding the goal.
     Camera and API errors count as misses, so a persistent failure ends the
     run rather than looping forever; the message lands in `status.error`.
   - **time_limit** — `MAX_RUN_SECONDS` (default 120) elapsed. If the network
     drops, `/stop` cannot be reached and nothing else bounds a drive that
     never arrives.
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

### Why the car weaves, and the fade that fixes it

The car acts on one decision until the next arrives seconds later. Holding a
turn for all of it keeps rotating long after the car already points at the
target, so it sails past and corrects back — weaving, at a rate no amount of
`TURN_GAIN` tuning fixes, because the problem is dead time rather than gain.

Turn authority therefore fades to zero over `TURN_DECAY` seconds. Forward
motion is untouched; only rotation fades:

```
 age   authority   left   right
 0.0s     1.00   +0.89  -0.05     turn hard, having just seen the target
 0.6s     0.50   +0.66  +0.18
 1.2s     0.00   +0.42  +0.42     coast straight until it sees again
```

Scans fade the same way, for the same reason: spinning for a whole cycle
sweeps the target straight back out of frame. `/status` reports the live value
as `last_command.turn_authority`.

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
| `VISION_WIDTH` / `VISION_HEIGHT` | `0` / `0` | Downscale before sending; `0` keeps camera resolution. 320x240 is 80 image tokens against 300 at 640x480 |
| `VISION_EXPLAIN` | `false` | Ask the model to justify its action. Readable while tuning, but output tokens are generated serially and cost latency on every call |
| `VISION_CONCURRENCY` | `1` | Requests in flight at once. `3` gives a decision ~3x as often, and costs 3x |
| `VISION_STAGGER` | `1.2` | Seconds between worker starts, so overlapping requests spread out instead of bunching |
| `TRACK` | `true` | Follow the box locally between model replies |
| `TRACK_HZ` | `20` | Tracker update rate |
| `TRACK_MIN_CONFIDENCE` | `0.4` | Surviving-point fraction below which tracking reports failure |
| `TRACK_MAX_AGE` | `1.5` | Seconds a seed may be followed before the model must re-confirm |
| `TRACK_MAX_DRIFT` | `0.30` | Fraction of the frame the box may wander from its seed |
| `TRACK_POINTS` | `80` | Features seeded inside each new box |
| `TRACK_FB_TOLERANCE` | `2.0` | Pixels of forward-backward error a point may have and still count |
| `MAX_COST_USD` | `1.00` | Estimated spend cap for the process; `0` disables it |
| `VISION_INPUT_USD_PER_MILLION` | `0.55` | Image-token input rate used by the estimate |
| `VISION_OUTPUT_USD_PER_MILLION` | `2.20` | Output-token rate used by the estimate |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `640` / `480` | Frame resolution |
| `CAMERA_ROTATION` | `0` | Rotate every frame `90`/`180`/`270` — for a camera mounted upside down |
| `CONTROL_INTERVAL` | `5` | Seconds between inference calls while the target is far; `0` polls as fast as model latency allows |
| `SHORT_INTERVAL` | `1` | Seconds between inference calls once the target is near (≥ `SHORT_INTERVAL_AREA`) |
| `SHORT_INTERVAL_AREA` | `0.15` | Box area fraction (of the 1000×1000 frame) that triggers the fast cadence |
| `CONTROL_HZ` | `10` | Steering updates per second, independent of inference cadence |
| `STALE_AFTER` | `8.0` | Seconds the car may drive on one box before the motors cut; must exceed your measured latency |
| `BASE_SPEED` | `0.5` | Forward throttle before area scaling — **lower this for a slow first test** |
| `TURN_GAIN` | `0.8` | Turn aggressiveness; lower it alongside `BASE_SPEED` |
| `TURN_DECAY` | `1.2` | Seconds over which a decision loses its turn authority; `0` disables the fade |
| `DEAD_ZONE` | `0.08` | Horizontal offset below which the car drives straight |
| `ARRIVED_AREA_FRACTION` | `0.5` | Box area fraction that counts as arrived |
| `ARRIVE_CONFIRM` | `2` | Successive detections that must agree before the run ends |
| `SEARCH_LIMIT` | `8` | Consecutive search cycles before giving up as `target_lost` |
| `MAX_RUN_SECONDS` | `120` | Hard cap on one drive; `0` disables. The backstop when `/stop` is unreachable |
| `SEARCH_SPEED` | `0.25` | Wheel speed when rotating in place to look around |
| `BACK_OFF_SPEED` | `0.2` | Reverse speed for `back_off` |
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

## Live demo runbook

Each step is cheap and reversible, and each one fails in a way that tells you
what is wrong. Do not skip ahead — every step below caught a real bug the first
time it was run.

**1. Camera, no model, no motors.** Zero spend.

```sh
.venv/bin/python cli.py --list-cameras
MOCK=true CAMERA=rpicam .venv/bin/python cli.py "anything" --save cam
```
`cam-1.jpg` must be a real photo. If the camera fails to acquire, check nothing
else holds it — libcamera reports a camera in use exactly like an absent one.

**2. Motors, no model, no camera.** Car on blocks, wheels free.

```sh
DRIVER=tb6612 .venv/bin/python cli.py --test-motors
```
It first prints the resolved pins and the gpiozero pin factory, then refuses to
run at all under the fake driver, so "nothing moved" can never be a silent
no-op.
Each step names a wheel and a direction; watch that the right wheel turns the
right way. Wrong direction → swap that motor's two direction pins. Wrong wheel
→ swap the A and B groups.

**3. Model, no motors.** A few cents.

```sh
export MOCK=false HUAWEI_API_KEY=<key> CAMERA=rpicam MAX_COST_USD=0.05
.venv/bin/python cli.py "a chair" -n 5 --interval 1 --raw --save look
```
Check the boxes land on the object in `look-*.jpg`, and note the latency
spread — that number sets `STALE_AFTER` and bounds a safe `BASE_SPEED`.

**4. Whole loop, motors disconnected.**

```sh
export DRIVER=fake CONTROL_INTERVAL=0 MAX_COST_USD=0.25 MAX_RUN_SECONDS=120
export BASE_SPEED=0.2 TURN_GAIN=0.3
.venv/bin/python server.py
```
Drive the target around by hand in front of the lens and watch `/status`:
`control_cycle` should outpace `cycle` about 10:1, `detection_age` should
sawtooth below `stale_after`, and `action` should switch to a search when you
take the target out of frame.

**5. Motors on blocks.** Same as step 4 with `DRIVER=tb6612`. The wheels now
follow the model's decisions with the car going nowhere.

**6. On the floor.** `./run-demo.sh` sets the whole live configuration in one
go — real camera, real model, real motors — so no single forgotten `export`
can leave you driving a simulation. Every value in it can be overridden:
`BASE_SPEED=0.3 ./run-demo.sh`, or `DRIVER=fake ./run-demo.sh` for a dry run.

The server prints a banner at startup saying what is real and what is fake;
`driver: FakeDriver <-- NOTHING WILL MOVE` is the one to look for. Start slow
and in open space. `MAX_RUN_SECONDS` is the
backstop if the car drives out of wifi range; the web UI's stop button is the
one you should actually be reaching for.

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
