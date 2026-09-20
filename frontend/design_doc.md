# Autonomous RC Car MVP — Frontend Design

## Goal and architecture

A React + TypeScript + Vite laptop dashboard controls a Raspberry Pi RC car on the same local network. The user records a navigation command; the backend obtains its transcript and meaning from Qwen, validates the result, and returns a structured intent. The browser owns the pending navigation queue. The Pi owns camera acquisition, perception requests, steering, and motor execution.

```text
Laptop microphone → PCM16 WAV → Vite proxy → Pi /voice/interpret
Pi → Qwen Chat Completions → transcript + intent JSON → validation
Validated intent → browser queue → Pi /direct → camera/vision/controller/motors
Pi /video → Vite proxy → browser camera view
Browser heartbeat + status polling ↔ Pi control loop
```

The HTTP audio, status, queue dispatch, proxy, and watchdog integration is implemented. Automated tests use fake providers; real gateway audio and physical hardware remain unverified. The layout-only mock is also retained.

## AI responsibility and API style

All transcription, language interpretation, prompts, provider credentials, and intent validation belong to the Python backend. Actual Qwen inference runs at the configured remote provider; model weights are not installed on the laptop or Pi. The frontend captures/encodes audio and displays results; it performs no production speech parsing, vision inference, steering, or throttle calculations.

Both vision and speech use the existing OpenAI-compatible **Chat Completions** API via the OpenAI Python SDK. Default provider URL: `https://yibuapi.com/v1`. Default model: `qwen3.8-omni-flash`. `HUAWEI_VOICE_MODEL` can override the voice model independently. Keep the provider key on its corresponding endpoint.

Speech sends a completed WAV as base64 `input_audio`, requests text output, collects the streaming HTTP response, and validates JSON. `stream=True` streams the response; it does not make microphone capture realtime. Realtime access is not required for queued navigation. Voice uses async I/O independently of the threaded vision loop, with no shared conversation history.

Reference: [Qwen-Omni HTTP documentation](https://www.alibabacloud.com/help/en/model-studio/qwen-omni). The actual sponsor gateway/account still needs a short audio compatibility check.

## Browser recording and interpretation

- First press requests microphone access and begins recording. Second press finishes and submits.
- Recording automatically ends at **10 seconds**, enforced by audio sample counting and a timer fallback.
- AudioWorklet captures mono PCM; the browser writes a complete PCM16 WAV (requested 16 kHz). No WebM conversion service or browser SpeechRecognition is used.
- Mic tracks are released on completion, cancellation, disconnection, and unmount.
- Show requesting, listening/countdown, processing, transcript, rejection, and error states.
- Disable the mic during permission setup and processing. A new clip may start only after the previous result is handled; no arbitrary cooldown is needed. Although recording and an earlier HTTP request could technically overlap, this MVP deliberately allows only one at a time.
- The backend independently allows one audio request at a time and returns HTTP 409 if busy.
- Navigation already in progress continues while audio is recorded or processed. Rejected/failed clips add no task.
- Discard results from canceled capture sessions. Deduplicate successful results by request ID.

The backend requests one intent: `navigate`, `stop`, `resume`, or `reject`. Navigation requires a nonempty concrete object description, bounded to 200 characters. Flags are examples, not a fixed object enum; preserve details such as “small red ball.” The prompt rejects unclear audio, silence, ambiguous references, negative commands, multiple targets/commands, and unrelated speech. Strict schema validation rejects malformed model output; conservative transcript checks also reject selected ambiguous patterns. Model semantic accuracy still requires real recordings to evaluate.

Unclear commands show: **“The transcription is unclear, so this command cannot be accepted. Please try again.”** No guessed navigation is queued.

`STOP_WORD` and `RESUME_WORD` are backend settings, returned by `/capabilities`. These control phrases must match the standalone normalized transcript. The live frontend does not interpret keywords. The layout mock retains its own demo-only parser.

Example result:

```json
{
  "requestId": "voice-123",
  "transcript": "Go to the blue flag",
  "intent": "navigate",
  "target": "blue flag",
  "reason": null,
  "message": null,
  "simulated": false
}
```

## HTTP contract

The backend defaults to port 8000. There is no WebSocket server.

| Method | Path | Contract |
| --- | --- | --- |
| GET | `/capabilities` | `voiceMode` (clip/disabled), `speechProvider`, `maxRecordingSeconds:10`, `audioFormats`, stop/resume words, receipt support, heartbeat interval |
| POST | `/voice/interpret` | Raw WAV body, `Content-Type: audio/wav`, `X-Request-ID` (1–100 chars). Returns the validated interpretation. Side-effect free: never starts/stops motors or edits a queue. |
| POST | `/direct` | JSON `{target, commandId, sessionId, expectedRevision, resume?}`. Starts one task; returns `{started:true,target,commandId,duplicate}`. |
| POST | `/stop` | Stops the loop/driver, increments revision, and sets the pause latch. Returns `{stopped:true}`. |
| POST | `/resume` | `{expectedRevision}`. Clears an idle pause latch without movement; returns `{resumed:true}`. |
| POST | `/heartbeat` | `{sessionId}`. Refreshes the active session lease. |
| GET | `/commands/{commandId}` | Latest in-memory receipt: command_id, target, session_id, status, running. Unknown ID returns 404. |
| GET | `/status` | Pollable snapshot; see below. |
| GET | `/video` | HTTP MJPEG (`multipart/x-mixed-replace; boundary=frame`). |

Audio validation: mono PCM16 WAV, 8–48 kHz, at least 0.1 seconds, at most 10 seconds/2 MB, and complete frame data. Audio stays in request memory; no persistence. Upload timeout: 15 seconds. Interpretation timeout: 30 seconds. Provider client timeout: 25 seconds, with automatic SDK retries disabled.

Operational errors use FastAPI's `detail` envelope containing a `code` and, where valid, `requestId`: 408 upload timeout, 409 speech busy, 413 size/duration, 415 content type, 422 invalid WAV/request ID, 503 unavailable/provider failure, 504 interpretation timeout. Provider messages, request bodies, and credentials are not returned. A valid rejection differs from an operational error.

`/status` includes:

```text
running, target, status, cycle, missed, last_command,
command_id, session_id, revision, paused, instance_id, driver, mock
```

`last_command` is initially null, otherwise includes left/right throttle, status, note, label, and box_2d. `driver` is a throttle pair for the fake driver and null for physical hardware. Statuses include idle, starting, moving, arrived, target_lost, stopped, connection_lost, and error. `instance_id` changes on backend restart; receipts are not durable.

Read the current revision before dispatch. Busy, stale-revision, or paused-without-resume requests return 409. Duplicate command IDs with the same target/session return an acknowledgement without restarting movement; conflicting reuse returns 409. The backend retains the most recent 200 accepted command IDs.

The legacy `/direct {target}` request remains for the backend test page. It has no session heartbeat protection. The React dashboard uses the full session contract. The demo assumes a trusted local network and has no authentication.

## Queue, failures, and uncertain outcomes

The browser owns an in-memory FIFO queue; the backend executes one active task. New navigation speech received while moving joins the queue. A direct call while busy is rejected rather than replacing the active target.

Only a correlated successful `arrived` completion automatically advances the queue. `target_lost`, backend errors, unexpected stop, connection loss, and restart pause dispatch and preserve pending targets. Resume retries an interrupted target with a new command ID before pending tasks, using `resume:true` and the current revision. An empty paused queue uses `/resume` to become ready again.

`find` uses existing backend detection behavior; there is no search maneuver. Three missed detections stop with `target_lost`. `arrived` follows the controller's existing bounding-box area threshold (at least 50% of normalized image area). The frontend adds no distance-based arrival logic.

If a direct request loses its response or times out, display **Outcome unknown**, pause, refresh status, and look up its command receipt. Do not automatically resend it. A 404 receipt is not proof that a delayed request cannot still arrive. Recovered receipts leave the queue paused until explicit resume. If the outcome remains unresolved, stop before resuming. Backend restart invalidates prior receipt history and pauses the UI.

Refresh clears the browser queue/log, creates a new session, and reads backend status before dispatch. It does not instantly stop an existing task; the old session lease expires if its page is gone. An active task belonging to another session is observed without taking over automatically.

## Stop and connection handling

Clip-based speech cannot recognize stop while audio is still being recorded. Recognition waits for capture, upload, and model processing. Once a validated stop result arrives, pause frontend dispatch immediately and call `/stop` outside the navigation queue. Preserve interrupted/pending tasks. Do not claim a stop succeeded if its HTTP outcome is uncertain.

The backend increments a generation on stop and rechecks it before applying inference results. Revision checks reject stale dispatch; an explicit resume handshake clears the pause latch. Camera, inference, and controller exceptions stop the current task. Initialization creates one hardware/provider set.

The frontend sends a heartbeat and polls status about once per second without overlapping polling cycles. The backend independently checks the active session lease every 0.1 seconds; default expiry is 3 seconds. On expiry it stops the driver and reports `connection_lost`. This watchdog does not wait for Qwen.

On HTTP connection failure, retain the queue/log but pause dispatch, disable recording, remove the MJPEG image, and show a lost/unavailable connection screen. Retry polling automatically. Reconnection restores observability, never automatic movement. Heartbeat monitoring covers browser connectivity, not perception freshness or backend process failure.

## Dashboard and camera

Keep the white background, black straight borders, and monospace layout. Camera and mic occupy the left column; queue and a large scrolling status log occupy the right. Avoid the redundant control/robot-state/current-target bottom strip. No additional instructions in the corner.

Live mode preserves the complete dashboard when unreachable: placeholder values, unavailable camera message, and disabled mic. It displays no layout-mock labels. When connected, `<img src="/video">` shows MJPEG with an unavailable/retry fallback. A camera-view error alone permits navigation to continue. A stream stall after loading cannot reliably be detected from image error events; per-frame freshness monitoring is outside this implementation.

Log observed status changes and actions with the laptop response-receipt time. Keep at most 200 entries; retain them over disconnect and clear on refresh. Polling may miss intermediate transitions. These are frontend observations, not authoritative Pi event timestamps or a complete decision history.

## Implemented structure

```text
frontend/src/
  components/VoiceControl.tsx       live microphone UI
  components/CameraView.tsx         illustration or MJPEG/error view
  hooks/useRobotConnection.ts       React subscription/lifecycle
  services/audioCapture.ts         microphone + WAV encoding
  services/robotApi.ts              HTTP client and response checks
  services/robotConnection.ts       queue, status, receipts, session lifecycle
  types/api.ts                     wire contract
frontend/public/pcm-recorder.js     bounded AudioWorklet recording
frontend/vite.config.ts             API/video proxy
backend/speech.py                   audio limits, Qwen adapter, strict intent validation
backend/server.py                   routes, active task, receipt cache, pause/watchdog
backend/vision.py                   existing image Chat Completions adapter
backend/controller.py               existing deterministic steering math
```

## Configuration and launch

Frontend `.env.local`:

```env
VITE_MOCK_MODE=false
ROBOT_API_URL=http://127.0.0.1:8000
```

For the car, replace the URL with `http://<pi-address>:8000` and restart Vite. All API calls and `/video` use relative paths forwarded by Vite. Open the dashboard at localhost on the laptop for microphone access; a remote plain-HTTP dashboard would not be a secure microphone context. This development proxy avoids a CORS requirement. Production hosting is outside the MVP.

Backend settings are exported shell variables; `.env` is not loaded automatically. `SPEECH_PROVIDER=fake|qwen|disabled` selects speech. `HUAWEI_API_KEY`, `HUAWEI_BASE_URL`, `HUAWEI_MODEL`, and optional `HUAWEI_VOICE_MODEL` stay backend-only. `MOCK` selects fake/real vision independently; `CAMERA` and `DRIVER` select hardware providers. See both READMEs and `.env.example` files for launch commands.

Use `MOCK=true SPEECH_PROVIDER=fake CAMERA=fake DRIVER=fake FAKE_SCENARIO=approach` for local integration. Fake speech returns a labeled scripted transcript, regardless of the recording; it verifies transport only. Use `SPEECH_PROVIDER=qwen` with fake hardware to test actual audio safely before connecting motors. `VITE_MOCK_MODE=true` instead selects the standalone layout demo with editable text and no microphone/backend.

## Verification and remaining gates

Automated backend checks cover WAV validation, interpretation rejection, the Qwen request shape, direct/receipt correlation, deduplication, stale requests, watchdog behavior, and late inference after stop. Frontend checks cover queue order, failure pause/retry, uncertain responses, reconnection, and the 10-second sample limit. The local smoke test submits generated WAV through Vite, dispatches its interpreted target, maintains a heartbeat, and checks arrival/receipt using fake providers.

Remaining real integration checks:

1. Test a spoken WAV against the actual gateway/account and `qwen3.8-omni-flash`; inspect transcript/intent quality and latency. Stub tests cannot establish account access or transcription accuracy.
2. Exercise microphone permission, actual recording, and playback-free capture in the intended laptop browser.
3. Confirm the Pi address/port and hotspot connectivity; run video, polling, and audio together.
4. Validate camera/motor hardware separately. Add a perception/motor freshness failsafe before physical driving: the current loop holds its last command during inference and the control interval. A heartbeat alone does not cover stale perception.
5. Measure clip-based stop latency; do not promise immediate spoken stop.
