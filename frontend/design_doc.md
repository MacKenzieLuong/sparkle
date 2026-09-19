# Autonomous RC Car MVP — Frontend Design

## Goal and actual architecture

Build a React + TypeScript + Vite laptop dashboard for a Raspberry Pi 5 RC car with Camera Module 3. The laptop and Pi share a hotspot. A user speaks a high-level navigation target; the dashboard sends it to the Pi, displays the camera view, and shows robot state.

This design follows the **implemented backend** in `backend/server.py`. The backend uses HTTP for commands and status, plus HTTP/MJPEG for video. It has **no WebSocket server**. The frontend is currently a Vite starter, so the dashboard, microphone interaction, and API integration remain to be built.

Perception is initiated by the Pi control loop: it reads the Pi camera, then calls `vision.detect(target, frame)`. With `MOCK=false`, the Pi sends a JPEG frame to a remote inference API. With `MOCK=true`, fake vision returns scripted detections. The laptop does not perform inference or send detections to the Pi in the current system.

```text
Speech -> browser recognition -> validated target -> POST /direct -> Pi
Pi camera -> GET /video (MJPEG) -> browser
Pi camera -> Pi vision call -> Pi steering -> motors
Pi state -> GET /status (polling) -> browser
```

The frontend must not calculate steering or throttle, control motors directly, perform detection, or encode the Pi camera stream.

## Backend communication contract

The Pi FastAPI server defaults to port `8000` (`HOST` and `PORT` are backend settings). Use a configurable host and port in the frontend.

| Method | Path | Request | Response and meaning |
| --- | --- | --- | --- |
| `POST` | `/direct` | JSON `{"target":"blue flag"}` | `{"started":true,"target":"blue flag"}`. Starts or updates the control loop; a blank target returns HTTP 400. |
| `POST` | `/stop` | No required body | `{"stopped":true}`. Stops the loop and driver. |
| `GET` | `/status` | None | Current state snapshot as JSON. The client must poll. |
| `GET` | `/video` | None | MJPEG response with media type `multipart/x-mixed-replace; boundary=frame`. |

The backend also serves a test UI at `/` and mock-only `/debug/*` endpoints; the operator dashboard does not need those debug routes.

Example `/status` response shape:

```json
{
  "running": true,
  "target": "blue flag",
  "status": "moving",
  "cycle": 2,
  "missed": 0,
  "last_command": {
    "left": 0.35,
    "right": 0.15,
    "status": "moving",
    "note": "dx 0.25, area 0.10",
    "label": "blue flag",
    "box_2d": [300, 600, 700, 850]
  },
  "driver": null,
  "mock": false
}
```

`last_command` is initially `null`. `driver` is a fake-driver throttle pair in fake mode and `null` with the real driver. Expected status values include `idle`, `starting`, `moving`, `arrived`, `target_lost`, and `stopped`; inference exceptions may set a string beginning `error:`. `running` indicates whether the control loop is active. Treat fields defensively, including nulls and future additions.

The backend does **not** provide command IDs, acknowledgement events, structured error codes, Pi event timestamps, or decision events. An HTTP 200 from `/direct` means the request was accepted; it does not mean the target was detected or reached. Show recognition, HTTP request outcome, and later robot status as separate steps. If a command request times out or loses its response, show `Outcome unknown`, refresh `/status`, and do not automatically resubmit it. Without command IDs, a status snapshot cannot always resolve whether that particular request was accepted; hold subsequent dispatch while its outcome remains uncertain.

## Navigation queue and backend semantics

Navigation commands must run sequentially in arrival order. A command received while the robot is working joins the queue; it must not replace or interrupt the active target. Continue accepting valid navigation speech while another task is active. The frontend owns an in-memory queue and dispatches through `/direct` only when the previous task is confirmed finished. The existing backend has no queue: calling `/direct` while running replaces the target.

Voice stop pauses navigation and queue dispatch, preserving the interrupted target and all pending targets. Refreshing the page clears the frontend queue; it does not itself guarantee that a currently moving robot stops, because its control loop lives on the Pi. On page load, read `/status` before dispatching any new work. The backend has no resume endpoint: resuming the interrupted target requires sending it again through `/direct`. The explicit resume interaction and behavior after a failed task or connection loss remain to be confirmed. Never interpret `stopped` as permission to automatically dispatch the next queued target.

`find` follows the existing backend detection behavior: there is no search maneuver. Three consecutive missed detections terminate the loop as `target_lost`. `arrived` follows the existing controller threshold: a detection bounding box covering at least 50% of the normalized image area stops the robot. Do not add a new search or distance-based arrival algorithm in the frontend.

## Dashboard and component boundaries

The dashboard should contain a live camera view, current target/state, backend reachability, a chronological state-change log, and a two-press microphone control. Keep API code out of presentational components. Suggested structure:

```text
src/
  components/{CameraView,DecisionLog,RobotStatus,ConnectionStatus,PushToTalk}/
  hooks/{useRobotStatus,usePushToTalk}.ts
  services/{robotApi,speechRecognition}.ts
  types/{robot,speech}.ts
  config/environment.ts
  mocks/mockRobot.ts
  App.tsx
```

Use React hooks and native browser APIs. No manual driving, frontend motor controls, database, persistent logs, or object detection are needed for the MVP. Keep `CameraView` isolated so a future video transport change does not affect status and voice components.

## Configuration and browser networking

Example frontend configuration:

```env
VITE_MOCK_MODE=true
VITE_ROBOT_HOST=192.168.1.24
VITE_ROBOT_PORT=8000
VITE_STOP_WORD=stop
```

Derive the API origin and `/video` URL in `config/environment.ts`. The current backend stream is `/video` on the same port as the API, not `/video_feed` on port 5000. Do not put the Pi IP in UI components; hotspot addresses may change.

For the laptop MVP, run Vite locally with `npm run dev` and open the dashboard on localhost. Configure its development proxy to forward `/direct`, `/stop`, `/status`, and `/video` to the configured Pi host and port. Frontend requests and the camera image use those relative paths through Vite. This avoids requiring backend CORS changes for the demo. The current Vite configuration does not yet implement this proxy. Select and test the actual laptop browser's speech recognition before the demo; production hosting is outside this MVP.

Test that the hotspot permits laptop-to-Pi traffic and that video and status polling work simultaneously.

## Camera view

Use an `<img>` with the configured `/video` URL for MJPEG. Preserve aspect ratio. Show loading, unavailable, and retry states without crashing the rest of the dashboard. Handle image errors; note that a stalled stream after a successful load may require an additional health check because an `<img>` does not expose per-frame timestamps. Mock mode may show a local prerecorded video or placeholder. Do not send video through a command channel.

A camera-view failure alone does not stop navigation or prevent queued navigation commands. Distinguish this from losing the robot connection: on robot connection loss, suspend the video request and replace the view with a `Lost connection` screen. MJPEG has no playback pause API, so suspend it by removing the image source and reconnect later. Camera recovery and robot connection recovery must be handled separately.

## Voice grammar and request mapping

First microphone press starts browser speech recognition; second press stops it and processes the transcript. Keep speech APIs behind `speechRecognition.ts`. Show listening state, transcript, interpreted target, unsupported-command feedback, microphone/browser errors, HTTP request outcome, and latest robot status.

Stop is a voice command. Its keyword is configurable through `VITE_STOP_WORD`, initially `stop`, so it can be changed without editing the parser. Detect a standalone stop command while the microphone is listening and act as soon as it is recognized, without waiting for the second press. Pause local navigation dispatch immediately, preserve the interrupted and pending targets, and call `/stop` outside the queue. Suppress any pending navigation transcript from that listening session so it cannot restart movement after stop. Do not require stopping or clearing the camera preview for a voice pause. Recognition only operates during an active microphone session; this is not an always-listening stop detector.

Recognize navigation verbs such as `go`, `drive`, `navigate`, `find`, `head`, and `move`, and extract a target description. Flags and colors in this document are examples/placeholders, not a confirmed vocabulary restriction. The backend accepts any nonblank target string, embeds it into a vision prompt, and uses the first parsed detection; it does not validate that the object is a flag or that a color belongs to an enum. Fake vision replays boxes regardless of target text. Confirm whether the frontend should accept general object descriptions or a configured target allowlist. Reject ambiguous speech such as “go over there,” negated commands, and conflicting targets; do not forward raw speech transcripts.

For a flag example, the frontend can parse a target description and translate it to the **actual backend wire format**:

```text
"Go to the blue flag" -> { target: "blue flag" }
                      -> POST /direct { "target": "blue flag" }
```

The API receives only the extracted text target. Keep any frontend vocabulary restrictions configurable once the supported targets are agreed.

## Status, connection, and log

Poll `/status` while the page is open. One second matches the backend test UI and is a starting interval. Avoid overlapping requests and cancel work on unmount. Display `connecting`, `connected`, or `disconnected` as **HTTP reachability**, not WebSocket state. On poll failure, retain the last status but visibly mark it stale; retry automatically with bounded backoff.

When the robot connection is lost, the required behavior is to stop the car, suspend the stream, and show `Lost connection`. A frontend `/stop` request cannot guarantee delivery over a broken connection. The Pi needs a heartbeat/lease watchdog or equivalent local mechanism that stops the motors when the laptop stops checking in. That mechanism does not exist in the current backend. Define the connection-loss timeout and reconnection/queue-resume policy before implementing it. A working HTTP server also does not prove perception is healthy; detection staleness requires a separate backend check.

The backend exposes snapshots, not a decision history. The frontend may log *observed state changes* and actions it initiated, such as command submitted, `starting`, `moving`, `arrived`, `target_lost`, `stopped`, and connection loss/restoration. Use laptop receipt time and label entries as frontend-observed. Do not claim they have Pi-generated timestamps or represent every robot decision: polling can miss intermediate transitions. Keep at most 200 entries, retain them across disconnects, and reset them on page refresh. Do not create a log entry for every poll or throttle change.

For this MVP, observed state changes with the successful status response's laptop receipt timestamp are sufficient. Authoritative Pi decision events are not required.

## Mock mode and tests

`VITE_MOCK_MODE=true` must run without a Pi, physical camera, inference API, or WebSocket. A mock service should implement the same interface as `robotApi.ts` and return the same response/status shapes. Keep mock switching in the service or hook, not in presentational components. Exercise the real browser microphone where supported; simulate HTTP success and subsequent robot status changes for valid speech.

Test startup with and without the Pi; status failure and recovery; malformed status responses; camera load failure while navigation continues; refresh; bounded logs; supported and unsupported speech; configurable stop keyword; voice stop bypassing queued tasks; commands received during navigation; unknown request outcomes without automatic resubmission; unsupported recognition APIs; microphone permission denial; second-press end of listening; and video plus polling plus real inference under hotspot load. Add connection-loss motor-stop and queue recovery tests once the missing backend behavior is implemented.

## Robot-side gaps and decisions

1. **Stop race:** an in-progress inference can currently finish and apply a motor command after `/stop`. The backend should recheck active state before applying a result.
2. **Motor timeout:** the backend holds its last command through inference and the five-second default control interval. A separate motor failsafe is needed for stale or hung inference before physical operation.
3. **Queue lifecycle:** the frontend owns the in-memory queue; voice stop pauses it and refresh clears it. Confirm the resume interaction and handling of failed tasks/disconnection. Successful tasks should advance in arrival order.
4. **Connection-loss stop:** implement a Pi-local watchdog; specify its timeout and whether reconnection requires an explicit resume command.
5. **Voice details:** confirm general target descriptions versus an allowlist, the resume command, and behavior when speech recognition ends before the second press. Voice stop acts immediately when recognized during listening.
6. Confirm the Pi's reachable address and the laptop browser. Perception stays on the existing Pi-initiated path, and logs use frontend receipt timestamps.

## MVP acceptance criteria

- The dashboard displays `/video` and handles detectable camera failure.
- It polls `/status`, shows target/state, marks status stale on disconnection, and recovers without clearing logs.
- Robot connection loss stops the car through a Pi-local mechanism, suspends the stream, and shows `Lost connection`; camera-view failure alone permits navigation to continue.
- Navigation commands received while moving queue in arrival order without replacing the active task.
- The configurable voice stop keyword acts during listening, bypasses the queue, calls `/stop`, and preserves the paused queue. Refresh clears the in-memory queue.
- The bounded log displays frontend-observed changes without presenting them as authoritative Pi decisions.
- The microphone starts on the first press and stops on the second; transcript and errors are visible.
- Supported speech yields a canonical text target sent through `POST /direct`; unsupported speech sends nothing.
- The UI distinguishes local recognition, HTTP request result, and later robot state.
- An uncertain command response shows `Outcome unknown` and refreshes status without automatic command resubmission.
- Mock mode runs with `npm run dev` without robot services and uses the same UI and data shapes.
- The frontend performs no steering, motor control, or object detection.
