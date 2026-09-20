# Sparkle frontend

React dashboard for the RC car: camera stream, command queue, observed status log, and microphone.

## Local frontend/backend demo

In a backend terminal:

```sh
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
MOCK=true SPEECH_PROVIDER=fake CAMERA=fake DRIVER=fake FAKE_SCENARIO=approach CONTROL_INTERVAL=1 .venv/bin/python server.py
```

In a frontend terminal:

```sh
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open the localhost URL printed by Vite. `VITE_MOCK_MODE=false` uses real HTTP communication with the backend. The local backend above uses synthetic video, fake motors, and **scripted speech**: recordings always yield `FAKE_VOICE_TRANSCRIPT` (default “go to the blue flag”). This tests transport and queue behavior, not transcription.

The first mic press starts recording; the second submits it. Recording ends automatically after 10 seconds. The mic is disabled during permission setup and processing. Audio is mono PCM16 WAV, processed only by the backend. No browser speech recognition or frontend API key is used.

## Connect to the Pi / actual Qwen speech

Set `ROBOT_API_URL=http://<pi-address>:8000` in `.env.local` and restart Vite. The laptop and Pi must be reachable on the same network. Open the dashboard on **localhost** so browser microphone access works. The Vite proxy forwards relative API and video URLs; no backend CORS configuration is needed for this development setup.

On the backend, set `SPEECH_PROVIDER=qwen` and `HUAWEI_API_KEY` to use real speech while keeping `MOCK=true` for simulated vision. The provider URL defaults to `https://yibuapi.com/v1`; the voice model defaults to `qwen3.8-omni-flash`. Set `MOCK=false` for real vision separately. See the backend README for camera/motor configuration. Never put provider keys in `VITE_*` variables.

One audio interpretation request runs at a time. Navigation can continue while audio is recorded/processed. Only validated targets enter the queue. Arrival advances the queue; failed navigation pauses it. Stop/resume phrases come from backend capabilities. Clip recognition has inference latency, including for stop.

Lost HTTP responses show **Outcome unknown**, pause dispatch, and check command receipts without resubmitting. Connection loss pauses the queue and removes the camera stream. The backend stops session-controlled navigation after heartbeat expiry. Reconnection does not resume automatically. Refresh clears the browser queue; it does not instantly stop the robot.

## Standalone layout demo

Set `VITE_MOCK_MODE=true` to use editable demo transcripts, the illustrated camera, and simulation controls without a backend or microphone. `VITE_STOP_WORD` affects this layout demo only.

## Verification

```sh
npm test
npm run build
npm run lint
```

With the local fake backend and Vite running, test the full proxy path from the repository root:

```sh
backend/.venv/bin/python backend/smoke_test.py --url http://127.0.0.1:5173
```

The smoke test requires fake speech, fake vision, and a fake driver. It submits generated WAV, dispatches a target, sends heartbeats, and checks arrival and the command receipt. Browser microphone permission, physical hardware, and actual gateway transcription require separate testing.
