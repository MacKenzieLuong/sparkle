# Sparkle

Hack the North 2026 RC car dashboard and robot backend.

## Run locally

Start from the repository root. Use two terminals so both servers stay running:

```sh
# Terminal 1
./run-backend-local.sh
```

```sh
# Terminal 2
./run-frontend-local.sh
```

The backend script loads `backend/.env` and runs the Python server. The frontend
script runs `npm install` and starts Vite. Stop either server with Ctrl+C.

First-time setup, if dependencies are not installed yet:

```sh
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ../frontend
npm install
```

Copy `backend/.env.example` to `backend/.env` and edit it for your backend
settings. The backend does not load this file automatically; the commands
below export its values. Keep your API key here, never in a `VITE_*` frontend variable. See
[backend/README.md](backend/README.md) for backend modes and provider settings.

The equivalent manual commands are:

**Terminal 1 — backend** (from the repository root):

```sh
cd backend
set -a
source .env
set +a
.venv/bin/python server.py
```

**Terminal 2 — frontend** (from the repository root):

```sh
cd frontend
npm install
npm run dev
```

Open the URL Vite prints, usually <http://127.0.0.1:5173/>. The frontend
proxies API requests to the backend; for local testing it should use
`ROBOT_API_URL=http://127.0.0.1:8000` and `VITE_MOCK_MODE=false` in
`frontend/.env.local` if those values are not already set. Restart Vite after
changing its environment file.

For more detail, see [frontend/README.md](frontend/README.md).
