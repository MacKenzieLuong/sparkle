#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/backend"

if [[ ! -f .env ]]; then
  echo "Missing backend/.env. Copy backend/.env.example to backend/.env and edit it." >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing backend/.venv. Run: cd backend && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

set -a
source .env
set +a
exec .venv/bin/python server.py
