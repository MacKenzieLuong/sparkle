#!/usr/bin/env bash
# Live demo: real camera, real model, real motors.
#
#   export HUAWEI_API_KEY=<key>
#   ./run-demo.sh
#
# Every value below can be overridden from the environment, e.g.
#   BASE_SPEED=0.3 ./run-demo.sh
#   DRIVER=fake ./run-demo.sh      # dry run, wheels disconnected
set -euo pipefail
cd "$(dirname "$0")"

: "${HUAWEI_API_KEY:?set HUAWEI_API_KEY first (and rotate it if it has ever been pasted anywhere)}"

# --- what is real -----------------------------------------------------------
export MOCK="${MOCK:-false}"
export CAMERA="${CAMERA:-rpicam}"
export CAMERA_ROTATION="${CAMERA_ROTATION:-180}"
export DRIVER="${DRIVER:-tb6612}"

# --- how fast ---------------------------------------------------------------
# 3.4s round trip means the car acts on where things were 3.4s ago. Slow.
export BASE_SPEED="${BASE_SPEED:-0.2}"
export TURN_GAIN="${TURN_GAIN:-0.3}"
export SEARCH_SPEED="${SEARCH_SPEED:-0.25}"

# --- pacing and limits ------------------------------------------------------
export CONTROL_INTERVAL="${CONTROL_INTERVAL:-0}"   # poll as fast as latency allows
export CONTROL_HZ="${CONTROL_HZ:-10}"
export STALE_AFTER="${STALE_AFTER:-8}"             # max seconds driving on one decision
export MAX_RUN_SECONDS="${MAX_RUN_SECONDS:-120}"   # backstop if /stop is unreachable
export MAX_COST_USD="${MAX_COST_USD:-0.25}"

exec .venv/bin/python server.py
