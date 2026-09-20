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

# Values already in the environment win, so a dry run is just
# DRIVER=fake ./run-demo.sh. The catch is that an export left over from an
# earlier step wins in exactly the same silent way — a stale DRIVER=fake once
# cost a whole demo — so anything inherited gets named before we start.
inherited=()
setting() {
    local name="$1" default="$2"
    if [ -n "${!name:-}" ]; then
        inherited+=("$name=${!name}")
    else
        export "$name=$default"
    fi
}

# --- what is real -----------------------------------------------------------
setting MOCK false
setting CAMERA rpicam
setting CAMERA_ROTATION 180
setting DRIVER tb6612

# --- how fast ---------------------------------------------------------------
# 3.4s round trip means the car acts on where things were 3.4s ago. Slow.
setting BASE_SPEED 0.2
setting TURN_GAIN 0.3
setting SEARCH_SPEED 0.25

# --- pacing and limits ------------------------------------------------------
setting CONTROL_INTERVAL 0      # poll as fast as latency allows
setting CONTROL_HZ 10
setting STALE_AFTER 8           # max seconds driving on one decision
setting MAX_RUN_SECONDS 120     # backstop if /stop is unreachable
setting MAX_COST_USD 0.25

if [ ${#inherited[@]} -gt 0 ]; then
    echo
    echo "Using these from your shell instead of the demo defaults:"
    for pair in "${inherited[@]}"; do
        echo "    $pair"
    done
    echo "  (unset them for the standard live configuration)"
fi

exec .venv/bin/python server.py
