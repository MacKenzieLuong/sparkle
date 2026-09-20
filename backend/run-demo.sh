#!/bin/sh
# Live demo: real camera, real model, real motors.
#
#   export HUAWEI_API_KEY=<key>
#   ./run-demo.sh
#
# Every value below can be overridden from the environment, e.g.
#   BASE_SPEED=0.3 ./run-demo.sh
#   DRIVER=fake ./run-demo.sh      # dry run, wheels disconnected
# POSIX sh, not bash: on Raspberry Pi OS `sh run-demo.sh` runs under dash.
set -eu
cd "$(dirname "$0")"

: "${HUAWEI_API_KEY:?set HUAWEI_API_KEY first (and rotate it if it has ever been pasted anywhere)}"

# Values already in the environment win, so a dry run is just
# DRIVER=fake ./run-demo.sh. The catch is that an export left over from an
# earlier step wins in exactly the same silent way — a stale DRIVER=fake once
# cost a whole demo — so anything inherited gets named before we start.
inherited=""
setting() {
    eval "current=\${$1:-}"
    if [ -n "$current" ]; then
        inherited="$inherited    $1=$current
"
    else
        export "$1=$2"
    fi
}

# --- what is real -----------------------------------------------------------
setting MOCK false
setting CAMERA rpicam
setting CAMERA_ROTATION 180
setting DRIVER tb6612

# --- latency ----------------------------------------------------------------
# Image tokens go as the area: 640x480 is 300 of them, 320x240 is 80. The
# reason field is output tokens, and output is generated serially, so it costs
# wall clock on every single call. Set VISION_EXPLAIN=true while tuning.
setting VISION_WIDTH 320
setting VISION_HEIGHT 240
setting VISION_EXPLAIN false
# Local tracking carries the box between replies, so the car steers on a box
# tens of milliseconds old instead of seconds. That is what makes it react,
# and it costs ~2% of one core rather than a multiple of the API bill.
setting TRACK true
setting TRACK_HZ 20
# Flow cannot report that a target has gone, so a seed is only followed for
# this long before the model has to confirm the target is still there.
setting TRACK_MAX_AGE 1.5
# With tracking doing the reacting, one request in flight is enough: the model
# only has to confirm the target and correct drift. Raise for faster
# confirmation at proportionally higher spend.
# Two in flight: the model confirms roughly every 1.7s instead of 3.4s, which
# halves how long the car can chase something no longer there. Costs 2x.
setting VISION_CONCURRENCY 2
setting VISION_STAGGER 1.2

# --- how fast ---------------------------------------------------------------
# 3.4s round trip means the car acts on where things were 3.4s ago. Slow.
setting BASE_SPEED 0.2
setting TURN_GAIN 0.3
# A pivoting car keeps rotating after the command stops, so call it centred
# sooner than a wheeled robot would. Raise further if it still overshoots.
setting DEAD_ZONE 0.15
setting SEARCH_SPEED 0.25

# --- pacing and limits ------------------------------------------------------
setting CONTROL_INTERVAL 0      # poll as fast as latency allows
setting CONTROL_HZ 10
setting STALE_AFTER 5           # max seconds driving on one decision
setting MAX_RUN_SECONDS 120     # backstop if /stop is unreachable
setting MAX_COST_USD 0.25

if [ -n "$inherited" ]; then
    echo
    echo "Using these from your shell instead of the demo defaults:"
    printf '%s' "$inherited"
    echo "  (unset them for the standard live configuration)"
fi

exec .venv/bin/python server.py
