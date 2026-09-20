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
# Minimum gap between successive inference frames, across all the workers, so
# they sample the whole cycle instead of firing together. The ideal is
# cycle / VISION_CONCURRENCY (~1.7s at two workers), but it is a floor rather
# than a target: a worker already running late takes its frame immediately, so
# setting it too high costs nothing but setting it too low lets frames bunch.
setting VISION_SPACING 1.5

# --- how fast ---------------------------------------------------------------
# 3.4s round trip means the car acts on where things were 3.4s ago. Slow.
# The stiction minimum is most of what reaches the wheel, so this is not
# proportional to speed: 0.075 puts the left wheel at 0.26 against 0.35 at
# 0.2. Matches .env.example so both launchers drive the same.
setting BASE_SPEED 0.075
setting TURN_GAIN 0.35
# Seconds of lead in the steering error, cancelling the rotation the car keeps
# after the power is cut. 0 is the plain proportional turn, which overshoots on
# a chassis with any momentum; set it from the coast calibrate.py measures.
setting TURN_LEAD 0
# A pivoting car keeps rotating after the command stops, so call it centred
# sooner than a wheeled robot would. Raise further if it still overshoots.
setting DEAD_ZONE 0.15
# Stop when the target's box covers this share of the frame -- about half
# the frame width for a square target. Well short of touching it, because
# approach speed barely falls as the target grows (the stiction minimum is
# most of the throttle) so the car arrives fast, stops dead, then coasts.
setting ARRIVED_AREA_FRACTION 0.25
# The in-place scan pivot. Floored by the stiction minimums below: as this
# approaches 0 the wheels still get MOTOR_*_MIN, because under that they do
# not turn at all. 0.05 is about as slow as this chassis pivots.
setting SEARCH_SPEED 0.05
# Rotation one decision may command before the car coasts straight. Raised
# alongside TURN_GAIN, or the larger turn is simply clipped and the gain
# changes nothing. DEG applies once calibration.json has usable pivots;
# throttle-seconds is the fallback until then.
setting ROTATION_BUDGET_DEG 30
setting ROTATION_BUDGET 0.7

# --- chassis trim (from calibrate.py) ---------------------------------------
# The two sides are not loaded equally, so equal throttle does not drive
# straight, and the heavier side needs more PWM before it moves at all.
setting MOTOR_LEFT_SCALE 0.95
setting MOTOR_RIGHT_SCALE 1.0
setting MOTOR_LEFT_MIN 0.20
setting MOTOR_RIGHT_MIN 0.15
# Brief higher throttle when a wheel starts from rest or reverses. Below
# breakaway a motor only buzzes, so without this a small correction cannot
# start the wheel at all. 0 disables it.
setting MOTOR_KICK 0.45
setting MOTOR_KICK_SECONDS 0.15

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
