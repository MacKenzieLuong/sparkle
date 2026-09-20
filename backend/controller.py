from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


FRAME_CENTER = 500
MIN_SPEED_SCALE = 0.15

# Read once at import: the server sets these before it starts. How fast the car
# may travel while blind between model calls is a property of the drivetrain,
# so it has to be tunable per car rather than baked in here.
DEAD_ZONE = _env_float("DEAD_ZONE", 0.08)
ARRIVED_AREA_FRACTION = _env_float("ARRIVED_AREA_FRACTION", 0.5)
BASE_SPEED = _env_float("BASE_SPEED", 0.5)
TURN_GAIN = _env_float("TURN_GAIN", 0.8)
SEARCH_SPEED = _env_float("SEARCH_SPEED", 0.25)
BACK_OFF_SPEED = _env_float("BACK_OFF_SPEED", 0.2)
TURN_DECAY = _env_float("TURN_DECAY", 1.2)
# How much rotation one model decision may command, in throttle-seconds (the
# integral of the turn throttle). A fresh decision can spin closer to the full
# TURN_DECAY window; a boxless search does so blind and sweeps the target past
# the car. Budgeting rotation instead of wall time bounds the angle actually
# turned, per decision, whatever the physical rotation rate is.
ROTATION_BUDGET = _env_float("ROTATION_BUDGET", 0.5)
# The calibrated form of the same budget, in degrees of actual heading change.
# Only meaningful once `calibration_deg_per_turn_second` has a measurement; the
# throttle-seconds budget above is the fallback when it has not.
ROTATION_BUDGET_DEG = _env_float("ROTATION_BUDGET_DEG", 30.0)
# Seconds of lead in the steering error. A chassis that keeps rotating after
# the power is cut overshoots under any proportional gain: by the time the
# error reads zero the car is still turning, and no value of TURN_GAIN closes
# that gap. Steering on where the error is *heading* cancels it, and the lead
# that cancels it is the time the car takes to stop — which calibrate.py
# measures as the coast. 0 disables the term, leaving the plain proportional
# turn, so it changes nothing until it has a measurement behind it.
TURN_LEAD = _env_float("TURN_LEAD", 0.0)
# Time constant of the low-pass on the measured error rate. Differencing a
# tracked box is noisy enough that the raw rate cannot go on the wheels.
TURN_RATE_SMOOTHING = _env_float("TURN_RATE_SMOOTHING", 0.15)
# Beyond these the samples are not describing motion: a gap this long means the
# loop stalled, and dx (which spans -1 to 1) cannot really slew this fast.
RATE_MAX_GAP = 0.5
RATE_MAX = 10.0

CALIBRATION_DEFAULT_FILE = Path(__file__).resolve().parent / "calibration.json"


def calibration_deg_per_turn_second(path: Optional[str] = None) -> Optional[float]:
    """Heading rate per unit of turn throttle, from calibrate.py's measurements.

    `calibrate.py` reports degrees per second at several pivot differentials.
    At a pivot the turn throttle equals the differential, so each reading is a
    rate/throttle point; the lowest usable one is a single-slope conversion from
    commanded turn throttle to heading rate, and None means no such measurement
    exists. A body that coasts after the cut is not part of the conversion: the
    budget spends flight time, not settling time.
    """
    raw = os.environ.get("CALIBRATION_FILE")
    if not path:
        path = raw or str(CALIBRATION_DEFAULT_FILE)
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    pivots = data.get("pivots") or {}
    usable = {
        float(d): float(v["deg_per_second"])
        for d, v in pivots.items()
        if float(v.get("deg_per_second", 0.0)) > 0
    }
    if not usable:
        return None
    differential = min(usable)
    return usable[differential] / differential


def turn_authority(age: float) -> float:
    """How much of a turn to still apply for a decision this many seconds old.

    The car acts on the same frame until the next one arrives seconds later.
    Holding the turn for all of it keeps rotating long after the car is already
    pointing at the target, which is what makes it weave. Authority falls to
    zero so it turns, then coasts straight until it can see again.
    """
    if TURN_DECAY <= 0:
        return 1.0
    return max(0.0, 1.0 - age / TURN_DECAY)


class TurnDamper:
    """Rate of change of the steering error, smoothed enough to steer on.

    The tracker republishes the box at TRACK_HZ, so the derivative the lead
    term needs is already being measured — it only has to be differenced and
    filtered. Owned by the control thread, since it is a running estimate and
    not a pure function: feeding it from anywhere else corrupts the rate.

    Two things are discontinuities rather than motion, and both report zero
    instead of an enormous velocity: a long gap between samples, and a jump too
    large to be real — which is what a re-seed onto a newly arrived model box
    looks like, the box having moved a second's worth in one tick. A caller
    that knows the box changed source should `reset()` rather than rely on
    the jump test catching it.
    """

    def __init__(self, smoothing: float = TURN_RATE_SMOOTHING) -> None:
        self._smoothing = max(0.0, smoothing)
        self.reset()

    def reset(self) -> None:
        self._dx: Optional[float] = None
        self._at: Optional[float] = None
        self._rate = 0.0

    def update(self, dx: Optional[float], now: float) -> float:
        """Feed the current error; get the filtered rate, in dx per second.

        Safe to call every control tick, including ticks whose box has not
        changed. Those difference to a rate of zero, but the next tick that
        does move covers the whole gap in a correspondingly shorter dt, so the
        estimate stays centred on the true rate rather than being dragged
        toward zero. Measured at a 5:1 tick-to-box ratio it tracked a known
        -0.20 dx/s at -0.203, so no separate sample-and-hold is warranted.
        """
        if dx is None:
            self.reset()
            return 0.0
        previous, previous_at = self._dx, self._at
        self._dx, self._at = dx, now
        if previous is None or previous_at is None:
            return 0.0
        dt = now - previous_at
        if dt <= 0 or dt > RATE_MAX_GAP:
            self._rate = 0.0
            return 0.0
        raw = (dx - previous) / dt
        if abs(raw) > RATE_MAX:
            self._rate = 0.0
            return 0.0
        alpha = 1.0 if self._smoothing <= 0 else dt / (self._smoothing + dt)
        self._rate += alpha * (raw - self._rate)
        return self._rate


@dataclass
class DriveCommand:
    left: float
    right: float
    status: str
    note: str = ""
    area_fraction: float = 0.0


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def dx_of(box_2d: Optional[Tuple[int, int, int, int]]) -> Optional[float]:
    """The steering error a box implies: offset from centre, -1 to 1.

    Shared with `command` so the damper differences exactly the quantity the
    steering acts on, rather than a second copy of the formula that could drift
    away from it.
    """
    if box_2d is None:
        return None
    _, xmin, _, xmax = box_2d
    return ((xmin + xmax) / 2.0 - FRAME_CENTER) / float(FRAME_CENTER)


def steer_components(cmd: DriveCommand) -> Tuple[float, float]:
    """The forward speed and turn throttle a command is made of.

    This chassis is skid-steered, so heading rate is set by the difference of
    the two wheel throttles while forward speed is their mean.
    """
    return (cmd.left + cmd.right) / 2.0, (cmd.left - cmd.right) / 2.0


def apply_turn(cmd: DriveCommand, turn: float) -> DriveCommand:
    """Rebuild a command around a new turn throttle, keeping forward speed.

    `status`, `note` and `area_fraction` carry over untouched.
    """
    forward, _ = steer_components(cmd)
    return DriveCommand(
        _clamp(forward + turn),
        _clamp(forward - turn),
        cmd.status,
        cmd.note,
        cmd.area_fraction,
    )


def enforce_turn_budget(
    cmd: DriveCommand, remaining: float, dt: float
) -> Tuple[DriveCommand, float]:
    """Clamp a command's turn to what the rotation budget has left.

    Returns the bounded command and the magnitude of turn throttle it applied,
    so the caller can spend the budget. With nothing left the command becomes a
    straight drive at the same forward speed: the car coasts straight rather
    than rotating past the target on a stale decision.
    """
    if remaining <= 0 or dt <= 0:
        return apply_turn(cmd, 0.0), 0.0
    _, turn = steer_components(cmd)
    room = remaining / dt
    bounded = max(-room, min(room, turn))
    return apply_turn(cmd, bounded), abs(bounded)


def enforce_turn_budget_deg(
    cmd: DriveCommand,
    remaining_deg: float,
    dt: float,
    deg_per_turn_second: Optional[float],
) -> Tuple[DriveCommand, float]:
    """Like `enforce_turn_budget`, but the budget is a heading angle.

    `deg_per_turn_second` is a calibrated slope of heading change per unit of
    turn throttle; the spent budget is reported in degrees so the caller
    accumulates real rotation, not raw throttle-time. Uncalibrated (None) or
    exhausted budgets coast straight rather than inventing a rotation rate.
    """
    if remaining_deg <= 0 or dt <= 0 or not deg_per_turn_second:
        return apply_turn(cmd, 0.0), 0.0
    _, turn = steer_components(cmd)
    room = remaining_deg / dt / deg_per_turn_second
    bounded = max(-room, min(room, turn))
    spent_deg = abs(bounded) * dt * deg_per_turn_second
    return apply_turn(cmd, bounded), spent_deg


def command(
    box_2d: Optional[Tuple[int, int, int, int]],
    turn_scale: float = 1.0,
    turn_rate: float = 0.0,
) -> DriveCommand:
    if box_2d is None:
        return DriveCommand(0.0, 0.0, "target_lost", "no detection")

    ymin, xmin, ymax, xmax = box_2d
    center_x = (xmin + xmax) / 2.0
    box_width = xmax - xmin
    box_height = ymax - ymin
    area_fraction = (box_width * box_height) / 1_000_000.0

    if area_fraction >= ARRIVED_AREA_FRACTION:
        return DriveCommand(
            0.0, 0.0, "arrived", f"area {area_fraction:.2f}", area_fraction
        )

    dx = (center_x - FRAME_CENTER) / float(FRAME_CENTER)
    scale = max(MIN_SPEED_SCALE, 1.0 - area_fraction / ARRIVED_AREA_FRACTION)

    # Steer on where the error will be TURN_LEAD seconds from now rather than
    # where it is, so the turn is already easing off while the car still has
    # momentum to shed. The dead zone stays on dx itself: inside it the car is
    # pointed close enough, and letting a noisy rate start a turn there only
    # makes it twitch about the centre.
    lead = TURN_LEAD * turn_rate
    turn = TURN_GAIN * (dx + lead) * scale * turn_scale if abs(dx) > DEAD_ZONE else 0.0
    speed = BASE_SPEED * scale

    left = _clamp(speed + turn)
    right = _clamp(speed - turn)

    note = f"dx {dx:.2f}, area {area_fraction:.2f}"
    if lead:
        note += f", lead {lead:+.2f}"
    return DriveCommand(left, right, "moving", note, area_fraction)


def act(
    action: str,
    box_2d: Optional[Tuple[int, int, int, int]],
    turn_scale: float = 1.0,
    turn_rate: float = 0.0,
) -> DriveCommand:
    """Turn the model's decision into throttles.

    The model chooses *what* to do; the speeds stay here, so no reply can make
    the car move faster than this machine was configured to allow. turn_scale
    fades rotation out as the decision ages — see turn_authority. turn_rate is
    how fast the error is closing, which damps the turn — see TurnDamper.
    """
    if action == "approach":
        return command(box_2d, turn_scale, turn_rate)
    if action in ("search_left", "search_right"):
        # A scan overshoots for the same reason a turn does: spinning for a
        # whole cycle sweeps the target straight back out of frame.
        speed = SEARCH_SPEED * turn_scale
        if action == "search_left":
            return DriveCommand(-speed, speed, "searching", "scanning left")
        return DriveCommand(speed, -speed, "searching", "scanning right")
    if action == "back_off":
        return DriveCommand(
            -BACK_OFF_SPEED, -BACK_OFF_SPEED, "backing_off", "clearing space"
        )
    return DriveCommand(0.0, 0.0, "halted", f"model said {action}")