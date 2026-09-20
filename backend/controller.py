from __future__ import annotations

import os
from dataclasses import dataclass
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


@dataclass
class DriveCommand:
    left: float
    right: float
    status: str
    note: str = ""
    area_fraction: float = 0.0


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def command(
    box_2d: Optional[Tuple[int, int, int, int]], turn_scale: float = 1.0
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

    turn = TURN_GAIN * dx * scale * turn_scale if abs(dx) > DEAD_ZONE else 0.0
    speed = BASE_SPEED * scale

    left = _clamp(speed + turn)
    right = _clamp(speed - turn)

    return DriveCommand(
        left, right, "moving", f"dx {dx:.2f}, area {area_fraction:.2f}", area_fraction
    )


def act(
    action: str,
    box_2d: Optional[Tuple[int, int, int, int]],
    turn_scale: float = 1.0,
) -> DriveCommand:
    """Turn the model's decision into throttles.

    The model chooses *what* to do; the speeds stay here, so no reply can make
    the car move faster than this machine was configured to allow. turn_scale
    fades rotation out as the decision ages — see turn_authority.
    """
    if action == "approach":
        return command(box_2d, turn_scale)
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