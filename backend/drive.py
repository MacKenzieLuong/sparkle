from __future__ import annotations

import math
import os
import threading
import time


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def condition(value: float, scale: float, minimum: float) -> float:
    """Turn a wanted throttle into one this motor will actually deliver.

    Two corrections, both measured per side because the chassis is not
    symmetric. `scale` trims out a load difference that would otherwise curve a
    commanded-straight drive, and `minimum` lifts small commands over the
    throttle at which the wheel starts turning at all — below it a loaded motor
    just sits there humming while the other side drives.
    """
    if value == 0.0:
        return 0.0
    magnitude = min(1.0, abs(value) * scale)
    if magnitude <= 0.0:
        return 0.0
    magnitude = minimum + (1.0 - minimum) * magnitude
    return math.copysign(min(1.0, magnitude), value)


class Driver:
    """Conditioning shared by every motor backend.

    Beyond the per-side scale and minimum, this breaks static friction. A
    stalled motor needs more torque to start than to keep turning, so a small
    steady command that is perfectly able to sustain rotation cannot begin it:
    the motor sits and buzzes. MOTOR_KICK is a brief higher throttle applied
    only when a wheel starts from rest or reverses, after which the requested
    value takes over and the wheel keeps moving at it.

    That is also what makes a slow pivot possible at all. Without it the
    slowest usable throttle is the breakaway one, so "turn more gently" has a
    floor; with it, the floor is whatever sustains motion once started, which
    is lower.
    """

    def __init__(self):
        self.left_scale = _env_float("MOTOR_LEFT_SCALE", 1.0)
        self.right_scale = _env_float("MOTOR_RIGHT_SCALE", 1.0)
        self.left_min = _env_float("MOTOR_LEFT_MIN", 0.0)
        self.right_min = _env_float("MOTOR_RIGHT_MIN", 0.0)
        # 0 disables the kick entirely, so behaviour is unchanged until it is
        # measured. Loaded breakaway is higher than what calibrate.py reports,
        # because that test lifts the wheel clear of the floor.
        self.kick = _env_float("MOTOR_KICK", 0.0)
        self.kick_seconds = _env_float("MOTOR_KICK_SECONDS", 0.15)
        self._previous = {"left": 0.0, "right": 0.0}
        self._kick_started: dict[str, float | None] = {"left": None, "right": None}

    def _now(self) -> float:
        """Overridable so a test can drive the kick window without sleeping."""
        return time.monotonic()

    def _kickstart(self, side: str, value: float, now: float) -> float:
        """Raise `value` to the kick throttle while a wheel is getting going."""
        previous = self._previous[side]
        self._previous[side] = value
        if self.kick <= 0.0 or value == 0.0:
            self._kick_started[side] = None
            return value
        # From rest, or through a reversal: in both cases the wheel is
        # stationary at the moment the new command arrives.
        if previous == 0.0 or (previous < 0.0) != (value < 0.0):
            self._kick_started[side] = now
        started = self._kick_started[side]
        if started is not None and now - started < self.kick_seconds:
            return math.copysign(min(1.0, max(abs(value), self.kick)), value)
        return value

    def apply(self, left: float, right: float) -> None:
        now = self._now()
        self._drive(
            self._kickstart("left", condition(left, self.left_scale, self.left_min), now),
            self._kickstart("right", condition(right, self.right_scale, self.right_min), now),
        )

    def _drive(self, left: float, right: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        # Straight to the motors: a stop must never be lifted by a minimum.
        # The wheels are about to be stationary, so the next command starts
        # from rest and has to kick again.
        self._previous = {"left": 0.0, "right": 0.0}
        self._kick_started = {"left": None, "right": None}
        self._drive(0.0, 0.0)


class FakeDriver(Driver):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self.last = (0.0, 0.0)
        self.commands = []

    def _drive(self, left: float, right: float) -> None:
        with self._lock:
            # The control thread re-applies at a fixed rate; only log changes.
            changed = (left, right) != self.last
            self.last = (left, right)
            self.commands.append((left, right))
        if changed:
            print(f"[FakeDriver] left={left:+.2f} right={right:+.2f}")


class TB6612Driver(Driver):
    def __init__(
        self,
        ain1: int = 17,
        ain2: int = 27,
        pwma: int = 13,
        bin1: int = 16,
        bin2: int = 20,
        pwmb: int = 12,
        stby: int = 21,
    ):
        super().__init__()
        from gpiozero import DigitalOutputDevice, Motor

        self.pins = {
            "left_forward": ain2, "left_reverse": ain1, "left_pwm": pwma,
            "right_forward": bin1, "right_reverse": bin2, "right_pwm": pwmb,
            "stby": stby,
        }
        self._stby = DigitalOutputDevice(stby, initial_value=True)
        # Matches the verified GPIO test: A forward is GPIO 27 (AIN2),
        # and B forward is GPIO 16 (BIN1).
        self._left = Motor(forward=ain2, backward=ain1, enable=pwma)
        self._right = Motor(forward=bin1, backward=bin2, enable=pwmb)

    def _drive(self, left: float, right: float) -> None:
        self._stby.on()
        self._left.value = max(-1.0, min(1.0, left))
        self._right.value = max(-1.0, min(1.0, right))

    def stop(self) -> None:
        super().stop()
        self._stby.off()


def make_driver() -> Driver:
    provider = os.environ.get("DRIVER", "fake").lower()
    if provider in ("tb6612", "l298n"):
        return TB6612Driver(
            ain1=int(os.environ.get("TB6612_AIN1", "17")),
            ain2=int(os.environ.get("TB6612_AIN2", "27")),
            pwma=int(os.environ.get("TB6612_PWMA", "13")),
            bin1=int(os.environ.get("TB6612_BIN1", "16")),
            bin2=int(os.environ.get("TB6612_BIN2", "20")),
            pwmb=int(os.environ.get("TB6612_PWMB", "12")),
            stby=int(os.environ.get("TB6612_STBY", "21")),
        )
    return FakeDriver()
