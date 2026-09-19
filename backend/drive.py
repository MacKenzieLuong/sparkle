from __future__ import annotations

import os
import threading


class Driver:
    def apply(self, left: float, right: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self.apply(0.0, 0.0)


class FakeDriver(Driver):
    def __init__(self):
        self._lock = threading.Lock()
        self.last = (0.0, 0.0)
        self.commands = []

    def apply(self, left: float, right: float) -> None:
        with self._lock:
            self.last = (left, right)
            self.commands.append((left, right))
        print(f"[FakeDriver] left={left:+.2f} right={right:+.2f}")


class L298NDriver(Driver):
    def __init__(
        self,
        left_fwd: int = 17,
        left_rev: int = 27,
        left_en: int = 13,
        right_fwd: int = 16,
        right_rev: int = 20,
        right_en: int = 12,
        stby: int = 21,
    ):
        from gpiozero import DigitalOutputDevice, Motor

        self._stby = DigitalOutputDevice(stby, initial_value=True)
        self._left = Motor(forward=left_fwd, backward=left_rev, enable=left_en)
        self._right = Motor(forward=right_fwd, backward=right_rev, enable=right_en)

    def apply(self, left: float, right: float) -> None:
        self._stby.on()
        self._left.value = max(-1.0, min(1.0, left))
        self._right.value = max(-1.0, min(1.0, right))

    def stop(self) -> None:
        super().stop()
        self._stby.off()


def make_driver() -> Driver:
    provider = os.environ.get("DRIVER", "fake").lower()
    if provider == "l298n":
        return L298NDriver(
            left_fwd=int(os.environ.get("L298N_LEFT_FWD", "17")),
            left_rev=int(os.environ.get("L298N_LEFT_REV", "27")),
            left_en=int(os.environ.get("L298N_LEFT_EN", "13")),
            right_fwd=int(os.environ.get("L298N_RIGHT_FWD", "16")),
            right_rev=int(os.environ.get("L298N_RIGHT_REV", "20")),
            right_en=int(os.environ.get("L298N_RIGHT_EN", "12")),
            stby=int(os.environ.get("L298N_STBY", "21")),
        )
    return FakeDriver()