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
        left_rev: int = 22,
        left_en: int = 18,
        right_fwd: int = 23,
        right_rev: int = 24,
        right_en: int = 25,
    ):
        from gpiozero import Motor

        self._left = Motor(forward=left_fwd, backward=left_rev, enable=left_en)
        self._right = Motor(forward=right_fwd, backward=right_rev, enable=right_en)

    def apply(self, left: float, right: float) -> None:
        self._left.value = max(-1.0, min(1.0, left))
        self._right.value = max(-1.0, min(1.0, right))


def make_driver() -> Driver:
    provider = os.environ.get("DRIVER", "fake").lower()
    if provider == "l298n":
        return L298NDriver(
            left_fwd=int(os.environ.get("L298N_LEFT_FWD", "17")),
            left_rev=int(os.environ.get("L298N_LEFT_REV", "22")),
            left_en=int(os.environ.get("L298N_LEFT_EN", "18")),
            right_fwd=int(os.environ.get("L298N_RIGHT_FWD", "23")),
            right_rev=int(os.environ.get("L298N_RIGHT_REV", "24")),
            right_en=int(os.environ.get("L298N_RIGHT_EN", "25")),
        )
    return FakeDriver()