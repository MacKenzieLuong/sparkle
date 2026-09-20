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
        from gpiozero import DigitalOutputDevice, Motor

        self._stby = DigitalOutputDevice(stby, initial_value=True)
        # Matches the verified GPIO test: A forward is GPIO 27 (AIN2),
        # and B forward is GPIO 16 (BIN1).
        self._left = Motor(forward=ain2, backward=ain1, enable=pwma)
        self._right = Motor(forward=bin1, backward=bin2, enable=pwmb)

    def apply(self, left: float, right: float) -> None:
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
