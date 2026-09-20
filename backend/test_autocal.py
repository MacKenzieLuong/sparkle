"""The auto calibration reads motion from the image, so test it against
synthetic motion where the true answer is known."""
import time

import cv2
import numpy as np
import pytest

import calibrate

WIDTH, HEIGHT = 320, 240


class PanningCamera:
    """Scene that slides sideways a fixed number of pixels per read."""

    def __init__(self, pixels_per_read=0.0, converging=False):
        rng = np.random.default_rng(3)
        self.scene = rng.integers(40, 215, (HEIGHT, WIDTH * 4), dtype=np.uint8)
        self.offset = 0.0
        self.pixels_per_read = pixels_per_read
        self.converging = converging

    def read(self):
        start = int(self.offset) % (WIDTH * 3)
        frame = self.scene[:, start:start + WIDTH]
        if self.converging:
            # Near parts of the scene outrun far ones: translation, not a spin.
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame = cv2.warpAffine(
                frame,
                np.float32([[1.06, 0, -0.03 * WIDTH], [0, 1.06, -0.03 * HEIGHT]]),
                (WIDTH, HEIGHT),
            )
        self.offset += self.pixels_per_read
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def test_flow_reports_no_motion_when_still():
    result = calibrate.flow_sample(PanningCamera(0.0), pairs=3)
    assert result is not None
    rate, _ = result
    assert abs(rate) < 50, f"a stationary scene should read near zero, got {rate}"


def test_flow_detects_motion_and_its_direction():
    right = calibrate.flow_sample(PanningCamera(-6.0), pairs=3)
    left = calibrate.flow_sample(PanningCamera(6.0), pairs=3)
    assert right is not None and left is not None
    assert abs(right[0]) > 100, "panning should register clearly"
    assert np.sign(right[0]) != np.sign(left[0]), "direction should flip"


def test_faster_panning_reads_faster():
    slow = calibrate.flow_sample(PanningCamera(-3.0), pairs=4)
    fast = calibrate.flow_sample(PanningCamera(-9.0), pairs=4)
    assert abs(fast[0]) > abs(slow[0]) * 1.5


def test_consistency_separates_a_spin_from_a_swing():
    """A spin moves the whole scene together; a swing also translates."""
    spin = calibrate.flow_sample(PanningCamera(-6.0), pairs=4)
    swing = calibrate.flow_sample(PanningCamera(-6.0, converging=True), pairs=4)
    assert spin is not None and swing is not None
    assert spin[1] > swing[1], (
        f"rigid rotation should look more consistent: {spin[1]:.2f} vs {swing[1]:.2f}"
    )


def test_flow_gives_up_on_a_blank_scene():
    class Blank:
        def read(self):
            return np.full((HEIGHT, WIDTH, 3), 128, np.uint8)

    assert calibrate.flow_sample(Blank(), pairs=2) is None


class RecordingDriver:
    """Turns only above a threshold, like a loaded chassis."""

    def __init__(self, spins_from=0.45):
        self.spins_from = spins_from
        self.tried = []
        self.current = 0.0

    def apply(self, left, right):
        self.tried.append(abs(left))
        self.current = abs(left)

    def _drive(self, left, right):
        self.apply(left, right)

    def stop(self):
        self.current = 0.0


def test_auto_pivot_finds_the_throttle_that_actually_spins(monkeypatch):
    driver = RecordingDriver(spins_from=0.45)

    class Linked(PanningCamera):
        def read(self):
            self.pixels_per_read = -7.0 if driver.current >= driver.spins_from else 0.0
            return super().read()

    monkeypatch.setattr(calibrate.time, "sleep", lambda *_: None)
    result = calibrate.auto_pivot(driver, Linked(), {"left": 0.1, "right": 0.2})

    assert result["minimum_spin"] == pytest.approx(0.45, abs=0.06), (
        f"should discover the real threshold, got {result['minimum_spin']}"
    )
    assert max(driver.tried) >= 0.45, "should have swept upward until it turned"


def test_auto_pivot_reports_failure_rather_than_a_number(monkeypatch):
    """A car that never spins is a mechanical problem, not a tuning one."""
    driver = RecordingDriver(spins_from=99)
    monkeypatch.setattr(calibrate.time, "sleep", lambda *_: None)

    result = calibrate.auto_pivot(driver, PanningCamera(0.0), {})

    assert result["minimum_spin"] is None
    assert max(driver.tried) >= 0.95, "should have tried full throttle before giving up"
