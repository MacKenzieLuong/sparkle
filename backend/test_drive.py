import os

import pytest

from drive import FakeDriver, condition


def test_condition_is_identity_by_default():
    for value in (-1.0, -0.3, 0.0, 0.2, 1.0):
        assert condition(value, 1.0, 0.0) == pytest.approx(value)


def test_zero_is_never_lifted_by_a_minimum():
    """A stop must reach the motors as a stop, not as the breakaway throttle."""
    assert condition(0.0, 1.0, 0.4) == 0.0


def test_scale_trims_a_side_down():
    assert condition(0.5, 0.8, 0.0) == pytest.approx(0.4)
    assert condition(-0.5, 0.8, 0.0) == pytest.approx(-0.4)


def test_minimum_lifts_small_commands_over_stiction():
    # 0.25 minimum: a 0.2 command becomes 0.25 + 0.75*0.2 = 0.40
    assert condition(0.2, 1.0, 0.25) == pytest.approx(0.40)
    assert condition(-0.2, 1.0, 0.25) == pytest.approx(-0.40)


def test_minimum_keeps_full_throttle_reachable():
    assert condition(1.0, 1.0, 0.25) == pytest.approx(1.0)


def test_output_stays_in_range():
    assert condition(2.0, 3.0, 0.5) == pytest.approx(1.0)
    assert condition(-2.0, 3.0, 0.5) == pytest.approx(-1.0)


def test_driver_applies_per_side_trim(monkeypatch):
    monkeypatch.setenv("MOTOR_LEFT_SCALE", "0.8")
    monkeypatch.setenv("MOTOR_RIGHT_SCALE", "1.0")
    driver = FakeDriver()

    driver.apply(0.5, 0.5)

    left, right = driver.last
    assert left == pytest.approx(0.4), "heavier right side should keep more throttle"
    assert right == pytest.approx(0.5)


def test_driver_stop_bypasses_the_minimum(monkeypatch):
    monkeypatch.setenv("MOTOR_LEFT_MIN", "0.3")
    monkeypatch.setenv("MOTOR_RIGHT_MIN", "0.3")
    driver = FakeDriver()

    driver.apply(0.4, 0.4)
    assert driver.last[0] > 0.4, "a small command should be lifted"

    driver.stop()
    assert driver.last == (0.0, 0.0), "stop must be an actual stop"


def _driver_with(**env):
    """A FakeDriver built with these settings.

    The knobs are read in Driver.__init__, so setting the environment and
    constructing is enough — no module reload, which would swap the classes
    other tests already hold.
    """
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        return FakeDriver()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _at(driver, clock):
    """Drive the kick window from a fake clock, never the real one."""
    driver._now = lambda: clock[0]
    return driver


def test_kick_is_off_by_default():
    driver = _driver_with(MOTOR_KICK="0", MOTOR_LEFT_MIN="0.2")
    driver.apply(0.05, 0.05)
    assert abs(driver.last[0]) < 0.3, "no kick unless MOTOR_KICK is set"


def test_kick_breaks_friction_then_settles():
    """A stalled motor needs more torque to start than to keep turning."""
    clock = [100.0]
    driver = _at(_driver_with(
        MOTOR_KICK="0.5", MOTOR_KICK_SECONDS="0.15", MOTOR_LEFT_MIN="0.2"), clock)
    driver.apply(0.05, 0.05)
    assert abs(driver.last[0]) == pytest.approx(0.5), "starts with the kick"
    clock[0] += 0.05
    driver.apply(0.05, 0.05)
    assert abs(driver.last[0]) == pytest.approx(0.5), "still kicking inside the window"
    clock[0] += 0.2
    driver.apply(0.05, 0.05)
    assert abs(driver.last[0]) < 0.3, "settles to the requested throttle"


def test_kick_repeats_after_a_stop_and_on_reversal():
    clock = [100.0]
    driver = _at(_driver_with(
        MOTOR_KICK="0.5", MOTOR_KICK_SECONDS="0.15", MOTOR_LEFT_MIN="0.2"), clock)
    driver.apply(0.05, 0.05)
    clock[0] += 1.0
    driver.apply(0.05, 0.05)
    assert abs(driver.last[0]) < 0.3
    driver.stop()
    driver.apply(0.05, 0.05)
    assert abs(driver.last[0]) == pytest.approx(0.5), "a stop means starting from rest"
    clock[0] += 1.0
    driver.apply(0.05, 0.05)
    assert abs(driver.last[0]) < 0.3
    driver.apply(-0.05, -0.05)
    assert abs(driver.last[0]) == pytest.approx(0.5), "a reversal is also from rest"


def test_kick_never_lifts_a_stop():
    driver = _driver_with(MOTOR_KICK="0.5", MOTOR_LEFT_MIN="0.2")
    driver.apply(0.05, 0.05)
    driver.stop()
    assert driver.last == (0.0, 0.0)
