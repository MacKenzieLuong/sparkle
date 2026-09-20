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
