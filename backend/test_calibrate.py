import builtins

import pytest

import calibrate


class StubDriver:
    def __init__(self):
        self.pulses = []
        self.stopped = 0

    def apply(self, left, right):
        self.pulses.append((left, right))

    def stop(self):
        self.stopped += 1
        self.pulses.append((0.0, 0.0))


@pytest.fixture
def instant(monkeypatch):
    """Skip the countdowns and settle delays so tests are not real-time."""
    monkeypatch.setattr(calibrate, "countdown", lambda *a, **k: True)
    monkeypatch.setattr(calibrate.time, "sleep", lambda *_: None)


def scripted(monkeypatch, answers):
    remaining = list(answers)
    monkeypatch.setattr(builtins, "input", lambda *_: str(remaining.pop(0)))


def test_ask_float_reads_a_measurement(monkeypatch):
    scripted(monkeypatch, ["12.5"])
    assert calibrate.ask_float("x") == 12.5


def test_ask_float_treats_blank_as_skip(monkeypatch):
    import math

    scripted(monkeypatch, [""])
    assert math.isnan(calibrate.ask_float("x"))


def test_ask_float_treats_q_as_abort(monkeypatch):
    scripted(monkeypatch, ["q"])
    assert calibrate.ask_float("x") is None


def test_ask_float_rejects_nonsense_then_accepts(monkeypatch):
    scripted(monkeypatch, ["banana", "31"])
    assert calibrate.ask_float("x") == 31.0


def test_pivots_solve_rate_and_coast(instant, monkeypatch):
    """1s -> 55 deg and 2s -> 95 deg means 40 deg/s with 15 deg of coast."""
    monkeypatch.setattr(calibrate, "PIVOT_DIFFERENTIALS", (0.3,))
    scripted(monkeypatch, ["55", "95"])
    driver = StubDriver()

    result = calibrate.measure_pivots(driver)

    assert result["0.3"]["deg_per_second"] == pytest.approx(40.0)
    assert result["0.3"]["coast_deg"] == pytest.approx(15.0)
    assert (0.3, -0.3) in driver.pulses, "should have pivoted, not driven straight"
    assert driver.stopped >= 2


def test_pivot_with_one_duration_reports_no_coast(instant, monkeypatch):
    monkeypatch.setattr(calibrate, "PIVOT_DIFFERENTIALS", (0.3,))
    monkeypatch.setattr(calibrate, "PIVOT_DURATIONS", (2.0,))
    scripted(monkeypatch, ["80"])

    result = calibrate.measure_pivots(StubDriver())

    assert result["0.3"]["deg_per_second"] == pytest.approx(40.0)
    assert result["0.3"]["coast_deg"] is None, "coast needs two durations"


def test_abort_stops_the_run(instant, monkeypatch):
    monkeypatch.setattr(calibrate, "PIVOT_DIFFERENTIALS", (0.25, 0.5))
    scripted(monkeypatch, ["q"])
    assert calibrate.measure_pivots(StubDriver()) == {}


def test_forward_speed(instant, monkeypatch):
    monkeypatch.setattr(calibrate, "FORWARD_THROTTLES", (0.2,))
    monkeypatch.setattr(calibrate, "FORWARD_SECONDS", 2.0)
    scripted(monkeypatch, ["1.2"])
    driver = StubDriver()

    assert calibrate.measure_forward(driver)["0.2"] == pytest.approx(0.6)
    assert (0.2, 0.2) in driver.pulses, "should have driven straight"


def test_report_survives_partial_data(capsys):
    calibrate.report({"pivots": {}, "scale": {}})
    assert "Not enough measurements" in capsys.readouterr().out


def test_report_derives_a_coast_aware_turn(capsys):
    calibrate.report({
        "pivots": {"0.3": {"deg_per_second": 40.0, "coast_deg": 15.0}},
        "scale": {"deg_per_pixel": 0.2, "frame_width": 320},
        "forward": {"0.2": 0.6},
    })
    out = capsys.readouterr().out
    assert "32 deg off centre" in out
    assert "0.42s" in out, "should subtract coast from the turn duration"
    assert "0.80s" in out
