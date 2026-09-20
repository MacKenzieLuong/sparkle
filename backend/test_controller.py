import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from controller import (
    TurnDamper,
    act,
    calibration_deg_per_turn_second,
    command,
    dx_of,
    enforce_turn_budget,
    enforce_turn_budget_deg,
    steer_components,
)


def _throttle_with(**env) -> float:
    """Left throttle for a centred box, from a fresh interpreter.

    The knobs are read at import, so a subprocess is the honest way to prove
    the environment actually reaches the steering math.
    """
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import controller;"
            "print(f'{controller.command((350, 400, 650, 600)).left:.6f}')",
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return float(result.stdout.strip())


def test_center_target_drives_straight():
    cmd = command((350, 400, 650, 600))
    assert cmd.status == "moving"
    assert cmd.left == pytest.approx(cmd.right, abs=1e-6)
    assert cmd.left > 0


def test_left_target_turns_left():
    cmd = command((300, 50, 700, 250))
    assert cmd.status == "moving"
    assert cmd.left < cmd.right


def test_right_target_turns_right():
    cmd = command((300, 750, 700, 950))
    assert cmd.status == "moving"
    assert cmd.left > cmd.right


def test_slight_right_turns_right_gently():
    hard = command((300, 750, 700, 950))
    slight = command((300, 580, 700, 720))
    assert slight.status == "moving"
    assert slight.left > slight.right
    assert (slight.left - slight.right) < (hard.left - hard.right)


def test_no_detection_stops():
    cmd = command(None)
    assert cmd.status == "target_lost"
    assert cmd.left == 0.0
    assert cmd.right == 0.0


def test_huge_target_arrives():
    cmd = command((120, 120, 880, 880))
    assert cmd.status == "arrived"
    assert cmd.left == 0.0
    assert cmd.right == 0.0


def test_outputs_clamped():
    cmd = command((0, 0, 1000, 1000))
    assert -1.0 <= cmd.left <= 1.0
    assert -1.0 <= cmd.right <= 1.0


def test_extreme_offscreen_right_turns_hard():
    cmd = command((300, 995, 700, 1000))
    assert cmd.status == "moving"
    assert cmd.left > 0.5
    assert cmd.right >= -1.0
    assert cmd.right <= 1.0


def test_bigger_target_slows_down():
    far = command((420, 300, 580, 700))
    near = command((300, 300, 700, 700))
    assert far.status == "moving"
    assert near.status == "moving"
    assert near.left < far.left
    assert near.right < far.right


def test_base_speed_is_configurable():
    # A centred box: no turn, so throttle is BASE_SPEED scaled by 0.88 for area.
    assert _throttle_with(BASE_SPEED="0.5") == pytest.approx(0.44, abs=1e-4)
    assert _throttle_with(BASE_SPEED="0.2") == pytest.approx(0.176, abs=1e-4)


def test_turn_gain_is_configurable():
    hard = subprocess.run(
        [
            sys.executable, "-c",
            "import controller;"
            "c = controller.command((300, 750, 700, 950));"
            "print(f'{c.left - c.right:.6f}')",
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env={**os.environ, "TURN_GAIN": "0.2"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert hard.returncode == 0, hard.stderr
    gentle = float(hard.stdout.strip())
    reference = command((300, 750, 700, 950))
    assert 0 < gentle < (reference.left - reference.right)


def test_turn_authority_fades_with_age():
    from controller import TURN_DECAY, turn_authority

    assert turn_authority(0.0) == 1.0
    assert turn_authority(TURN_DECAY / 2) == pytest.approx(0.5)
    assert turn_authority(TURN_DECAY) == 0.0
    assert turn_authority(TURN_DECAY * 10) == 0.0, "never negative"


def test_stale_decision_stops_turning_but_keeps_driving():
    """The overshoot fix: hold the turn briefly, then coast straight."""
    off_centre = (300, 750, 700, 950)
    fresh = command(off_centre, turn_scale=1.0)
    faded = command(off_centre, turn_scale=0.0)

    assert fresh.left != fresh.right, "a fresh decision still turns"
    assert faded.left == pytest.approx(faded.right), "a faded one drives straight"
    assert faded.left > 0, "and keeps moving forward"


def _hard_turn_cmd():
    # dx 0.7, area 0.08 — a strong turn well clear of the arrival threshold.
    return command((300, 750, 700, 950))


def test_turn_budget_exhausted_coasts_straight_at_forward_speed():
    cmd, spent = enforce_turn_budget(_hard_turn_cmd(), remaining=0.0, dt=0.1)
    assert spent == 0.0
    assert cmd.left == pytest.approx(cmd.right)
    assert cmd.left > 0, "crossing the zero line is a stop, not a straight coast"


def test_turn_budget_holds_back_part_of_a_turn():
    original = _hard_turn_cmd()
    cmd, spent = enforce_turn_budget(original, remaining=0.02, dt=0.1)
    _, turn = steer_components(cmd)
    assert turn == pytest.approx(0.2), "only the remaining 0.2 of turn is allowed"
    assert spent == pytest.approx(0.2)
    forward, _ = steer_components(cmd)
    assert forward == pytest.approx(
        (original.left + original.right) / 2
    ), "the budget re-centres the command without slowing it"


def test_turn_budget_full_turn_passes_when_roomy():
    original = _hard_turn_cmd()
    cmd, spent = enforce_turn_budget(original, remaining=10.0, dt=0.1)
    assert spent == pytest.approx(abs(original.left - original.right) / 2)
    assert cmd.left == pytest.approx(original.left)
    assert cmd.right == pytest.approx(original.right)


def test_rotation_budget_is_configurable():
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import controller; print(controller.ROTATION_BUDGET)",
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env={**os.environ, "ROTATION_BUDGET": "0.2"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert float(result.stdout.strip()) == pytest.approx(0.2)


def test_rotation_budget_default_is_nonzero():
    import controller

    assert controller.ROTATION_BUDGET > 0, "a zero budget would never turn"


def test_calibration_loader_reads_lowest_usable_pivot(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "pivots": {
            "0.2": {"deg_per_second": 20.0},
            "0.25": {"deg_per_second": 0.0},  # below stiction — never usable
            "0.5": {"deg_per_second": 55.0},
        }
    }))
    assert calibration_deg_per_turn_second(str(path)) == pytest.approx(20.0 / 0.2)


def test_calibration_loader_none_when_unusable(tmp_path):
    missing = tmp_path / "nope.json"
    assert calibration_deg_per_turn_second(str(missing)) is None

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert calibration_deg_per_turn_second(str(bad)) is None

    no_pivots = tmp_path / "empty.json"
    no_pivots.write_text(json.dumps({"spin": {"minimum_spin": 0.3}}))
    assert calibration_deg_per_turn_second(str(no_pivots)) is None


def test_turn_budget_degrees_spends_degrees():
    original = _hard_turn_cmd()
    cmd, spent = enforce_turn_budget_deg(
        original, remaining_deg=10.0, dt=0.1, deg_per_turn_second=100.0
    )
    roomy = abs(original.left - original.right) / 2 * 0.1 * 100.0
    assert spent == pytest.approx(roomy), "spent is reported in degrees"
    assert cmd.left == pytest.approx(original.left), "a roomy budget changes nothing"


def test_turn_budget_degrees_holds_back_part_of_a_turn():
    cmd, spent = enforce_turn_budget_deg(
        _hard_turn_cmd(), remaining_deg=2.0, dt=0.1, deg_per_turn_second=100.0
    )
    _, turn = steer_components(cmd)
    assert turn == pytest.approx(0.2), "only 2 deg / (dt * k) of turn is allowed"
    assert spent == pytest.approx(2.0)


def test_turn_budget_degrees_exhausted_coasts_straight():
    cmd, spent = enforce_turn_budget_deg(
        _hard_turn_cmd(), remaining_deg=0.0, dt=0.1, deg_per_turn_second=100.0
    )
    assert spent == 0.0
    assert cmd.left == pytest.approx(cmd.right)
    assert cmd.left > 0, "exhaustion is a straight coast, not a stop"


def test_turn_budget_degrees_uncalibrated_coasts():
    cmd, spent = enforce_turn_budget_deg(
        _hard_turn_cmd(), remaining_deg=10.0, dt=0.1, deg_per_turn_second=None
    )
    assert spent == 0.0
    assert cmd.left == pytest.approx(cmd.right), (
        "no rotation rate -> never invent a degree spend"
    )


def test_degree_budget_default_is_nonzero():
    import controller

    assert controller.ROTATION_BUDGET_DEG > 0


def test_search_scan_also_fades():
    assert act("search_left", None, 0.0).left == 0.0
    partial = act("search_left", None, 0.5)
    full = act("search_left", None, 1.0)
    assert abs(partial.right) == pytest.approx(abs(full.right) / 2)


def test_search_actions_rotate_in_place():
    left = act("search_left", None)
    right = act("search_right", None)
    assert left.status == "searching"
    assert left.left < 0 < left.right, "left search spins counter-clockwise"
    assert right.right < 0 < right.left
    assert left.left == pytest.approx(-right.left)


def test_back_off_reverses_both_wheels():
    cmd = act("back_off", None)
    assert cmd.left < 0 and cmd.right < 0
    assert cmd.status == "backing_off"


def test_approach_uses_the_steering_math():
    box = (300, 750, 700, 950)
    assert act("approach", box) == command(box)


def test_unknown_action_halts():
    for action in ("launch_missiles", "", "APPROACH"):
        cmd = act(action, (300, 300, 700, 700))
        assert (cmd.left, cmd.right) == (0.0, 0.0), action
        assert cmd.status == "halted"


def _turn_throttle(box, turn_rate: float, **env) -> float:
    """The turn half of a command, from a fresh interpreter.

    TURN_LEAD is read at import, so the environment only reaches the steering
    math in a subprocess — same reason as `_throttle_with`.
    """
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import sys, controller;"
            "box = tuple(int(v) for v in sys.argv[1].split(','));"
            "c = controller.command(box, turn_rate=float(sys.argv[2]));"
            "print(f'{(c.left - c.right) / 2:.6f}')",
            ",".join(str(v) for v in box), str(turn_rate),
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return float(result.stdout.strip())


HARD_RIGHT = (300, 750, 700, 950)
CENTRED = (350, 400, 650, 600)


def test_dx_of_matches_the_steering_math():
    assert dx_of(None) is None
    assert dx_of((300, 400, 700, 600)) == pytest.approx(0.0)
    assert dx_of((300, 750, 700, 950)) == pytest.approx(0.7)


def test_turn_lead_is_off_by_default():
    import controller

    assert controller.TURN_LEAD == 0.0, "the lead term must be opt-in"


def test_turn_rate_does_nothing_without_lead():
    box = (300, 750, 700, 950)
    assert command(box, turn_rate=5.0) == command(box, turn_rate=0.0)


def test_lead_eases_a_closing_turn_and_hardens_an_opening_one():
    """The overshoot fix: steer on where the error is heading."""
    env = {"TURN_LEAD": "0.3"}
    static = _turn_throttle(HARD_RIGHT, 0.0, **env)
    closing = _turn_throttle(HARD_RIGHT, -1.0, **env)  # swinging back to centre
    opening = _turn_throttle(HARD_RIGHT, 1.0, **env)  # still running away
    assert closing < static < opening
    assert static > 0


def test_lead_does_not_wake_the_dead_zone():
    """A centred box stays straight however fast the noise says it is moving."""
    assert _turn_throttle(CENTRED, 5.0, TURN_LEAD="0.3") == pytest.approx(0.0)


def test_damper_has_no_rate_from_one_sample():
    damper = TurnDamper(smoothing=0.0)
    assert damper.update(0.5, 0.0) == 0.0


def test_damper_measures_a_steady_rate():
    damper = TurnDamper(smoothing=0.0)
    damper.update(0.0, 0.0)
    assert damper.update(0.1, 0.1) == pytest.approx(1.0)


def test_damper_filters_toward_the_raw_rate():
    damper = TurnDamper(smoothing=0.1)
    damper.update(0.0, 0.0)
    first = damper.update(0.1, 0.1)
    assert 0 < first < 1.0, "a filtered rate lags the raw one"
    second = damper.update(0.2, 0.2)
    assert first < second < 1.0, "and converges toward it"


def test_damper_treats_a_jump_as_a_reseed_not_a_velocity():
    damper = TurnDamper(smoothing=0.0)
    damper.update(0.0, 0.0)
    assert damper.update(0.9, 0.001) == 0.0


def test_damper_drops_a_long_gap():
    damper = TurnDamper(smoothing=0.0)
    damper.update(0.0, 0.0)
    assert damper.update(0.1, 5.0) == 0.0


def test_damper_resets_when_the_box_is_lost():
    damper = TurnDamper(smoothing=0.0)
    damper.update(0.0, 0.0)
    assert damper.update(None, 0.1) == 0.0
    assert damper.update(0.1, 0.2) == 0.0, "the first sample after a loss has no rate"


def test_damper_reports_a_closing_error_as_negative():
    damper = TurnDamper(smoothing=0.0)
    damper.update(0.7, 0.0)
    assert damper.update(0.6, 0.1) < 0, "an error shrinking must damp, not amplify"


def test_area_fraction_reported():
    cmd = command((300, 300, 700, 700))
    assert cmd.area_fraction == pytest.approx(0.16)
    assert command((420, 300, 580, 700)).area_fraction < 0.16
    assert command(None).area_fraction == 0.0
    assert command((120, 120, 880, 880)).area_fraction == pytest.approx(0.5776)