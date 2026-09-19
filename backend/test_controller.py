import pytest

from controller import command


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


def test_area_fraction_reported():
    cmd = command((300, 300, 700, 700))
    assert cmd.area_fraction == pytest.approx(0.16)
    assert command((420, 300, 580, 700)).area_fraction < 0.16
    assert command(None).area_fraction == 0.0
    assert command((120, 120, 880, 880)).area_fraction == pytest.approx(0.5776)