import cv2
import numpy as np
import pytest

from tracker import BoxTracker, to_normalised, to_pixels

WIDTH, HEIGHT = 320, 240


def scene(shift_x: int = 0, shift_y: int = 0, radius: int = 40) -> np.ndarray:
    """Textured background with a marked target, so there is something to track."""
    rng = np.random.default_rng(11)
    frame = rng.integers(60, 190, (HEIGHT, WIDTH), dtype=np.uint8)
    centre = (WIDTH // 3 + shift_x, HEIGHT // 2 + shift_y)
    cv2.circle(frame, centre, radius, 30, -1)
    # Speckle inside the target: a flat disc has no corners to follow.
    for offset in range(-radius + 6, radius - 5, 7):
        cv2.line(frame, (centre[0] + offset, centre[1] - radius + 6),
                 (centre[0] + offset, centre[1] + radius - 6), 220, 2)
    return frame


def box_around(shift_x: int = 0, shift_y: int = 0, radius: int = 40):
    centre_x, centre_y = WIDTH // 3 + shift_x, HEIGHT // 2 + shift_y
    return to_normalised(
        centre_x - radius, centre_y - radius,
        centre_x + radius, centre_y + radius, WIDTH, HEIGHT,
    )


def centre_of(box):
    ymin, xmin, ymax, xmax = box
    return (xmin + xmax) / 2, (ymin + ymax) / 2


def test_pixel_round_trip():
    box = (250, 100, 750, 900)
    pixels = to_pixels(box, WIDTH, HEIGHT)
    assert to_normalised(*pixels, WIDTH, HEIGHT) == box


def test_normalised_clamps_outside_the_frame():
    ymin, xmin, ymax, xmax = to_normalised(-50, -50, WIDTH + 99, HEIGHT + 99, WIDTH, HEIGHT)
    assert (xmin, ymin) == (0, 0)
    assert xmax == 1000 and ymax == 1000


def test_seed_needs_features():
    tracker = BoxTracker()
    blank = np.full((HEIGHT, WIDTH), 128, np.uint8)
    assert tracker.seed(blank, box_around()) is False
    assert tracker.tracking is False


def test_box_follows_a_moving_target():
    tracker = BoxTracker()
    assert tracker.seed(scene(), box_around()) is True

    moved = None
    for step in (6, 12, 18):
        result = tracker.update(scene(shift_x=step))
        assert result is not None, f"lost the target after {step}px"
        moved, confidence = result
        assert confidence > 0.5

    start_x, _ = centre_of(box_around())
    end_x, _ = centre_of(moved)
    expected_x, _ = centre_of(box_around(shift_x=18))
    assert end_x > start_x, "box did not follow the target"
    assert abs(end_x - expected_x) < 40, f"drifted too far: {end_x} vs {expected_x}"


def test_box_grows_as_the_target_approaches():
    tracker = BoxTracker()
    assert tracker.seed(scene(radius=40), box_around(radius=40)) is True
    result = tracker.update(scene(radius=52))
    assert result is not None
    box, _ = result
    ymin, xmin, ymax, xmax = box
    original = box_around(radius=40)
    assert (xmax - xmin) > (original[3] - original[1]), "box should grow"


def test_gives_up_once_the_box_wanders_too_far():
    """The observed failure: points drift onto background and the car chases it.

    Surviving-point count stays high the whole time, so only a bound on how far
    the box may travel from where the model put it catches this.
    """
    tracker = BoxTracker()
    assert tracker.seed(scene(), box_around()) is True

    lost_at = None
    for step in range(6, 200, 6):
        result = tracker.update(scene(shift_x=step))
        if result is None:
            lost_at = step
            break
    assert lost_at is not None, "tracker followed the target forever without confirming"
    assert lost_at > 20, f"gave up far too eagerly, at {lost_at}px"


def test_gives_up_when_the_box_leaves_the_frame():
    tracker = BoxTracker()
    # Seeded near the right edge, then pushed off it.
    start = to_normalised(WIDTH - 90, 100, WIDTH - 10, 180, WIDTH, HEIGHT)
    frame = scene()
    cv2.rectangle(frame, (WIDTH - 90, 100), (WIDTH - 10, 180), 200, -1)
    for offset in range(WIDTH - 86, WIDTH - 12, 6):
        cv2.line(frame, (offset, 104), (offset, 176), 40, 2)
    assert tracker.seed(frame, start) is True

    for _ in range(12):
        shifted = np.roll(frame, 20, axis=1)
        frame = shifted
        if tracker.update(frame) is None:
            break
    else:
        pytest.fail("kept tracking a box that had left the frame")


def test_update_without_seed_returns_none():
    assert BoxTracker().update(scene()) is None


def test_tracking_gives_up_when_the_scene_is_replaced():
    """A cut to an unrelated frame must report failure, not a confident lie."""
    tracker = BoxTracker()
    assert tracker.seed(scene(), box_around()) is True
    rng = np.random.default_rng(99)
    for _ in range(3):
        result = tracker.update(rng.integers(0, 255, (HEIGHT, WIDTH), dtype=np.uint8))
        if result is None:
            break
    else:
        pytest.fail("tracker kept reporting success on an unrelated scene")
    assert tracker.tracking is False
