import numpy as np
import pytest

from vision import OmniVision, _parse_boxes


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_plain_json_array():
    boxes = _parse_boxes('[{"box_2d": [100, 200, 300, 400], "label": "red ball"}]')
    assert len(boxes) == 1
    assert boxes[0]["box_2d"] == [100, 200, 300, 400]


def test_fenced_json_block():
    text = '```json\n[{"box_2d": [0, 0, 500, 500], "label": "door"}]\n```'
    boxes = _parse_boxes(text)
    assert len(boxes) == 1
    assert boxes[0]["label"] == "door"


def test_prose_around_array():
    text = 'Sure! Here is the result:\n[{"box_2d": [350, 450, 650, 700], "label": "cone"}]\nHope this helps.'
    boxes = _parse_boxes(text)
    assert len(boxes) == 1
    assert boxes[0]["box_2d"] == [350, 450, 650, 700]


def test_empty_object_not_found():
    assert _parse_boxes("[]") == []


def test_malformed_returns_empty():
    assert _parse_boxes("not json at all") == []


def test_invalid_entries_skipped():
    text = (
        '[{"box_2d": [1, 2, 3, 4], "label": "ok"}, '
        '{"box_2d": [1, 2, 3], "label": "bad"}, "string"]'
    )
    boxes = _parse_boxes(text)
    assert len(boxes) == 1
    assert boxes[0]["label"] == "ok"


def test_object_root_returns_empty():
    assert _parse_boxes('{"box_2d": [1, 2, 3, 4], "label": "x"}') == []


def test_out_of_range_coordinates_passthrough():
    boxes = _parse_boxes('[{"box_2d": [9999, 9999, 9999, 9999], "label": "x"}]')
    assert len(boxes) == 1


def test_cost_cap_defaults_to_one_dollar(monkeypatch):
    monkeypatch.delenv("MAX_COST_USD", raising=False)
    assert OmniVision(api_key="test-key").cost_cap_usd == 1.0


def test_cost_estimate_for_a_640x480_frame(monkeypatch):
    monkeypatch.setenv("VISION_INPUT_USD_PER_MILLION", "0.55")
    monkeypatch.setenv("VISION_OUTPUT_USD_PER_MILLION", "2.20")
    vision = OmniVision(api_key="test-key")
    # 20 x 15 = 300 image tokens of input, plus the 128-token output cap
    assert vision._call_cost(_frame()) == pytest.approx(
        (300 * 0.55 + 128 * 2.20) / 1_000_000
    )


def test_cap_blocks_the_call_before_spending(monkeypatch):
    monkeypatch.setenv("MAX_COST_USD", "0.0001")
    vision = OmniVision(api_key="test-key")
    with pytest.raises(RuntimeError, match="MAX_COST_USD"):
        vision.detect("ball", _frame())
    assert vision.estimated_spend_usd == 0.0, "a blocked call must not be billed"


def test_cap_trips_on_the_call_that_would_cross_it(monkeypatch):
    monkeypatch.setenv("MAX_COST_USD", "1.0")
    vision = OmniVision(api_key="test-key")
    vision._spent = 1.0 - vision._call_cost(_frame()) / 2
    with pytest.raises(RuntimeError, match="cap"):
        vision.detect("ball", _frame())
