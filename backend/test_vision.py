import pytest

from vision import _parse_boxes


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