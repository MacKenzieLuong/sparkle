from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

Box2D = tuple  # (ymin, xmin, ymax, xmax) normalized 0-1000

_UNSET = object()


@dataclass
class FakeScene:
    boxes: List[Optional[Box2D]] = field(default_factory=list)
    index: int = 0
    override: object = _UNSET

    def peek(self):
        if self.override is not _UNSET:
            return self.override
        if not self.boxes:
            return None
        return self.boxes[self.index % len(self.boxes)]

    def advance(self):
        box = self.peek()
        if self.override is _UNSET:
            self.index += 1
        return box

    def reset(self):
        self.index = 0

    def set_override(self, box):
        self.override = box

    def clear_override(self):
        self.override = _UNSET


SCENARIOS = {
    "center": FakeScene(boxes=[(350, 400, 650, 600)]),
    "left": FakeScene(boxes=[(300, 50, 700, 250)]),
    "right": FakeScene(boxes=[(300, 750, 700, 950)]),
    "slight_right": FakeScene(boxes=[(300, 580, 700, 720)]),
    "slight_left": FakeScene(boxes=[(300, 280, 700, 420)]),
    "huge": FakeScene(boxes=[(120, 120, 880, 880)]),
    "not_found": FakeScene(boxes=[None]),
    "approach": FakeScene(
        boxes=[
            (380, 100, 620, 900),
            (300, 150, 700, 850),
            (200, 200, 800, 800),
            (100, 100, 900, 900),
        ]
    ),
}