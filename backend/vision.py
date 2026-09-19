from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from scenarios import FakeScene, SCENARIOS

Box2D = tuple  # (ymin, xmin, ymax, xmax) normalized 0-1000

DEFAULT_BASE_URL = "https://yibuapi.com/v1"
DEFAULT_MODEL = "qwen3.5-omni-flash"


def is_mock() -> bool:
    return os.environ.get("MOCK", "true").strip().lower() in ("1", "true", "yes")


@dataclass
class DetectedObject:
    box_2d: Box2D
    label: str


class VisionProvider:
    def detect(self, target: str, frame: np.ndarray) -> Optional[DetectedObject]:
        raise NotImplementedError


class FakeVision(VisionProvider):
    def __init__(self, scene: FakeScene):
        self._scene = scene

    def detect(self, target: str, frame: np.ndarray) -> Optional[DetectedObject]:
        box_2d = self._scene.advance()
        if box_2d is None:
            return None
        return DetectedObject(box_2d=box_2d, label=target)


class OmniVision(VisionProvider):
    def __init__(
        self,
        api_key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        jpeg_quality: int = 85,
    ):
        api_key = api_key or os.environ.get("HUAWEI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "HUAWEI_API_KEY is required to call the inference endpoint; "
                "drop it via MOCK=true in mock mode"
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._jpeg_quality = jpeg_quality

    def detect(self, target: str, frame: np.ndarray) -> Optional[DetectedObject]:
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not ok:
            return None

        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

        prompt = (
            f"Detect the object described as '{target}' in this image. "
            'Respond with a JSON array of bounding boxes matching exactly this shape: '
            '[{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}, ...] '
            "with coordinates normalized to 0-1000. "
            "If the object is not present, respond with []. Return only the JSON."
        )

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )

        text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content

        boxes = _parse_boxes(text)
        if not boxes:
            return None
        box = boxes[0]
        return DetectedObject(box_2d=tuple(box["box_2d"]), label=box.get("label", target))


def _parse_boxes(text: str) -> List[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return [b for b in data if isinstance(b, dict) and len(b.get("box_2d", [])) == 4]


def make_vision(scene: Optional[FakeScene] = None) -> VisionProvider:
    if not is_mock():
        return OmniVision(
            api_key=os.environ.get("HUAWEI_API_KEY", ""),
            base_url=os.environ.get("HUAWEI_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("HUAWEI_MODEL", DEFAULT_MODEL),
        )
    return FakeVision(scene=scene or SCENARIOS.get("center", FakeScene()))