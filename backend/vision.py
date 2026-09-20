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
DEFAULT_MODEL = "qwen3.8-omni-flash"


def is_mock() -> bool:
    return os.environ.get("MOCK", "true").strip().lower() in ("1", "true", "yes")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass
class DetectedObject:
    box_2d: Box2D
    label: str


class VisionProvider:
    def start(self, target: str) -> None:
        """Open any per-navigation resources. Stateless providers do nothing."""

    def stop(self) -> None:
        """Close any per-navigation resources. Stateless providers do nothing."""

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
        jpeg_quality: int = 70,
        max_tokens: int = 128,
    ):
        api_key = api_key or os.environ.get("HUAWEI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "HUAWEI_API_KEY is required to call the inference endpoint; "
                "drop it via MOCK=true in mock mode"
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=_env_float("VISION_TIMEOUT", 10.0),
            # A retry would re-send a frame describing where the car used to
            # be. Failing fast costs one miss and the next cycle sends a new one.
            max_retries=0,
        )
        self._model = model
        self._jpeg_quality = jpeg_quality
        self._max_tokens = max_tokens
        self._max_cost = _env_float("MAX_COST_USD", 1.0)
        self._input_rate = _env_float("VISION_INPUT_USD_PER_MILLION", 0.55)
        self._output_rate = _env_float("VISION_OUTPUT_USD_PER_MILLION", 2.20)
        self._spent = 0.0
        self.last_raw: Optional[str] = None

    @property
    def estimated_spend_usd(self) -> float:
        return self._spent

    @property
    def cost_cap_usd(self) -> float:
        return self._max_cost

    def _call_cost(self, frame: np.ndarray) -> float:
        """Upper-bound cost of one detection. Qwen bills one image token per
        32x32 pixel block; output is charged at its cap. The provider remains
        the source of truth — this only exists to stop runaway spend."""
        height, width = frame.shape[:2]
        image_tokens = ((width + 31) // 32) * ((height + 31) // 32)
        return (
            image_tokens * self._input_rate + self._max_tokens * self._output_rate
        ) / 1_000_000

    def detect(self, target: str, frame: np.ndarray) -> Optional[DetectedObject]:
        cost = self._call_cost(frame)
        if self._max_cost > 0 and self._spent + cost > self._max_cost:
            raise RuntimeError(
                f"MAX_COST_USD cap of ${self._max_cost:g} reached "
                f"(estimated ${self._spent:.4f} spent); restart to reset"
            )

        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not ok:
            return None
        self._spent += cost

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
            max_tokens=self._max_tokens,
            stream=True,
        )

        text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content

        self.last_raw = text
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
            max_tokens=int(os.environ.get("VISION_MAX_TOKENS", "128")),
        )
    return FakeVision(scene=scene or SCENARIOS.get("center", FakeScene()))
