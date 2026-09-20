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


# What the model is allowed to decide. Local code executes these; it never
# invents one, and anything outside this set is treated as a stop.
ACTIONS = ("approach", "search_left", "search_right", "back_off", "stop")


@dataclass
class DetectedObject:
    box_2d: Optional[Box2D]
    label: str
    action: str = "approach"
    reason: str = ""


class VisionProvider:
    def start(self, target: str) -> None:
        """Open any per-navigation resources. Stateless providers do nothing."""

    def stop(self) -> None:
        """Close any per-navigation resources. Stateless providers do nothing."""

    def detect(self, target: str, frame: np.ndarray) -> Optional[DetectedObject]:
        raise NotImplementedError


class FakeVision(VisionProvider):
    def __init__(self, scene: FakeScene, missing_action: str = "search_right"):
        self._scene = scene
        # A scripted "not in frame" stands in for the model deciding to look
        # around, which is what the real one does rather than returning nothing.
        self._missing_action = missing_action

    def detect(self, target: str, frame: np.ndarray) -> Optional[DetectedObject]:
        box_2d = self._scene.advance()
        if box_2d is None:
            return DetectedObject(
                box_2d=None, label=target,
                action=self._missing_action, reason="scripted miss",
            )
        return DetectedObject(box_2d=box_2d, label=target, action="approach")


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
        self._history: List[str] = []
        self.last_raw: Optional[str] = None
        # Downscale before sending: image tokens go as the area, so halving
        # each side quarters them. 0 keeps the camera's own resolution.
        self._send_width = int(os.environ.get("VISION_WIDTH", "0"))
        self._send_height = int(os.environ.get("VISION_HEIGHT", "0"))
        # A reason is worth reading while tuning, but it is output tokens, and
        # output is generated one token at a time — it costs wall clock.
        self._explain = os.environ.get("VISION_EXPLAIN", "false").lower() in (
            "1", "true", "yes",
        )

    def start(self, target: str) -> None:
        self._history.clear()

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

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        if self._send_width > 0 and self._send_height > 0:
            return cv2.resize(frame, (self._send_width, self._send_height))
        return frame

    def detect(self, target: str, frame: np.ndarray) -> Optional[DetectedObject]:
        frame = self._downscale(frame)
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

        prompt = self._prompt(target)

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
        plan = _parse_plan(text, target)
        if plan is None:
            return None
        self._history.append(plan.action)
        del self._history[:-HISTORY_LENGTH]
        return plan

    def _prompt(self, target: str) -> str:
        recent = ", ".join(self._history) if self._history else "none"
        shape = (
            '{"action":"...","box_2d":[ymin,xmin,ymax,xmax] or null,"reason":"..."}'
            if self._explain
            else '{"action":"...","box_2d":[ymin,xmin,ymax,xmax] or null}'
        )
        prompt = (
            f"Drive a robot car toward: '{target}'. Recent actions: {recent}.\n"
            f"Reply with only compact JSON, no whitespace: {shape}\n"
            "action: approach (goal visible, give box_2d) | search_left | "
            "search_right (not visible, keep turning the same way) | "
            "back_off (blocked) | stop (reached it).\n"
            "box_2d is normalized 0-1000."
        )
        if self._explain:
            prompt += " reason: under 6 words."
        return prompt


HISTORY_LENGTH = 4


def _parse_plan(text: str, target: str) -> Optional[DetectedObject]:
    """Read the model's decision, tolerating the shapes it actually emits.

    Falls back to the older bare-array-of-boxes reply, and refuses to invent an
    action: anything unrecognised becomes an approach when a box came with it,
    and a stop when none did.
    """
    entry = _parse_object(text)
    if entry is None:
        boxes = _parse_boxes(text)
        if not boxes:
            return None
        entry = boxes[0]

    coords = _coords_of(entry)
    box = tuple(coords) if coords else None
    action = entry.get("action")
    if not isinstance(action, str) or action not in ACTIONS:
        action = "approach" if box else "stop"
    if action == "approach" and box is None:
        action = "stop"

    label = entry.get("label")
    reason = entry.get("reason")
    return DetectedObject(
        box_2d=box,
        label=label if isinstance(label, str) and label else target,
        action=action,
        reason=reason if isinstance(reason, str) else "",
    )


def _parse_object(text: str) -> Optional[dict]:
    """The first JSON object in the reply, unwrapping a single-element list."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    for candidate in (text, _first_match(r"\{[\s\S]*\}", text)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
        if isinstance(data, dict):
            return data
    return None


def _first_match(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    return match.group(0) if match else None


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

    boxes = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        coords = _coords_of(entry)
        if coords is None:
            continue
        box = {"box_2d": coords}
        if "label" in entry:
            box["label"] = entry["label"]
        boxes.append(box)
    return boxes


# Qwen answers with "bbox_2d" where Gemini uses "box_2d". Accepting both costs
# nothing and the alternative is silently discarding every real detection.
BOX_KEYS = ("box_2d", "bbox_2d", "bbox", "box", "bounding_box")


def _coords_of(entry: dict) -> Optional[List]:
    for key in BOX_KEYS:
        value = entry.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return list(value)
    return None


def make_vision(scene: Optional[FakeScene] = None) -> VisionProvider:
    if not is_mock():
        return OmniVision(
            api_key=os.environ.get("HUAWEI_API_KEY", ""),
            base_url=os.environ.get("HUAWEI_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("HUAWEI_MODEL", DEFAULT_MODEL),
            max_tokens=int(os.environ.get("VISION_MAX_TOKENS", "128")),
        )
    return FakeVision(scene=scene or SCENARIOS.get("center", FakeScene()))
