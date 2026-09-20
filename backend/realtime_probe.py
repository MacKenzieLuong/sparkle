"""Find out whether the Qwen Realtime WebSocket accepts an image, and how fast.

    export HUAWEI_API_KEY=<key>
    python realtime_probe.py

An earlier attempt at this concluded Realtime could not take images. It used
`input_image_buffer.append`, which is not the protocol: yibuapi's own example
sends content through `conversation.item.create`, and names the model
`qwen3.5-omni-plus-realtime` rather than the flash variant that has no access.
This probe asks the right way and tries each plausible image shape.

It also times a second and third request on the *same* connection, which is the
number that matters: if the handshake is what costs seconds, a held-open socket
removes it from every subsequent frame.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from typing import Any, Optional
from urllib.parse import urlencode

import cv2
import numpy as np

DEFAULT_MODEL = "qwen3.5-omni-plus-realtime"
DEFAULT_ENDPOINT = "wss://yibuapi.com/v1/realtime"

PROMPT = (
    'Reply with only compact JSON: {"action":"...","box_2d":[ymin,xmin,ymax,xmax]}. '
    "Find the red ball. box_2d normalized 0-1000, or null if absent."
)


def test_frame(width: int = 320, height: int = 240) -> str:
    """A grey frame with one red circle, right of centre. Known ground truth."""
    frame = np.full((height, width, 3), 210, np.uint8)
    cv2.circle(frame, (int(width * 0.7), height // 2), height // 6, (40, 40, 220), -1)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        raise RuntimeError("could not encode the test frame")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def image_shapes(data_url: str) -> list[tuple[str, dict]]:
    """Every plausible spelling for an image content part, most likely first."""
    return [
        ("input_image + bare url", {"type": "input_image", "image_url": data_url}),
        ("input_image + nested url", {"type": "input_image", "image_url": {"url": data_url}}),
        ("image_url + nested url", {"type": "image_url", "image_url": {"url": data_url}}),
        ("input_image + image", {"type": "input_image", "image": data_url}),
    ]


def receive(ws, timeout: float) -> dict[str, Any]:
    event = json.loads(ws.recv(timeout=timeout))
    if not isinstance(event, dict):
        raise RuntimeError("server event is not a JSON object")
    return event


def collect_response(ws, timeout: float = 90.0) -> tuple[str, dict]:
    """Read events until response.done, returning the text and that event."""
    parts: list[str] = []
    while True:
        event = receive(ws, timeout)
        kind = str(event.get("type") or "")
        if kind == "error":
            raise RuntimeError(json.dumps(event.get("error") or event, ensure_ascii=False))
        if kind in ("response.text.delta", "response.audio_transcript.delta"):
            parts.append(str(event.get("delta") or ""))
        elif kind == "response.text.done" and event.get("text"):
            parts = [str(event["text"])]
        elif kind == "response.audio_transcript.done" and not parts:
            parts = [str(event.get("transcript") or "")]
        elif kind == "response.done":
            return "".join(parts).strip(), event


def ask(ws, content: list[dict], timeout: float = 90.0) -> tuple[str, dict, float]:
    started = time.monotonic()
    ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user", "content": content},
    }, ensure_ascii=False))
    ws.send(json.dumps({"type": "response.create"}))
    text, done = collect_response(ws, timeout)
    return text, done, time.monotonic() - started


def usage_of(done: dict) -> str:
    usage = (done.get("response") or {}).get("usage") or {}
    if not usage:
        return "usage not reported"
    return (
        f"in={usage.get('input_tokens')} out={usage.get('output_tokens')} "
        f"total={usage.get('total_tokens')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("REALTIME_MODEL", DEFAULT_MODEL))
    parser.add_argument("--endpoint", default=os.environ.get("REALTIME_URL", DEFAULT_ENDPOINT))
    parser.add_argument("--repeat", type=int, default=3, help="requests on one socket")
    args = parser.parse_args()

    api_key = os.environ.get("HUAWEI_API_KEY") or os.environ.get("YIBU_API_KEY") or ""
    if not api_key:
        print("set HUAWEI_API_KEY (or YIBU_API_KEY)", file=sys.stderr)
        return 2

    try:
        from websockets.sync.client import connect
    except ImportError:
        print("pip install 'websockets>=15,<17'", file=sys.stderr)
        return 2

    url = args.endpoint + ("&" if "?" in args.endpoint else "?") + urlencode({"model": args.model})
    data_url = test_frame()
    print(f"model    : {args.model}")
    print(f"endpoint : {url}")
    print(f"image    : {len(data_url)} chars of base64 data URL\n")

    connect_started = time.monotonic()
    # proxy=None so HTTP(S)_PROXY/ALL_PROXY are never picked up, per yibu's example.
    with connect(
        url,
        additional_headers={"Authorization": f"Bearer {api_key}"},
        proxy=None,
        open_timeout=30,
        close_timeout=5,
        max_size=32 * 1024 * 1024,
    ) as ws:
        created = receive(ws, 20)
        if created.get("type") != "session.created":
            print(f"expected session.created, got {created.get('type')}: {created}")
            return 1
        print(f"connected in {(time.monotonic() - connect_started) * 1000:.0f} ms")

        ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "instructions": "You are a vision detector. Reply only with JSON.",
                "turn_detection": None,
            },
        }, ensure_ascii=False))
        updated = receive(ws, 20)
        if updated.get("type") != "session.updated":
            print(f"expected session.updated, got {updated.get('type')}: {updated}")
            return 1
        print("session configured for text output\n")

        # Text first: proves the socket works before blaming the image.
        try:
            text, done, elapsed = ask(ws, [{"type": "input_text", "text": "Reply with the word OK."}])
            print(f"text-only     {elapsed * 1000:7.0f} ms  {text!r}  ({usage_of(done)})")
        except Exception as exc:
            print(f"text-only     FAILED: {exc}")
            return 1

        working: Optional[dict] = None
        for name, part in image_shapes(data_url):
            try:
                text, done, elapsed = ask(ws, [part, {"type": "input_text", "text": PROMPT}])
                print(f"{name:26} {elapsed * 1000:7.0f} ms  {text!r}  ({usage_of(done)})")
                working = part
                break
            except Exception as exc:
                message = str(exc).replace("\n", " ")[:160]
                print(f"{name:26} rejected: {message}")

        if working is None:
            print("\nNo image shape accepted. Realtime is text-only for this model;")
            print("the HTTP path stays the only way to send frames.")
            return 1

        print(f"\nImage accepted. Timing {args.repeat} more on the SAME socket:")
        timings = []
        for attempt in range(args.repeat):
            _text, _done, elapsed = ask(ws, [working, {"type": "input_text", "text": PROMPT}])
            timings.append(elapsed)
            print(f"  request {attempt + 2}: {elapsed * 1000:7.0f} ms")
        print(f"\nmedian on a warm socket: {sorted(timings)[len(timings) // 2] * 1000:.0f} ms")
        print("Compare against the HTTP path's measured 2.5-4.3s per call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
