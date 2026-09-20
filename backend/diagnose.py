"""Say what this machine is actually configured to do, and then try it.

    .venv/bin/python diagnose.py            # reads the environment as-is
    set -a; . ./.env; set +a; .venv/bin/python diagnose.py

"It cannot identify anything" has at least five causes that look identical from
the driving seat: a fake provider, the red-blob fallback, a model the key
cannot reach, an upside-down or black frame, and a reply that arrived but did
not parse. This resolves each in turn and prints what it found, so the answer
comes from the machine rather than from guessing.

Safe to run while the server is up, except that a real camera can only be held
by one process -- stop the server first if the camera check reports it busy.
Never prints the API key.
"""
from __future__ import annotations

import os
import sys
import traceback

SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def redact(name: str, value: str) -> str:
    if any(hint in name.upper() for hint in SECRET_HINTS):
        return f"set, {len(value)} chars, ends {value[-4:]}" if value else "EMPTY"
    return value


def heading(text: str) -> None:
    print(f"\n{text}\n" + "-" * len(text))


def show_environment() -> None:
    heading("1. Environment the process actually sees")
    interesting = (
        "MOCK", "VISION_PROVIDER", "CAMERA", "CAMERA_ROTATION", "DRIVER",
        "HUAWEI_API_KEY", "HUAWEI_BASE_URL", "HUAWEI_MODEL", "HUAWEI_VOICE_MODEL",
        "SPEECH_PROVIDER", "VISION_WIDTH", "VISION_HEIGHT",
    )
    for name in interesting:
        value = os.environ.get(name)
        print(f"  {name:<20} {redact(name, value) if value is not None else '(unset -> code default)'}")
    if not os.path.exists(".env"):
        print("\n  NOTE: no backend/.env here. run-backend-local.sh needs one;")
        print("  starting server.py directly never reads it either way.")


def show_provider() -> None:
    heading("2. Which vision provider that resolves to")
    import vision

    provider = vision.make_vision(None)
    kind = type(provider).__name__
    print(f"  provider     {kind}")
    if kind == "FakeVision":
        print("  -> MOCK is on. Boxes are scripted and ignore the camera entirely.")
    elif kind == "LocalVision":
        print("  -> VISION_PROVIDER=local. An HSV red-blob finder: it ignores the")
        print("     target string and can only ever find red things.")
    else:
        print(f"  model        {getattr(provider, 'model', '?')}")
        print(f"  base_url     {getattr(provider, 'base_url', '?')}")
        print(f"  code default {vision.DEFAULT_MODEL}")
        if not os.environ.get("HUAWEI_API_KEY"):
            print("  -> no HUAWEI_API_KEY: every call will fail.")


def show_camera() -> None:
    heading("3. What the camera returns")
    import numpy as np
    from camera import make_camera
    from scenarios import SCENARIOS

    camera = make_camera(SCENARIOS["center"])
    try:
        frame = camera.read()
        height, width = frame.shape[:2]
        top = float(np.mean(frame[: height // 2]))
        bottom = float(np.mean(frame[height // 2:]))
        print(f"  camera       {type(camera).__name__}")
        print(f"  frame        {width}x{height}, mean brightness {float(np.mean(frame)):.1f}/255")
        print(f"  top half {top:.1f}   bottom half {bottom:.1f}")
        if float(np.mean(frame)) < 8:
            print("  -> almost black. Lens cap, no light, or the sensor is not producing.")
        print("  Saved diagnose-frame.jpg -- OPEN IT. If it is upside down, set")
        print("  CAMERA_ROTATION=0 (or 180). A flipped frame sinks identification")
        print("  while leaving every other symptom looking healthy.")
        import cv2

        cv2.imwrite("diagnose-frame.jpg", frame)
    finally:
        release = getattr(camera, "release", None)
        if release:
            release()


def show_detection(target: str) -> None:
    heading(f"4. One real detection for {target!r}")
    import time

    from camera import make_camera
    from scenarios import SCENARIOS
    from vision import make_vision

    provider = make_vision(SCENARIOS["center"])
    camera = make_camera(SCENARIOS["center"])
    try:
        frame = camera.read()
        started = time.monotonic()
        detection = provider.detect(target, frame)
        elapsed = time.monotonic() - started
    except Exception as exc:
        print(f"  RAISED after the call: {type(exc).__name__}: {exc}")
        print("\n  Full traceback:")
        traceback.print_exc()
        return
    finally:
        release = getattr(camera, "release", None)
        if release:
            release()

    print(f"  round trip   {elapsed * 1000:.0f} ms")
    raw = getattr(provider, "last_raw", None)
    print(f"  raw reply    {raw!r}")
    if detection is None:
        print("  parsed       None -- nothing came back that the parser accepted.")
        print("  -> If the raw reply above is empty or an error, it is the gateway")
        print("     or the model name. If it has content, it is the parser.")
        return
    print(f"  label        {detection.label!r}")
    print(f"  action       {detection.action!r}")
    print(f"  box_2d       {detection.box_2d}")
    print(f"  reason       {detection.reason!r}")
    if detection.box_2d is None:
        print("  -> Reached the model, which reported it cannot see the target.")
        print("     Try a blunter target, better light, or VISION_WIDTH=640.")
    else:
        print("  -> Working. Check the box actually covers the object in the frame.")


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "the red ball"
    print("=" * 66)
    print("  sparkle diagnostics")
    print("=" * 66)
    for step in (show_environment, show_provider, show_camera):
        try:
            step()
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    try:
        show_detection(target)
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    print("\nPaste all of the above, plus whether diagnose-frame.jpg looks right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
