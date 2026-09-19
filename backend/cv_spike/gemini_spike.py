"""
Throwaway spike: send the swing video to Gemini for perception and see whether
it tells us things our deterministic metrics structurally cannot.

This is the Phase 1 gate. If the output here isn't clearly better than what the
metrics alone produce, the rest of the perception plan isn't worth building.

Two modes, so we can actually compare rather than assume:

  --mode whole    whole clip at default 1 FPS (roughly what note.txt's Gemini run saw)
  --mode clipped  one video block per detected swing, clipped to that swing's
                  window at high FPS (our CV layer telling Gemini where to look)

Usage:
    GEMINI_API_KEY=... uv run python cv_spike/gemini_spike.py --mode clipped
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from app.analysis.landmarks import load_pose_sequence
from app.analysis.phases import segment_swings

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
VIDEO_PATH = SCRIPT_DIR / "sample_videos" / "forehand.MOV"
LANDMARKS_PATH = SCRIPT_DIR / "output" / "forehand_landmarks.json"

MODEL = "gemini-3.8-flash"

# How much of the swing to show Gemini around each contact. Wider than the
# metrics window on purpose -- preparation and recovery are exactly the things
# the metrics can't see, so we want them in frame.
PRE_CONTACT_S = 1.0
POST_CONTACT_S = 0.8

SYSTEM_INSTRUCTION = """\
You are a tennis biomechanics analyst watching practice footage. The player is
right-handed, hitting forehands against a wall, filmed from behind the baseline.

You are the PERCEPTION half of a two-part system. A separate deterministic
pose-tracking pipeline already measures, reliably and reproducibly:
  - shoulder and hip rotation angles, and their separation, at contact
  - swing path width, arm extension at contact, stance width at contact
  - follow-through finish height
  - how each of those changes from swing to swing

Do not report those. They are covered, and measured more precisely than you can
see them. Your job is everything that pipeline is blind to, which is mostly
things that only exist in motion or in depth:

  - lower body: knee bend, hip height, how loaded or upright the player is
  - balance: is the player over their base, falling away, recovering well
  - timing and rhythm: preparation, how early the unit turn starts, split-step,
    how rushed or unhurried the swing looks
  - spacing to the ball: jammed, reaching, comfortable
  - contact point in space relative to the body (front/late, high/low)
  - anything visibly wrong or visibly good that a coach would call out

Report only what you actually see. If a swing is partially out of frame, or the
footage doesn't let you judge something, say so plainly rather than guessing --
a stated "couldn't tell" is more useful to us than a confident invention.
"""


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# Per-swing observations. Free-text fields rather than enums/scores on purpose:
# this is the spike, and over-constraining now would hide what Gemini actually
# notices unprompted.
SCHEMA = {
    "type": "object",
    "properties": {
        "swings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "swing_index": {"type": "integer"},
                    "lower_body": {"type": "string"},
                    "balance": {"type": "string"},
                    "timing_and_preparation": {"type": "string"},
                    "spacing_to_ball": {"type": "string"},
                    "contact_point": {"type": "string"},
                    "not_visible": {"type": "string"},
                },
                "required": [
                    "swing_index",
                    "lower_body",
                    "balance",
                    "timing_and_preparation",
                    "spacing_to_ball",
                    "contact_point",
                    "not_visible",
                ],
            },
        },
        "across_swings": {"type": "string"},
        "doing_well": {"type": "string"},
    },
    "required": ["swings", "across_swings", "doing_well"],
}


def build_input(mode: str, contact_times: list[float], video_b64: str, mime: str, fps: float) -> list[dict]:
    if mode == "whole":
        video: dict = {"type": "video", "data": video_b64, "mime_type": mime, "resolution": "high"}
        if fps:
            # Whole clip but sampled faster than the 1 FPS default. Keeps the
            # between-shot footwork (split-step, recovery) that clipping cuts
            # away, while still resolving the fast motion around contact.
            video["processing"] = {"type": "static", "fps": fps}
        return [
            {
                "type": "text",
                "text": (
                    f"The clip contains {len(contact_times)} forehands. Ball contact happens at "
                    f"{', '.join(f'{t:.2f}s' for t in contact_times)}. "
                    "Number them 1, 2, 3 in that order and analyze each. Also pay attention to what "
                    "the player does BETWEEN shots -- recovery, split-step, footwork adjustments."
                ),
            },
            video,
        ]

    # Clipped: one video block per swing, each windowed around its contact at
    # high FPS. Default 1 FPS would sample ~1 frame per swing -- useless when
    # the thing you're analyzing lasts under a second.
    parts: list[dict] = [
        {
            "type": "text",
            "text": (
                f"{len(contact_times)} forehand swings follow, one video clip each, in order. "
                "Each clip is windowed around ball contact. Analyze each clip as its own swing, "
                "numbered by the label that precedes it."
            ),
        }
    ]
    for i, t in enumerate(contact_times, start=1):
        start = max(0.0, t - PRE_CONTACT_S)
        end = t + POST_CONTACT_S
        parts.append({"type": "text", "text": f"--- Swing {i} (contact at {t:.2f}s in the original clip) ---"})
        parts.append(
            {
                "type": "video",
                "data": video_b64,
                "mime_type": mime,
                "resolution": "high",
                "processing": {
                    "type": "static",
                    "fps": 10.0,
                    "start_offset": f"{start:.2f}s",
                    "end_offset": f"{end:.2f}s",
                },
            }
        )
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["whole", "clipped"], default="clipped")
    ap.add_argument("--no-schema", action="store_true", help="skip response_format, take prose back")
    ap.add_argument("--fps", type=float, default=0.0, help="whole mode: sample rate (0 = API default 1 FPS)")
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not set (expected in backend/.env)")

    pose = load_pose_sequence(LANDMARKS_PATH)
    swings = segment_swings(pose, hand="right")
    contact_times = [float(pose.timestamps_s[s.contact_frame]) for s in swings]
    print(f"{len(swings)} swings, contact at {[f'{t:.2f}s' for t in contact_times]}")

    video_b64 = _b64(VIDEO_PATH)
    size_mb = VIDEO_PATH.stat().st_size / 1e6
    print(f"video: {VIDEO_PATH.name} ({size_mb:.1f} MB), mode={args.mode}, fps={args.fps or 'default'}, schema={not args.no_schema}\n")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    kwargs: dict = {
        "model": MODEL,
        "system_instruction": SYSTEM_INSTRUCTION,
        "input": build_input(args.mode, contact_times, video_b64, "video/mov", args.fps),
    }
    if not args.no_schema:
        # response_format IS the JSON Schema -- its `type` is the root JSON type
        # ('object' here), not an OpenAI-style "json_schema" wrapper.
        kwargs["response_format"] = SCHEMA

    t0 = time.monotonic()
    result = client.interactions.create(**kwargs)
    elapsed = time.monotonic() - t0

    print(f"--- latency: {elapsed:.1f}s ---")
    usage = getattr(result, "usage", None)
    if usage:
        print(f"--- usage: {usage} ---")
    print()

    text = getattr(result, "output_text", None) or str(result)
    try:
        print(json.dumps(json.loads(text), indent=2))
    except (json.JSONDecodeError, TypeError):
        print(text)


if __name__ == "__main__":
    main()
