"""Video perception via Gemini -- the half of the analysis our pose metrics
structurally cannot do.

The deterministic CV layer measures angles and distances precisely but is blind
to anything that only exists in motion or in depth: knee bend, balance, contact
height, spacing to the ball, split-step timing. Claude has no video input, so
this runs on Gemini, which does.

Division of labour (enforced by the system prompt): metrics own anything
numeric or cross-swing; perception owns everything else. Where they overlap and
disagree, that disagreement is a signal -- see the agent prompts.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

from google import genai
from langfuse import get_client, observe

from app.analysis.landmarks import PoseSequence
from app.analysis.phases import Swing

logger = logging.getLogger(__name__)

MODEL = "gemini-3.8-flash"

# Whole clip, sampled at 6 FPS. Measured against the alternatives on the sample
# video (see cv_spike/gemini_spike.py):
#   whole @ 1 FPS   1.4k tokens, 14s -- misjudged fast motion at contact
#   clipped @ 10FPS  13k tokens, 31s -- missed the split-step entirely, because
#                                       clipping cuts out the between-shot window
#   whole @ 6 FPS    8k tokens, 17s -- caught both, cheaper and faster than clipped
# Frame rate is what mattered; windowing actively hurt.
FPS = 6.0

_MIME_BY_SUFFIX = {
    ".mov": "video/mov",
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".avi": "video/avi",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".flv": "video/x-flv",
    ".3gp": "video/3gpp",
}

SYSTEM_INSTRUCTION = """\
You are a tennis biomechanics analyst watching practice footage, filmed from
behind the baseline.

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
  - what happens BETWEEN shots: recovery, footwork adjustments, readiness
  - anything visibly wrong or visibly good that a coach would call out

Report only what you actually see. If a swing is partially out of frame, or the
footage doesn't let you judge something, say so plainly rather than guessing --
a stated "couldn't tell" is more useful to us than a confident invention.
"""

# Free-text fields rather than scores: these are qualitative perceptions feeding
# a reasoning step, and forcing them onto numeric scales would invent precision
# that isn't there. `not_visible` exists to give the model somewhere to put
# uncertainty other than into a confident-sounding observation.
RESPONSE_SCHEMA: dict[str, Any] = {
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
        "between_shots": {"type": "string"},
        "across_swings": {"type": "string"},
        "doing_well": {"type": "string"},
    },
    "required": ["swings", "between_shots", "across_swings", "doing_well"],
}


def _mime_for(video_path: Path) -> str:
    return _MIME_BY_SUFFIX.get(video_path.suffix.lower(), "video/mp4")


@observe(as_type="generation", name="gemini_perception")
def _call_gemini(video_b64: str, mime: str, prompt: str) -> dict[str, Any]:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    result = client.interactions.create(
        model=MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        # response_format IS the JSON Schema -- its `type` is the root JSON type,
        # not an OpenAI-style {"type": "json_schema"} wrapper.
        response_format=RESPONSE_SCHEMA,
        input=[
            {"type": "text", "text": prompt},
            {
                "type": "video",
                "data": video_b64,
                "mime_type": mime,
                "resolution": "high",
                "processing": {"type": "static", "fps": FPS},
            },
        ],
    )

    text = getattr(result, "output_text", None) or str(result)
    observations = json.loads(text)

    # Langfuse is provider-agnostic but won't auto-detect a non-Anthropic call,
    # so model and usage have to be set by hand. The video itself is omitted
    # from the traced input -- 6MB of base64 in every trace is not useful.
    usage = getattr(result, "usage", None)
    get_client().update_current_generation(
        model=MODEL,
        input=[{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": prompt}],
        output=observations,
        usage_details=(
            {"input": usage.total_input_tokens, "output": usage.total_output_tokens} if usage else None
        ),
        metadata={"fps": FPS, "resolution": "high"},
    )
    return observations


def analyze_video(
    video_path: str | Path,
    pose: PoseSequence,
    swings: list[Swing],
) -> dict[str, Any] | None:
    """Watch the video and return qualitative observations keyed by swing index.

    The CV-derived contact times go into the prompt so Gemini numbers its swings
    the same way our metrics do -- that alignment is what lets the reasoning step
    put a measurement and an observation about the same swing side by side.

    Returns None if perception fails for any reason. That is deliberate: this is
    a supplementary signal, and losing it should degrade the analysis to
    metrics-only rather than fail the user's upload.
    """
    video_path = Path(video_path)
    contact_times = [float(pose.timestamps_s[s.contact_frame]) for s in swings]

    prompt = (
        f"The clip contains {len(contact_times)} groundstrokes. Ball contact happens at "
        f"{', '.join(f'{t:.2f}s' for t in contact_times)}. "
        f"Number them 1 to {len(contact_times)} in that order and analyze each. "
        "Also report what the player does between shots -- recovery, split-step, footwork."
    )

    try:
        video_b64 = base64.b64encode(video_path.read_bytes()).decode("utf-8")
        return _call_gemini(video_b64, _mime_for(video_path), prompt)
    except Exception:
        logger.exception("Gemini perception failed; continuing with metrics only")
        return None
