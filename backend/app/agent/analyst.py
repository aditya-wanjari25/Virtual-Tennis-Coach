"""Turn measured metrics + visual observations into coaching feedback.

One Claude call. This was a three-node LangGraph (diagnose -> prioritize ->
synthesize); measured at ~46s for output no better than a single ~8s call. The
chain was a linear sequence with no branching, no loops, no tool use and no
persisted state -- i.e. a function call chain, which is what it is now.

LangGraph is still the right tool for the chat feature, which genuinely needs
conditional edges, cycles, tool nodes and checkpointed state. It just wasn't
doing anything here.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from anthropic import Anthropic
from langfuse import get_client, observe

from app.agent.prompts import ANALYZE_AND_COACH_SYSTEM_PROMPT

MODEL = "claude-sonnet-5"


@lru_cache(maxsize=1)
def _client() -> Anthropic:
    # Cached: constructing a client per call meant a fresh TLS handshake every
    # time. The SDK client is thread-safe and designed to be reused.
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _evidence(swings: list[dict], observations: dict[str, Any] | None) -> str:
    parts = [f"MEASURED METRICS (deterministic, from pose tracking):\n{json.dumps(swings, indent=2)}"]

    if observations:
        parts.append(
            "VISUAL OBSERVATIONS (from a model that watched the video; "
            f"perceptual, not measured):\n{json.dumps(observations, indent=2)}"
        )
    else:
        parts.append(
            "VISUAL OBSERVATIONS: unavailable for this video -- perception failed or was "
            "skipped. Work from the metrics alone and do not speculate about anything "
            "they don't cover."
        )
    return "\n\n".join(parts)


@observe(as_type="generation", name="analyze_and_coach")
def analyze(swings: list[dict], observations: dict[str, Any] | None = None) -> str:
    """Produce the player-facing coaching feedback."""
    user_content = _evidence(swings, observations)

    response = _client().messages.create(
        model=MODEL,
        max_tokens=4096,
        system=ANALYZE_AND_COACH_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")

    get_client().update_current_generation(
        model=MODEL,
        input=[
            {"role": "system", "content": ANALYZE_AND_COACH_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        output=text,
        usage_details={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
    )
    return text
