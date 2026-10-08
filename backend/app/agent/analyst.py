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
import logging
import os
from functools import lru_cache
from typing import Any

from anthropic import Anthropic
from langfuse import get_client, observe

from app.agent.prompts import ANALYZE_AND_COACH_SYSTEM_PROMPT
from app.guardrails.classify import (
    MEDICAL_FALLBACK,
    REPAIR_INSTRUCTIONS,
    FeedbackVerdict,
    review_feedback,
)
from app.guardrails.containment import contain

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"


@lru_cache(maxsize=1)
def _client() -> Anthropic:
    # Cached: constructing a client per call meant a fresh TLS handshake every
    # time. The SDK client is thread-safe and designed to be reused.
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _evidence(swings: list[dict], observations: dict[str, Any] | None) -> str:
    # Metrics are numbers out of MediaPipe -- derived from the video but with no
    # path for text to ride along, so they go in uncontained. The observations
    # are model-written prose about whatever was visible, which is a different
    # kind of input entirely; see guardrails.containment.
    parts = [f"MEASURED METRICS (deterministic, from pose tracking):\n{json.dumps(swings, indent=2)}"]

    if observations:
        parts.append(
            "VISUAL OBSERVATIONS (from a model that watched the video; "
            "perceptual, not measured):\n"
            + contain(observations, source="gemini-video-perception")
        )
    else:
        parts.append(
            "VISUAL OBSERVATIONS: unavailable for this video -- perception failed or was "
            "skipped. Work from the metrics alone and do not speculate about anything "
            "they don't cover."
        )
    return "\n\n".join(parts)


@observe(as_type="generation", name="analyze_and_coach")
def _draft(user_content: str, previous: str | None = None, repair: str | None = None) -> str:
    """One analysis call.

    On a repair attempt the rejected draft goes back as the assistant turn it
    actually was, with the problem named in the user turn after it. The model
    needs to see what it wrote to fix it -- asking for a rewrite while
    withholding the draft is just a reroll with extra instructions.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
    if previous and repair:
        messages.append({"role": "assistant", "content": previous})
        messages.append({"role": "user", "content": repair})

    response = _client().messages.create(
        model=MODEL,
        max_tokens=4096,
        system=ANALYZE_AND_COACH_SYSTEM_PROMPT,
        messages=messages,
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


@observe(name="analyze_with_review")
def analyze(swings: list[dict], observations: dict[str, Any] | None = None) -> str:
    """Produce the player-facing coaching feedback, reviewed before it ships.

    One repair attempt, not a loop. Naming the specific problem and asking for
    a rewrite fixes this far more often than regenerating blind, but a second
    failure means the model isn't going to fix it on a third try either -- and
    an unbounded loop on a background job is how a cheap check turns into an
    expensive one.

    The two verdicts end differently on purpose. Ungrounded coaching that
    survives a repair still ships, with a warning logged: slightly overreaching
    feedback is a worse write-up, not a harmful one, and shipping nothing after
    the player waited for an analysis is its own failure. Medical advice doesn't
    ship either way.
    """
    user_content = _evidence(swings, observations)

    text = _draft(user_content)
    verdict = review_feedback(text, user_content)
    if verdict is FeedbackVerdict.OK:
        return text

    logger.warning("Feedback failed review (%s); attempting one repair", verdict)
    repaired = _draft(user_content, previous=text, repair=REPAIR_INSTRUCTIONS[verdict])

    second = review_feedback(repaired, user_content)
    if second is FeedbackVerdict.OK:
        return repaired

    if second is FeedbackVerdict.MEDICAL:
        logger.error("Feedback still gave medical advice after repair; withholding it")
        return MEDICAL_FALLBACK

    logger.warning("Feedback still reviewed as %s after repair; shipping it anyway", second)
    return repaired
