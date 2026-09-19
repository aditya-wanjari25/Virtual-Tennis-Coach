"""LangGraph coaching pipeline: diagnose_flaws -> prioritize -> synthesize_feedback.

Each node makes one Claude call. State flows through as an AgentState
dict; every node returns only the field(s) it updates and LangGraph
merges that into the running state.
"""

from __future__ import annotations

import json
import os

from anthropic import Anthropic
from langfuse import get_client, observe
from langgraph.graph import END, START, StateGraph

from app.agent.prompts import (
    DIAGNOSE_SYSTEM_PROMPT,
    PRIORITIZE_SYSTEM_PROMPT,
    SYNTHESIZE_SYSTEM_PROMPT,
)
from app.agent.state import AgentState

MODEL = "claude-sonnet-5"


@observe(as_type="generation", name="claude_call")
def _call_claude(system: str, user_content: str) -> str:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")

    get_client().update_current_generation(
        model=MODEL,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user_content}],
        output=text,
        usage_details={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
    )
    return text


def _evidence(state: AgentState) -> str:
    """The two evidence sources, rendered once and passed to every node.

    Previously each node received only the *previous node's prose*, so by the
    time we reached synthesize_feedback the model no longer had the metrics or
    the observations at all -- it was writing coaching advice from a summary of
    a summary. Every node now sees the underlying evidence.
    """
    parts = [f"MEASURED METRICS (deterministic, from pose tracking):\n{json.dumps(state['swings'], indent=2)}"]

    observations = state.get("observations")
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


@observe()
def diagnose_flaws(state: AgentState) -> dict:
    diagnosis = _call_claude(DIAGNOSE_SYSTEM_PROMPT, _evidence(state))
    return {"diagnosis": diagnosis}


@observe()
def prioritize(state: AgentState) -> dict:
    user = f"{_evidence(state)}\n\nOBSERVATIONS FROM THE ANALYST:\n{state['diagnosis']}"
    return {"priorities": _call_claude(PRIORITIZE_SYSTEM_PROMPT, user)}


@observe()
def synthesize_feedback(state: AgentState) -> dict:
    user = (
        f"{_evidence(state)}\n\n"
        f"OBSERVATIONS FROM THE ANALYST:\n{state['diagnosis']}\n\n"
        f"PRIORITIES TO COVER:\n{state['priorities']}"
    )
    return {"feedback": _call_claude(SYNTHESIZE_SYSTEM_PROMPT, user)}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("diagnose_flaws", diagnose_flaws)
    graph.add_node("prioritize", prioritize)
    graph.add_node("synthesize_feedback", synthesize_feedback)

    graph.add_edge(START, "diagnose_flaws")
    graph.add_edge("diagnose_flaws", "prioritize")
    graph.add_edge("prioritize", "synthesize_feedback")
    graph.add_edge("synthesize_feedback", END)

    return graph.compile()
