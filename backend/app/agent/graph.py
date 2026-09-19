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


@observe()
def diagnose_flaws(state: AgentState) -> dict:
    metrics_json = json.dumps(state["swings"], indent=2)
    diagnosis = _call_claude(DIAGNOSE_SYSTEM_PROMPT, f"Swing metrics:\n{metrics_json}")
    return {"diagnosis": diagnosis}


@observe()
def prioritize(state: AgentState) -> dict:
    priorities = _call_claude(PRIORITIZE_SYSTEM_PROMPT, f"Observations:\n{state['diagnosis']}")
    return {"priorities": priorities}


@observe()
def synthesize_feedback(state: AgentState) -> dict:
    feedback = _call_claude(SYNTHESIZE_SYSTEM_PROMPT, f"Priorities:\n{state['priorities']}")
    return {"feedback": feedback}


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
