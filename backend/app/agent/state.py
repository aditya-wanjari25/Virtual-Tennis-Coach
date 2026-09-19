"""Shared state that flows through the coaching LangGraph.

Each node function takes the current state and returns a dict of the
fields it wants to update -- LangGraph merges that into the running
state before calling the next node.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class AgentState(TypedDict):
    swings: list[dict]  # serialized SwingMetrics, one per detected swing

    # Qualitative observations from the video-perception step. NotRequired
    # because perception is supplementary -- if Gemini fails we still run, on
    # metrics alone. Nodes must handle its absence.
    observations: NotRequired[dict[str, Any] | None]

    diagnosis: str  # raw findings grounded in both evidence sources
    priorities: str  # the 1-3 things worth focusing on, and why
    feedback: str  # final coach-voice narrative + drills
