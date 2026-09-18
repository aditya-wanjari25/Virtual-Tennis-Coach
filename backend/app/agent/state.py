"""Shared state that flows through the coaching LangGraph.

Each node function takes the current state and returns a dict of the
fields it wants to update -- LangGraph merges that into the running
state before calling the next node.
"""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    swings: list[dict]  # serialized SwingMetrics, one per detected swing
    diagnosis: str  # raw findings grounded in the metrics
    priorities: str  # the 1-3 things worth focusing on, and why
    feedback: str  # final coach-voice narrative + drills
