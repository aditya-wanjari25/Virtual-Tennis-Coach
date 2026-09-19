"""CLI: pose landmarks JSON -> swing metrics -> coaching feedback.

    uv run python -m app.agent.run path/to/landmarks.json
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe

from app.agent.graph import build_graph
from app.analysis.landmarks import load_pose_sequence
from app.analysis.metrics import compute_swing_metrics
from app.analysis.phases import segment_swings

load_dotenv()


@observe(name="analyze_swing_video")
def analyze(landmarks_path: str | Path) -> str:
    pose = load_pose_sequence(landmarks_path)
    swings = segment_swings(pose, hand="right")
    swing_metrics = [asdict(compute_swing_metrics(pose, s, hand="right")) for s in swings]

    graph = build_graph()
    result = graph.invoke({"swings": swing_metrics})
    return result["feedback"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m app.agent.run path/to/landmarks.json")
    print(analyze(sys.argv[1]))
    get_client().flush()
