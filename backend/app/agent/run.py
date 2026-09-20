"""CLI: pose landmarks JSON -> swing metrics -> coaching feedback.

    uv run python -m app.agent.run path/to/landmarks.json
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe

from app.agent.analyst import analyze as analyze_swings
from app.analysis.landmarks import load_pose_sequence
from app.analysis.metrics import compute_swing_metrics
from app.analysis.phases import segment_swings

load_dotenv()


@observe(name="analyze_swing_video")
def analyze(landmarks_path: str | Path) -> str:
    """Metrics-only path -- this CLI works from a landmarks JSON, so there is no
    video to run perception on. Feedback will cover what the metrics can see."""
    pose = load_pose_sequence(landmarks_path)
    swings = segment_swings(pose, hand="right")
    swing_metrics = [asdict(compute_swing_metrics(pose, s, hand="right")) for s in swings]

    return analyze_swings(swing_metrics, observations=None)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m app.agent.run path/to/landmarks.json")
    print(analyze(sys.argv[1]))
    get_client().flush()
