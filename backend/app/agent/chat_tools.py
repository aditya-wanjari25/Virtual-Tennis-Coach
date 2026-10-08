"""Tools the chat agent can call to fetch evidence about an analysed swing.

Each tool takes `state: Annotated[dict, InjectedState]` -- LangGraph fills that
in from graph state and it is NOT part of the schema the model sees. So the
model calls `get_swing_metrics(swing_number=2)` and never knows a job_id
exists, while the tool still reads the right row.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.db import get_session
from app.models import JobModel

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).parents[2] / "storage" / "videos"


def _job(state: dict) -> JobModel | None:
    session = get_session()
    try:
        return session.get(JobModel, state["job_id"])
    finally:
        session.close()


def _stored_video(video_key: str) -> Path | None:
    """Resolve a job's video_key inside STORAGE_DIR, or None.

    video_key is server-generated today, so this isn't currently reachable --
    but main.get_video_file already guards the same join, and having one of two
    paths to the same directory unguarded is how that stops being true.
    """
    path = (STORAGE_DIR / video_key).resolve()
    if not path.is_relative_to(STORAGE_DIR.resolve()) or not path.exists():
        return None
    return path


def _nth(items: list | None, swing_number: int) -> Any | None:
    """Swing numbers are 1-based for the player; lists are 0-based."""
    if not items or swing_number < 1 or swing_number > len(items):
        return None
    return items[swing_number - 1]


@tool
def get_swing_metrics(swing_number: int, state: Annotated[dict, InjectedState]) -> str:
    """Get the precise measured metrics for one swing (1 = first swing).

    These are deterministic numbers from pose tracking: rotation angles, stance
    width, arm extension, swing path width, follow-through height. Use this when
    a question is about something measurable, or to check a claim.
    """
    job = _job(state)
    m = _nth(job.metrics if job else None, swing_number)
    if m is None:
        n = len(job.metrics) if job and job.metrics else 0
        return f"No metrics for swing {swing_number}. This video has {n} swing(s)."
    return json.dumps(m, indent=2)


@tool
def get_observations(swing_number: int, state: Annotated[dict, InjectedState]) -> str:
    """Get the visual observations recorded for one swing (1 = first swing).

    These came from a model that watched the video during the original analysis:
    knee bend, balance, timing, spacing to the ball, contact height. Use this for
    questions the metrics can't answer. If it doesn't cover what was asked, use
    rewatch_swing instead to look at the video again.
    """
    job = _job(state)
    obs = job.observations if job else None
    if not obs:
        return "No visual observations were recorded for this video."
    s = _nth(obs.get("swings"), swing_number)
    if s is None:
        return f"No observations for swing {swing_number}. This video has {len(obs.get('swings', []))} swing(s)."
    extra = {k: obs[k] for k in ("between_shots", "across_swings", "doing_well") if k in obs}
    return json.dumps({"this_swing": s, "whole_session": extra}, indent=2)


@tool
def rewatch_swing(swing_number: int, question: str, state: Annotated[dict, InjectedState]) -> str:
    """Watch one swing again to answer a specific question about it.

    Sends just that swing's clip back to the video model with your question
    attached. Use this when the recorded observations don't cover what was
    asked -- e.g. "was my elbow bent at contact?", "where were my feet
    pointing?". Slower than the other tools, so prefer them when they suffice.
    """
    from app.analysis.perception import rewatch  # local import: avoids a cycle

    job = _job(state)
    if job is None or not job.video_key or not job.metrics:
        return "The video for this session is no longer available to re-examine."

    contact = _nth([m.get("contact_time_s") for m in job.metrics], swing_number)
    if contact is None:
        return f"No swing {swing_number} in this video (it has {len(job.metrics)})."

    video_path = _stored_video(job.video_key)
    if video_path is None:
        return "The video file for this session is no longer on disk."

    answer = rewatch(video_path, contact_time_s=float(contact), question=question)
    return answer or "Could not re-examine the video just now -- the vision service failed."


CHAT_TOOLS = [get_swing_metrics, get_observations, rewatch_swing]
