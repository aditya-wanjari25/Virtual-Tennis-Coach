"""FastAPI app: upload a swing video, get back coaching feedback.

Processing (pose extraction + metrics + the LangGraph agent) takes real
time -- too long to block an HTTP request -- so uploads kick off a
background task and the client polls for the result. Job state lives in
Postgres so it survives restarts and works across multiple workers.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langfuse import get_client, observe
from pydantic import BaseModel, ConfigDict, Field

from app.agent.analyst import analyze as analyze_swings
from app.agent.chat import ask as chat_ask, close_pools, history as chat_history
from app.analysis.metrics import compute_swing_metrics
from app.analysis.perception import analyze_video
from app.analysis.phases import segment_swings
from app.analysis.pose_extraction import extract_pose_sequence
from app.db import get_session
from app.guardrails import (
    MAX_CHAT_CHARS,
    VideoRejected,
    check_filename,
    stream_to_disk,
    validate_video_file,
)
from app.models import JobModel

load_dotenv()

logger = logging.getLogger(__name__)


VIDEO_RETENTION_DAYS = int(os.environ.get("VIDEO_RETENTION_DAYS", "7"))


def _purge_old_videos() -> None:
    """Delete stored videos past the retention window.

    We stopped deleting videos after analysis so chat's rewatch_swing could use
    them, which left storage growing without bound. This is the replacement --
    a sweep at startup, which is enough for a single-instance deployment. Object
    storage lifecycle rules replace it when this moves to S3.
    """
    if not STORAGE_DIR.exists():
        return
    cutoff = time.time() - VIDEO_RETENTION_DAYS * 86400
    freed = 0
    for path in STORAGE_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                freed += path.stat().st_size
                path.unlink()
        except OSError:
            logger.warning("Could not purge %s", path.name, exc_info=True)
    if freed:
        logger.info("Purged %.1f MB of videos older than %d days", freed / 1e6, VIDEO_RETENTION_DAYS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _purge_old_videos()
    yield
    get_client().flush()  # make sure any in-flight traces are sent before the process exits
    close_pools()  # checkpointer connection pools


app = FastAPI(title="Virtual Tennis Coach", lifespan=lifespan)

# Which browser origins may call this API. Comma-separated env var so the
# deployed frontend's origin can be set per environment -- hardcoding localhost
# means every request from a deployed frontend fails CORS.
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = Path(__file__).parents[1] / "storage" / "videos"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class Stage(str, Enum):
    """Pipeline steps, surfaced so the client can show real progress rather
    than a spinner for the ~30s an analysis takes."""

    TRACKING = "tracking"   # pose extraction + swing segmentation
    WATCHING = "watching"   # Gemini video perception
    COACHING = "coaching"   # Claude writes the feedback


class Job(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: JobStatus
    stage: Stage | None = None
    created_at: datetime
    feedback: str | None = None
    error: str | None = None
    # Lets the UI say "3 swings analysed" without shipping the whole payload.
    swing_count: int | None = None


class UploadResponse(BaseModel):
    job_id: str


@observe(name="analyze_swing_video")
def _process_video(job_id: str, video_path: Path) -> None:
    session = get_session()
    try:
        job = session.get(JobModel, job_id)
        job.status = JobStatus.PROCESSING

        def stage(name: str) -> None:
            job.stage = name
            session.commit()

        stage(Stage.TRACKING)
        pose = extract_pose_sequence(video_path)
        swings = segment_swings(pose, hand="right")
        if not swings:
            raise ValueError("No swings detected in video")
        swing_metrics = [asdict(compute_swing_metrics(pose, s, hand="right")) for s in swings]

        # Returns None on failure rather than raising -- perception is
        # supplementary, so losing it degrades the analysis to metrics-only
        # instead of failing the job.
        stage(Stage.WATCHING)
        observations = analyze_video(video_path, pose, swings)

        stage(Stage.COACHING)
        # Persist the evidence, not just the conclusion: chat grounds follow-up
        # questions in these, and without them the feedback is unauditable.
        job.metrics = swing_metrics
        job.observations = observations
        job.feedback = analyze_swings(swing_metrics, observations)
        job.status = JobStatus.DONE
        job.stage = None
        session.commit()
    except Exception as e:
        job = session.get(JobModel, job_id)
        job.error = str(e)
        job.status = JobStatus.ERROR
        session.commit()
    finally:
        session.close()
        # The video is deliberately NOT deleted here -- chat's rewatch_swing
        # tool needs it to re-examine a specific swing. Retention is unbounded
        # for now; a lifecycle policy belongs with the move to S3.


@app.post("/videos", response_model=UploadResponse)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile) -> UploadResponse:
    # Validate before the job row and background task exist, so a bad upload
    # gets an immediate, specific error instead of becoming a job the client
    # only discovers has failed ~30s later behind a polling spinner.
    #
    # stream_to_disk also replaces the unbounded read loop that used to live
    # here: it streams in chunks (so a 100MB upload isn't a 100MB memory spike)
    # AND stops at the size cap, which the old loop had no notion of.
    video_path: Path | None = None
    try:
        suffix = check_filename(file.filename)
        job_id = str(uuid.uuid4())
        video_path = STORAGE_DIR / f"{job_id}{suffix}"
        await stream_to_disk(file, video_path)
        validate_video_file(video_path)
    except VideoRejected as e:
        # unlink is idempotent here -- stream_to_disk cleans up its own partial
        # writes, so this is for the probe rejecting a fully-written file.
        if video_path is not None:
            video_path.unlink(missing_ok=True)
        logger.info("Rejected upload (%d): %s", e.status, e.detail)
        raise HTTPException(status_code=e.status, detail=e.detail) from e

    session = get_session()
    try:
        session.add(JobModel(id=job_id, status=JobStatus.PENDING, video_key=video_path.name))
        session.commit()
    finally:
        session.close()

    background_tasks.add_task(_process_video, job_id, video_path)

    return UploadResponse(job_id=job_id)


class Swing(BaseModel):
    """One swing, as the UI needs it.

    Metrics are deliberately NOT sent as raw values. Our rotation figures are
    angles of a 2D projected shoulder line from a back view -- the absolute
    number isn't in units a player thinks in, and quoting it implies precision
    we don't have. What IS trustworthy is how a quantity compares across the
    swings in one video, so each metric ships as a 0-1 position within this
    video's own range, for relative bars rather than numbers.
    """

    index: int
    contact_time_s: float
    observations: dict[str, str] = {}
    relative: dict[str, float] = {}


# Metrics worth comparing across swings, with player-facing labels.
_COMPARABLE = {
    "max_shoulder_rotation_during_backswing_deg": "Shoulder turn",
    "stance_width_at_contact": "Stance width",
    "swing_path_width": "Swing size",
    "arm_extension_at_contact": "Arm extension",
}


def _relative(metrics: list[dict], key: str) -> list[float | None]:
    """Each swing's value as a 0-1 position within this video's own range."""
    raw = [m.get(key) for m in metrics]
    vals = [abs(v) for v in raw if isinstance(v, (int, float))]
    if len(vals) < 2:
        return [None] * len(raw)
    lo, hi = min(vals), max(vals)
    span = hi - lo
    return [
        None if not isinstance(v, (int, float)) else (0.5 if span == 0 else (abs(v) - lo) / span)
        for v in raw
    ]


@app.get("/videos/{job_id}/swings", response_model=list[Swing])
async def get_swings(job_id: str) -> list[Swing]:
    """Per-swing breakdown, so the UI can show the evidence behind the feedback."""
    session = get_session()
    try:
        job = session.get(JobModel, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        metrics, observations = job.metrics or [], job.observations or {}
    finally:
        session.close()

    obs_by_index = {s.get("swing_index"): s for s in observations.get("swings", [])}
    rel = {label: _relative(metrics, key) for key, label in _COMPARABLE.items()}

    out = []
    for i, m in enumerate(metrics):
        obs = obs_by_index.get(i + 1, {})
        out.append(
            Swing(
                index=i + 1,
                contact_time_s=m.get("contact_time_s", 0.0),
                observations={
                    k: v for k, v in obs.items()
                    if k != "swing_index" and isinstance(v, str) and v.strip()
                },
                relative={label: vals[i] for label, vals in rel.items() if vals[i] is not None},
            )
        )
    return out


@app.get("/videos/{job_id}/file")
async def get_video_file(job_id: str) -> FileResponse:
    """Serve the uploaded video back for playback.

    FileResponse handles HTTP range requests, which is what makes seeking work
    in a <video> element -- without 206 responses the browser can only play
    from the start.
    """
    session = get_session()
    try:
        job = session.get(JobModel, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        key = job.video_key
    finally:
        session.close()

    if not key:
        raise HTTPException(status_code=404, detail="No video stored for this analysis")

    # Guard against a crafted video_key escaping the storage directory.
    path = (STORAGE_DIR / key).resolve()
    if not path.is_relative_to(STORAGE_DIR.resolve()) or not path.exists():
        raise HTTPException(status_code=404, detail="Video file is no longer available")

    return FileResponse(path, media_type="video/mp4")


class ChatRequest(BaseModel):
    # Capped so a scripted request can't run up an unbounded token bill. Far
    # above anything typed into a chat box; it's a ceiling, not a budget.
    message: str = Field(min_length=1, max_length=MAX_CHAT_CHARS)


class ChatMessage(BaseModel):
    role: str
    content: str


@app.post("/videos/{job_id}/chat", response_model=ChatMessage)
async def post_chat(job_id: str, body: ChatRequest) -> ChatMessage:
    """Ask a follow-up question about a completed analysis.

    Conversation history is held by the LangGraph checkpointer, keyed on
    job_id, so the client sends only the new message.
    """
    session = get_session()
    try:
        job = session.get(JobModel, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != JobStatus.DONE:
            raise HTTPException(status_code=409, detail=f"Analysis is {job.status}, not ready for questions")
    finally:
        session.close()

    # The agent loop is blocking (LLM + tool calls); run it off the event loop
    # so it doesn't stall other requests.
    reply = await run_in_threadpool(chat_ask, job_id, body.message)
    return ChatMessage(role="assistant", content=reply)


@app.get("/videos/{job_id}/chat", response_model=list[ChatMessage])
async def get_chat(job_id: str) -> list[ChatMessage]:
    """Past conversation turns, for restoring the UI on load."""
    turns = await run_in_threadpool(chat_history, job_id)
    return [ChatMessage(**t) for t in turns]


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe for the platform. Deliberately does not touch the DB --
    it answers 'is this process up', not 'is everything healthy', so a slow
    database can't cause the platform to kill an otherwise-fine machine."""
    return {"status": "ok"}


@app.get("/videos/{job_id}", response_model=Job)
async def get_video_status(job_id: str) -> Job:
    session = get_session()
    try:
        job = session.get(JobModel, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        result = Job.model_validate(job)
        result.swing_count = len(job.metrics) if job.metrics else None
        return result
    finally:
        session.close()


# --- Static frontend -------------------------------------------------------
# Mounted LAST, on purpose. A mount at "/" matches everything, so registering
# it before the API routes above would shadow them all. In production the
# Docker build drops the compiled Vite output here; in local dev the directory
# doesn't exist and the frontend runs separately on the Vite dev server, which
# is why this is conditional rather than assumed.
STATIC_DIR = Path(__file__).parents[1] / "static"

if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    logger.info("No static/ directory — API only (frontend served by Vite in dev)")
