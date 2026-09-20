"""FastAPI app: upload a swing video, get back coaching feedback.

Processing (pose extraction + metrics + the LangGraph agent) takes real
time -- too long to block an HTTP request -- so uploads kick off a
background task and the client polls for the result. Job state lives in
Postgres so it survives restarts and works across multiple workers.
"""

from __future__ import annotations

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
from langfuse import get_client, observe
from pydantic import BaseModel, ConfigDict

from app.agent.analyst import analyze as analyze_swings
from app.agent.chat import ask as chat_ask, close_pools, history as chat_history
from app.analysis.metrics import compute_swing_metrics
from app.analysis.perception import analyze_video
from app.analysis.phases import segment_swings
from app.analysis.pose_extraction import extract_pose_sequence
from app.db import get_session
from app.models import JobModel

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    get_client().flush()  # make sure any in-flight traces are sent before the process exits
    close_pools()  # checkpointer connection pools


app = FastAPI(title="Virtual Tennis Coach", lifespan=lifespan)

# Dev-only: allow the Vite frontend (different origin/port) to call this
# API from the browser. Tighten to the real deployed frontend origin
# once this goes to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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


class Job(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: JobStatus
    created_at: datetime
    feedback: str | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    job_id: str


@observe(name="analyze_swing_video")
def _process_video(job_id: str, video_path: Path) -> None:
    session = get_session()
    try:
        job = session.get(JobModel, job_id)
        job.status = JobStatus.PROCESSING
        session.commit()

        pose = extract_pose_sequence(video_path)
        swings = segment_swings(pose, hand="right")
        if not swings:
            raise ValueError("No swings detected in video")
        swing_metrics = [asdict(compute_swing_metrics(pose, s, hand="right")) for s in swings]

        # Returns None on failure rather than raising -- perception is
        # supplementary, so losing it degrades the analysis to metrics-only
        # instead of failing the job.
        observations = analyze_video(video_path, pose, swings)

        # Persist the evidence, not just the conclusion: chat grounds follow-up
        # questions in these, and without them the feedback is unauditable.
        job.metrics = swing_metrics
        job.observations = observations
        job.feedback = analyze_swings(swing_metrics, observations)
        job.status = JobStatus.DONE
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
    job_id = str(uuid.uuid4())
    video_path = STORAGE_DIR / f"{job_id}{Path(file.filename or '').suffix or '.mp4'}"

    # Stream to disk in chunks. `await file.read()` pulls the whole video into
    # memory first -- a 100MB upload is a 100MB spike, per concurrent request.
    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    session = get_session()
    try:
        session.add(JobModel(id=job_id, status=JobStatus.PENDING, video_key=video_path.name))
        session.commit()
    finally:
        session.close()

    background_tasks.add_task(_process_video, job_id, video_path)

    return UploadResponse(job_id=job_id)


class ChatRequest(BaseModel):
    message: str


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


@app.get("/videos/{job_id}", response_model=Job)
async def get_video_status(job_id: str) -> Job:
    session = get_session()
    try:
        job = session.get(JobModel, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return Job.model_validate(job)
    finally:
        session.close()
