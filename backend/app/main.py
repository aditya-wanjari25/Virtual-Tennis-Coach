"""FastAPI app: upload a swing video, get back coaching feedback.

Processing (pose extraction + metrics + the LangGraph agent) takes real
time -- too long to block an HTTP request -- so uploads kick off a
background task and the client polls for the result. Job state lives in
an in-memory dict for now; swap for Postgres once we need jobs to
survive a server restart or run across multiple workers.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from langfuse import get_client, observe
from pydantic import BaseModel

from app.agent.graph import build_graph
from app.analysis.metrics import compute_swing_metrics
from app.analysis.phases import segment_swings
from app.analysis.pose_extraction import extract_pose_sequence

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    get_client().flush()  # make sure any in-flight traces are sent before the process exits


app = FastAPI(title="Virtual Tennis Coach", lifespan=lifespan)

STORAGE_DIR = Path(__file__).parents[1] / "storage" / "videos"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class Job(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    feedback: str | None = None
    error: str | None = None


# In-memory job store: job_id -> Job. Fine for a single-process dev
# server; not safe across multiple workers/restarts.
JOBS: dict[str, Job] = {}


class UploadResponse(BaseModel):
    job_id: str


@observe(name="analyze_swing_video")
def _process_video(job_id: str, video_path: Path) -> None:
    JOBS[job_id].status = JobStatus.PROCESSING
    try:
        pose = extract_pose_sequence(video_path)
        swings = segment_swings(pose, hand="right")
        if not swings:
            raise ValueError("No swings detected in video")
        swing_metrics = [asdict(compute_swing_metrics(pose, s, hand="right")) for s in swings]

        graph = build_graph()
        result = graph.invoke({"swings": swing_metrics})

        JOBS[job_id].feedback = result["feedback"]
        JOBS[job_id].status = JobStatus.DONE
    except Exception as e:
        JOBS[job_id].error = str(e)
        JOBS[job_id].status = JobStatus.ERROR
    finally:
        video_path.unlink(missing_ok=True)


@app.post("/videos", response_model=UploadResponse)
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile) -> UploadResponse:
    job_id = str(uuid.uuid4())
    video_path = STORAGE_DIR / f"{job_id}{Path(file.filename or '').suffix or '.mp4'}"

    with open(video_path, "wb") as f:
        f.write(await file.read())

    JOBS[job_id] = Job(id=job_id, status=JobStatus.PENDING, created_at=datetime.now(timezone.utc))
    background_tasks.add_task(_process_video, job_id, video_path)

    return UploadResponse(job_id=job_id)


@app.get("/videos/{job_id}", response_model=Job)
async def get_video_status(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
