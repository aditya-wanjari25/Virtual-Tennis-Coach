"""ORM models. Just one table for now: the analysis job that used to
live in the in-memory JOBS dict in main.py."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class JobModel(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String, nullable=False)
    # Which part of the pipeline is currently running. Analysis takes ~30s, so
    # the client needs something more informative than a spinner.
    stage: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The evidence behind `feedback`. Previously these were local variables in
    # _process_video and were discarded once it returned -- the analysis was
    # unauditable after the fact and chat had nothing to ground follow-up
    # questions in. JSONB rather than Text so we can query into them later.
    metrics: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    observations: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Where the uploaded video lives, relative to STORAGE_DIR. Retained after
    # analysis so chat's rewatch_swing tool can re-examine a specific swing.
    # Becomes an S3 key when storage moves off local disk.
    video_key: Mapped[str | None] = mapped_column(String, nullable=True)
