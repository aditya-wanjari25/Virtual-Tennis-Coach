FROM python:3.13-slim

# libgl1/libglib2.0-0 etc are needed by opencv-python and mediapipe at
# import time (they link against libGL) -- without these the container
# crashes on the first `import cv2` with a missing-shared-library error.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # ffmpeg downscales video before it goes to Gemini. Without it the code
    # falls back to sending the original -- no crash, but a 70x larger upload
    # and a slower call, which is the kind of regression you'd never notice.
    ffmpeg \
    # curl fetches the pose model below.
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# The MediaPipe pose model is a 9.4MB binary, deliberately not committed to
# git. Fetching it at build time keeps the repo light AND keeps the build
# self-contained -- relying on the local file meant the image only worked when
# built from a working copy that happened to have it, not from a clean clone.
RUN mkdir -p cv_spike/models && \
    curl -fsSL -o cv_spike/models/pose_landmarker_full.task \
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"

# Copy dependency files first so `uv sync` is cached unless deps change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Run migrations, then start the API -- keeps the DB schema in sync with
# the deployed code on every release without a separate manual step.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
