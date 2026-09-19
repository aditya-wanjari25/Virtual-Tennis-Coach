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
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

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
