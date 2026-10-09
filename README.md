# Virtual Tennis Coach

Upload a video of your groundstrokes and get specific, measured coaching feedback — then ask follow-up questions about it.

A computer-vision pipeline measures your body mechanics frame by frame, a vision model watches what the measurements can't see, and a language model turns both into coaching you can act on.

**Stack** — React 19 · TypeScript · Vite · Tailwind 4 · FastAPI · Python 3.13 · MediaPipe · OpenCV · NumPy · Claude Sonnet 5 · Claude Haiku 4.5 · Gemini 3.8 Flash · LangGraph · PostgreSQL · SQLAlchemy · Alembic · Langfuse · Docker · Fly.io

---

## Features

**Pose tracking and swing detection.** MediaPipe extracts body landmarks from every frame. Swings are found from wrist-speed kinematics — one peak per swing, bounded by two valleys — so no ball tracking or trained classifier is needed.

**Measured mechanics.** Shoulder and hip rotation, hip–shoulder separation at contact, stance width, arm extension, swing path width and follow-through height. Distances are normalized by body scale, so measurements stay comparable across videos shot at different distances.

**Video perception.** Gemini watches the clip for what pose metrics structurally can't see: knee bend, balance, contact height, spacing to the ball, split-step timing, and what you do between shots.

**Coaching feedback.** Claude combines both sources — measurements lead on anything numeric, observations cover the rest — and writes one to three prioritized points, each with what it noticed, why it matters, and a drill or cue.

**Relative metrics, not raw numbers.** Each metric renders as a 0–1 bar within that video's own range. The rotation figures are 2D projected angles from a back view, so how a quantity *changes* across your swings is meaningful where the absolute value isn't.

**Video playback with contact markers.** Your clip plays back in the browser with markers at each detected contact point, alongside a per-swing breakdown.

**Follow-up chat.** A LangGraph agent answers questions about the analysis using three tools: look up a swing's metrics, look up what was observed, or re-watch a specific swing — which sends that swing's clip back to Gemini with your question attached. Conversations persist in Postgres.

**Staged progress.** Analysis runs as a background job with live stages (`tracking` → `watching` → `coaching`), so the UI shows real progress instead of a spinner.

**Input and output guardrails.** Uploads are checked for type, size, codec, duration, resolution and frame count before any work starts. Chat messages are screened by Claude Haiku, with injury and off-topic questions redirected rather than answered. Feedback is reviewed against its own evidence before it reaches you, with one repair attempt if it fails.

**Full tracing.** Every model call, tool call and guardrail decision is traced to Langfuse with token usage, so cost and behaviour are visible per request.

---

## How it works

Analysis takes tens of seconds, so an upload returns a job id immediately and the client polls for the result.

```
POST /videos                 → job_id, right away
  ├─ guardrails: type, size, duration, codec, resolution, frames
  ├─ stream to disk, INSERT jobs (status=pending)
  └─ schedule a background task

   [background]
     stage=tracking   pose extraction → swing segmentation → metrics
     stage=watching   Gemini watches the clip
     stage=coaching   Claude writes the feedback, reviewed before it ships
     status=done

GET  /videos/{id}            → client polls until done
GET  /videos/{id}/swings     → per-swing breakdown
POST /videos/{id}/chat       → follow-up questions
```

---

## Running it locally

**Prerequisites:** Docker, [uv](https://docs.astral.sh/uv/), Node 22+, and `ffmpeg` on PATH (`brew install ffmpeg`).

**1. Fetch the pose model** (9.4MB, not committed):

```bash
mkdir -p backend/cv_spike/models
curl -fsSL -o backend/cv_spike/models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
```

**2. Start Postgres:**

```bash
docker compose up -d postgres
```

**3. Backend:**

```bash
cd backend
cp .env.example .env          # fill in your API keys
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
```

**4. Frontend**, in a second shell:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** and upload a clip of a few groundstrokes filmed from behind the baseline. Five to fifteen seconds is plenty.

### Configuration

Set in `backend/.env`:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Coaching and guardrail calls |
| `GEMINI_API_KEY` | — | Video perception |
| `LANGFUSE_PUBLIC_KEY` | — | Tracing |
| `LANGFUSE_SECRET_KEY` | — | Tracing |
| `LANGFUSE_BASE_URL` | `https://us.cloud.langfuse.com` | Tracing |
| `DATABASE_URL` | — | Postgres connection string |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |
| `VIDEO_RETENTION_DAYS` | `7` | How long uploads are kept |
| `MAX_UPLOAD_MB` | `100` | Upload size cap |
| `MAX_VIDEO_DURATION_S` | `90` | Clip length cap |
| `MAX_VIDEO_FRAMES` | `3000` | Frame budget for pose extraction |
| `MAX_VIDEO_LONG_EDGE_PX` | `3840` | Resolution cap |
| `MAX_CHAT_CHARS` | `2000` | Chat message cap |

---

## Tests

```bash
cd backend && uv run pytest
```

94 tests, no database or API keys required. Test videos are generated with ffmpeg at run time.

---

## API

| Method | Path | |
|---|---|---|
| `POST` | `/videos` | Upload a clip; returns `job_id` |
| `GET` | `/videos/{id}` | Status, current stage, feedback |
| `GET` | `/videos/{id}/swings` | Per-swing breakdown |
| `GET` | `/videos/{id}/file` | The video, with range support for seeking |
| `POST` | `/videos/{id}/chat` | Ask a follow-up question |
| `GET` | `/videos/{id}/chat` | Conversation history |
| `GET` | `/healthz` | Liveness probe |

---

## Deployment

`fly.toml` deploys one app serving both the API and the built frontend, so the deployed setup is same-origin and needs no CORS. A Fly volume holds uploaded videos so they survive deploys.

```bash
fly secrets set ANTHROPIC_API_KEY=... GEMINI_API_KEY=...   # etc.
fly deploy
```
