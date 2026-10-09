# Virtual Tennis Coach

Upload a video of your groundstrokes and get specific, measured coaching feedback — then ask follow-up questions about it.

A computer-vision pipeline measures your body mechanics frame by frame, a vision model watches what the measurements can't see, and a language model turns both into coaching. The two halves are deliberately separate, and so is the evidence each produces.

**Stack** — React 19 · TypeScript · Vite · Tailwind 4 · FastAPI · Python 3.13 · MediaPipe · OpenCV · NumPy · Claude Sonnet 5 · Claude Haiku 4.5 · Gemini 3.8 Flash · LangGraph · PostgreSQL · SQLAlchemy · Alembic · Langfuse · Docker · Fly.io

---

## How it works

Analysis takes tens of seconds, which is too long to hold an HTTP request open. So an upload returns a job id immediately and the client polls:

```
POST /videos                    → job_id, right away
  ├─ guardrails: extension, size, ffprobe, codec, duration, resolution, frames
  ├─ stream to disk, INSERT jobs (status=pending)
  └─ schedule a background task

   [background] _process_video(job_id, path)
     stage=tracking   MediaPipe pose extraction → swing segmentation → metrics
     stage=watching   Gemini watches the clip
     stage=coaching   Claude writes the feedback, reviewed before it ships
     status=done

GET  /videos/{id}               → client polls ~1.5s until done/error
GET  /videos/{id}/swings        → per-swing breakdown
POST /videos/{id}/chat          → follow-up questions
```

### Two models, split by what each can actually do

**Deterministic CV** owns anything numeric or cross-swing. Swing segmentation needs no ball tracking or classifier — a groundstroke's racket-hand speed dips at the top of the backswing, peaks near contact, and falls through the follow-through, so each swing is one peak in wrist speed bounded by two valleys. Distances are normalized by the median shoulder width across well-tracked frames, which keeps measurements comparable across videos shot at different distances.

**Gemini** owns everything the pose metrics are structurally blind to — knee bend, balance, contact height, spacing to the ball, split-step timing. Things that only exist in motion or in depth.

**Claude Sonnet 5** combines them, told explicitly that the two sources aren't equally reliable: metrics win on anything they measure, observations are a careful observer's report, and where they disagree that disagreement is itself a signal worth surfacing.

### Measurements are never shown as numbers

The rotation figures are angles of a 2D projected shoulder line seen from behind. "23°" is not the shoulder turn a coach means, and quoting it implies precision the pipeline doesn't have. Direction and change across swings are trustworthy; absolute values aren't. So the UI ships each metric as a 0–1 position within that video's own range — relative bars, not numbers — and the prompt forbids raw measurements in the written feedback.

### Chat

Follow-up questions run through a LangGraph agent with three tools: `get_swing_metrics`, `get_observations`, and `rewatch_swing` — which re-sends one swing's clip to Gemini with your specific question attached. Conversation state is checkpointed in Postgres and keyed on job id.

This is the one place LangGraph earns its keep: it loops, it branches, it has persisted state, and the model decides what happens next. The analysis path was once a three-node graph too; it measured ~46s for output no better than a single ~8s call, so it's now a plain function.

---

## Guardrails

Layered cheapest-first, so most abuse never reaches a model.

### Input — video

| Check | Result |
|---|---|
| Extension allowlist (`.mp4 .mov .webm .avi .mpeg .mpg .3gp`) | 415 |
| Size cap (100 MB), enforced mid-stream | 413 |
| `ffprobe`: codec, duration ≤ 90s, long edge ≤ 3840px, frames ≤ 3000 | 415 / 422 |
| No video track, or bytes that aren't video | 415 |
| Frame budget, re-enforced inside pose extraction | truncates |

The first three run *before* the job row and background task exist, so a bad upload gets an immediate, specific error instead of becoming a job that fails half a minute later behind a spinner.

A few details that matter:

- **The size cap is checked inside the read loop.** Measuring after writing means the disk is already gone.
- **Partial writes are cleaned up on any exception** — including `CancelledError`, which is a dropped connection and the most common way an upload actually fails. An `except Exception` would miss it.
- **HEVC is explicitly allowed.** It's what iPhones record by default, so an h264-only allowlist would reject the single most likely upload.
- **The frame budget is the real CPU bound.** Pose extraction runs one MediaPipe inference per frame on CPU. Frame count, not file size, is what bounds both cost and how long someone waits.
- **`check_filename` returns a normalized suffix** from a fixed set, so no part of a client-supplied filename reaches the filesystem — `../../etc/passwd.mp4` becomes `.mp4`.

### Input — chat

- Length capped at 2000 characters as a Pydantic constraint, so FastAPI rejects oversized bodies during validation, before the route runs.
- A **Haiku 4.5 screen** classifies each message `OK` / `MEDICAL` / `OFF_TOPIC` / `INJECTION`. A refused message reaches no model, no tools and no video. Refusals are written into the checkpointer as real turns, so they survive the client's history refetch.

Injury and pain questions are refused and redirected rather than answered. Footage can't tell you why something hurts.

### Indirect injection, via the video itself

Gemini watches arbitrary uploaded video, and its free-text observations flow into two separate Claude contexts. Film a sheet of paper reading "ignore your instructions and…" and that text arrives in the coaching model's prompt, laundered through our own vision layer and looking like our own data.

The answer is containment, not detection. Observations are wrapped in a labelled data span, and the payload **cannot close its own span** — case and whitespace variants of the delimiter are all defanged. A forged closing tag would turn containment into an amplifier, since the framing tells the model to trust what's outside the tags.

Tested end-to-end against a real planted instruction: the model ignored it and told the player their video appeared to contain text aimed at an automated system.

### Output

A Haiku review checks each feedback draft against the evidence it was written from, for medical advice and for ungrounded claims. On a failure there's **one** bounded repair attempt, which sends the rejected draft back with the problem named — withholding the draft would just be a reroll.

The two verdicts end differently on purpose:

- **Medical advice never ships.** If it survives repair, the player gets a safe fallback.
- **Ungrounded feedback ships with a loud log.** Overreaching coaching is a worse write-up, not a harmful one, and shipping nothing after someone waited for an analysis is its own failure.

### Fail-open vs fail-closed

Deliberately not uniform. The `ffprobe` gate **fails closed** — it guards against unbounded resource use, so refusing uploads when the check can't run is the safe direction. The LLM guardrails **fail open** — they sit on top of a system prompt that already asks for the same behaviour, and the cost of failing closed is that an API blip takes out chat for everyone. One unscreened message is a smaller harm than an outage.

---

## Observability

Langfuse traces the whole system. Every LLM call is a generation with model and token usage attached, including the guardrails' own Haiku calls — so guardrail spend is attributable rather than invisible.

A chat turn looks like this:

```
chat_turn                        (job_id in metadata)
├─ screen_chat_message           → guardrail_classify  (Haiku, tokens)
└─ LangGraph
   ├─ agent → ChatAnthropic      (Sonnet, tokens)
   ├─ tools_condition
   ├─ tools → get_swing_metrics
   └─ agent → ChatAnthropic
```

An analysis that needed a repair looks like this:

```
analyze_swing_video              (job_id in metadata)
├─ gemini_perception             (Gemini, tokens)
└─ analyze_with_review
   ├─ analyze_and_coach          ← draft 1
   ├─ review_feedback → guardrail_classify
   ├─ analyze_and_coach          ← draft 2, repaired
   └─ review_feedback → guardrail_classify
```

Guardrail decisions are recorded as **categorical scores**, so they're countable and chartable rather than buried in text:

| Score | Values |
|---|---|
| `upload_outcome` | `accepted` · `rejected` |
| `chat_screen` | `OK` · `MEDICAL` · `OFF_TOPIC` · `INJECTION` |
| `feedback_review` | `OK` · `MEDICAL` · `UNGROUNDED` |
| `analysis_outcome` | `clean` · `repaired` · `withheld` · `shipped_ungrounded` |

`analysis_outcome = withheld` is the one worth alerting on — it means a player got the fallback instead of coaching. Langfuse alerts on scores, so that needs no extra infrastructure.

Reasons travel in span metadata (`analysis_outcome_reason`, `verdict_reason`) rather than score comments, which don't persist. Tracing calls are wrapped so that an observability failure can never break the request it describes — losing a span beats losing the response.

> Langfuse ingest lags by roughly 10–30 seconds. A trace that looks missing right after a request usually isn't.

---

## Running it locally

**Prerequisites:** Docker, [uv](https://docs.astral.sh/uv/), Node 22+, and `ffmpeg` on PATH (`brew install ffmpeg`). ffmpeg is required, not optional — `ffprobe` backs the upload guardrail and fails closed without it.

```bash
# 1. Postgres
docker compose up -d postgres

# 2. Backend
cd backend
cp .env.example .env          # fill in ANTHROPIC_API_KEY, GEMINI_API_KEY, LANGFUSE_*
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000

# 3. Frontend (separate shell)
cd frontend
npm install
npm run dev                   # http://localhost:5173
```

The MediaPipe pose model isn't committed. Fetch it once:

```bash
mkdir -p backend/cv_spike/models
curl -fsSL -o backend/cv_spike/models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Coaching + guardrail calls |
| `GEMINI_API_KEY` | — | Video perception |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | — | Tracing |
| `DATABASE_URL` | — | Postgres, SQLAlchemy-flavoured |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |
| `VIDEO_RETENTION_DAYS` | `7` | Uploaded videos are swept at startup |
| `MAX_UPLOAD_MB` | `100` | |
| `MAX_VIDEO_DURATION_S` | `90` | |
| `MAX_VIDEO_FRAMES` | `3000` | The one that bounds CPU and latency |
| `MAX_VIDEO_LONG_EDGE_PX` | `3840` | |
| `MAX_CHAT_CHARS` | `2000` | |

Videos are retained after analysis so `rewatch_swing` can re-examine a swing; the retention sweep bounds the growth. **Starting the server deletes stored videos older than the window** — old jobs keep their feedback and chat history, but can no longer be re-watched.

---

## Tests

```bash
cd backend && uv run pytest
```

94 tests, no live database or API keys needed. Test videos are generated with ffmpeg at run time rather than committed — the sample clips are gitignored, so committed-fixture tests would pass locally and fail everywhere else.

The guardrail suite is **mutation-tested**: disabling each check individually (allowlist, size cap, codec, duration, frame budget, cleanup, containment defanging, screening, repair loop, outcome recording) makes tests fail. A validator that silently stops validating looks exactly like one that works, so "the tests pass" is only meaningful if they'd fail when the guardrail is broken.

The suite sets `LANGFUSE_TRACING_ENABLED=false` before any app import, so running tests doesn't file synthetic spans against real observability data.

---

## API

| Method | Path | |
|---|---|---|
| `POST` | `/videos` | Upload; returns `job_id` |
| `GET` | `/videos/{id}` | Status, stage, feedback |
| `GET` | `/videos/{id}/swings` | Per-swing breakdown with relative metrics |
| `GET` | `/videos/{id}/file` | The video, with range support for seeking |
| `POST` | `/videos/{id}/chat` | Ask a follow-up |
| `GET` | `/videos/{id}/chat` | Conversation history |
| `GET` | `/healthz` | Liveness; deliberately doesn't touch the DB |

---

## Deployment

`fly.toml` deploys one app serving both the API and the built frontend, which makes the deployed setup same-origin and removes CORS entirely rather than configuring it. A volume holds uploaded videos, since Fly machine filesystems are ephemeral and `rewatch_swing` needs them to survive a deploy.

The VM is 1GB by measurement, not guess: the 9.4MB pose model plus frame-by-frame video decoding OOMs a 512MB machine during extraction.

```bash
fly secrets set ANTHROPIC_API_KEY=... GEMINI_API_KEY=...   # etc.
fly deploy
```

---

## Known limitations

- **No authentication or rate limiting.** Job ids are unguessable UUIDs, which is capability-URL security by accident rather than by design. These two are coupled — any rate limit today would be per-IP and trivially evaded.
- **Raw measurements are only prompt-forbidden.** The output reviewer judges medical advice and groundedness, not units, so a leaked "23°" would pass review. A deterministic regex validator would close this cheaply and isn't written yet.
- **Chat replies aren't output-reviewed**, only the analysis is. Chat is interactive and a second review call per turn would be felt; the input screen covers the main vector.
- **Right-handed only** — `hand="right"` is hardcoded in the processing path.
- **Local disk storage**, not S3. Fine for one instance; a lifecycle policy belongs with the move to object storage.
- **Pose extraction is CPU-bound and much slower in deployment.** Measured ~59 fps locally on an M3 with Metal-backed GL, but `delegate=CPU` on a shared vCPU is realistically 5–15 fps. A 45-second clip is minutes, not seconds.
- **Perception failures degrade rather than fail.** If Gemini is unavailable the analysis continues metrics-only, which is the intended trade, but the feedback is quietly narrower.
