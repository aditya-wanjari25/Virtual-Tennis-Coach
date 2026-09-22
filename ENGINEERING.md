# Virtual Tennis Coach — Engineering Walkthrough

A guided tour of how this system is built and *why* it's built that way. Written
to be read top-to-bottom by someone who wants to get better at engineering, not
as API reference. Every section ties a concrete decision in this codebase to a
transferable principle.

**The system in one sentence:** you upload a video of your tennis groundstrokes,
a computer-vision pipeline measures your body mechanics frame by frame, a vision
model watches what the measurements can't see, a language model turns both into
coaching feedback, and you can then ask it follow-up questions.

**Stack:** React 19 + TypeScript + Vite + Tailwind 4 · FastAPI + Python 3.13 ·
MediaPipe + OpenCV + NumPy · Claude Sonnet 5 · Gemini 3.8 Flash · LangGraph ·
PostgreSQL + SQLAlchemy + Alembic · Langfuse · Docker + Fly.io

---

## Table of contents

1. [System shape and the lifecycle of one upload](#part-1)
2. [The computer vision layer](#part-2)
3. [The AI layer — why two models](#part-3)
4. [The agent layer — function vs. graph](#part-4)
5. [Persistence, migrations, and state](#part-5)
6. [The frontend](#part-6)
7. [Deployment and operations](#part-7)
8. [Cross-cutting lessons](#part-8)

---

<a name="part-1"></a>
## Part 1 — System shape and the lifecycle of one upload

### The constraint that determines the architecture

Everything about this architecture falls out of one fact: **analysis takes ~30
seconds.**

That's too long for an HTTP request. Browsers, reverse proxies and Fly's edge
all have timeouts in that neighbourhood, and even if they didn't, holding a
connection open for 30s means a user on a flaky phone connection loses their
entire analysis to a dropped socket. So the first design decision is forced:
**the upload and the result must be two separate conversations.**

This is the *asynchronous job* pattern. Learn to recognise when you need it: any
time the work outlives the request.

### The lifecycle

```
POST /videos                 → returns a job_id immediately
  ├─ stream the file to disk
  ├─ INSERT into jobs (status=pending)
  └─ schedule a background task

        [background]  _process_video(job_id, path)
          stage=tracking  → MediaPipe pose extraction → segmentation → metrics
          stage=watching  → Gemini watches the video
          stage=coaching  → Claude writes the feedback
          status=done

GET  /videos/{id}            → client polls every 1.5s until done/error
GET  /videos/{id}/swings     → the per-swing evidence table
GET  /videos/{id}/file       → the video back, for playback
POST /videos/{id}/chat       → follow-up Q&A grounded in the stored evidence
GET  /videos/{id}/chat       → transcript, for restoring the UI
GET  /healthz                → liveness probe
```

### Decision 1 — stream the upload, never read it whole

`backend/app/main.py`:

```python
with open(video_path, "wb") as f:
    while chunk := await file.read(1024 * 1024):
        f.write(chunk)
```

The obvious version is `content = await file.read()` — one line. The problem is
that a 100MB phone video becomes a 100MB memory spike *per concurrent upload*.
Three simultaneous uploads on a 1GB machine and you OOM. Chunking caps memory at
1MB per request regardless of file size.

> **Principle.** When the size of some data is controlled by the user, never
> hold all of it in memory at once. Stream it.

### Decision 2 — job state lives in Postgres, not in a dict

The module docstring records that this used to be an in-memory `JOBS` dict. That
works perfectly until one of three things happens:

- the process restarts → every in-flight job vanishes
- you run more than one worker → a poll hits worker B, which has never heard of
  a job created on worker A
- you deploy → same as a restart

Moving job state to Postgres makes the API server **stateless**, which is the
property that lets you scale horizontally, restart freely, and deploy without
losing work.

> **Principle.** "Stateless" doesn't mean the system has no state — it means the
> state isn't *in the process*. The same idea drives the LangGraph checkpointer
> in Part 4.

### Decision 3 — the `stage` field exists for a human reason

`Stage.TRACKING / WATCHING / COACHING` is not needed by any machine. It exists
because 30 seconds of undifferentiated spinner feels broken, while 30 seconds of
"Tracking your body → Watching your swings → Writing your feedback" feels like
progress. The background task commits a `stage` update at each step; the
client's 1.5s poll picks it up.

> **Principle.** Perceived latency is a real engineering property, not a design
> afterthought. You didn't make it faster — you made the existing 30s legible.
> That's usually cheaper and often just as effective.

### The failure model — two classes of dependency

Look at how `_process_video` handles errors. There are two distinct classes,
treated deliberately differently:

| Failure | Behaviour | Why |
|---|---|---|
| No swings detected | raises → `status=error` | The result would be meaningless. Fail loudly. |
| Gemini perception fails | returns `None` → continues | Supplementary signal. Degrade, don't fail. |

That second one is **graceful degradation**, and it required an explicit
judgement call: perception is the nice-to-have half, metrics are the
load-bearing half. `analyze_video()` catches everything and returns `None`, and
`analyst._evidence()` has a branch for the `None` case that tells the model
perception is unavailable and to not speculate about what it can't see.

> **Principle.** Classify every external dependency as load-bearing or
> supplementary, and encode that in the code. Most systems fail badly because
> nobody did this and *everything* is implicitly load-bearing.

### Path traversal — a security note worth internalising

`GET /videos/{job_id}/file` reads a filename out of the database and joins it to
a directory. That is exactly the shape of a **path traversal** vulnerability: if
`video_key` were ever `../../etc/passwd`, you'd serve it.

```python
path = (STORAGE_DIR / key).resolve()
if not path.is_relative_to(STORAGE_DIR.resolve()) or not path.exists():
    raise HTTPException(status_code=404, ...)
```

`resolve()` collapses `..` segments into a real absolute path, and
`is_relative_to` then asserts containment. Here the key is a server-generated
UUID so it's safe by construction — but "safe by construction" is an invariant
someone can break in a later refactor, whereas the check is enforced.

> **Principle.** Validate at the boundary where the dangerous operation happens,
> not only where the value originates. Defence in depth means not relying on an
> upstream guarantee you can't see from here.

### The structural weak point — be honest about it

`BackgroundTasks` is FastAPI's in-process background runner. It's why there's no
Celery/RQ/Arq in the dependency list, and for a single-machine personal project
that's the right call — a task queue is a whole extra component (broker,
workers, monitoring, a second deploy target).

But be clear about what was traded: **if the process dies mid-analysis, that job
is stranded at `status=processing` forever.** Nothing retries it, nothing marks
it failed. The DB row survives (the win over the dict) but no one ever finishes
it. A second consequence: `BackgroundTasks` runs on the same machine that must
also serve HTTP, so a CPU-bound pose extraction competes with request handling.

Two fixes, in increasing cost:

1. **Cheap:** a startup sweep that marks any `processing` job older than N
   minutes as `error` — the same shape as the existing `_purge_old_videos()`.
2. **Proper:** a real queue with at-least-once delivery and a visibility
   timeout, and separate worker processes.

> **Principle.** Know the failure mode of the shortcut you took. A shortcut you
> understand is engineering; a shortcut you've forgotten is debt.

---

<a name="part-2"></a>
## Part 2 — The computer vision layer

This is the most interesting engineering in the project, because it finds tennis
swings in a video **using no machine learning at all** beyond off-the-shelf pose
detection. Everything downstream is kinematics and signal processing.

Three files, a clean pipeline, each stage with one job:

```
video file
  → pose_extraction.py   per-frame joint coordinates   (MediaPipe)
  → landmarks.py         reshape into numpy arrays     (data structure)
  → phases.py            find the swings               (signal processing)
  → metrics.py           measure each swing            (geometry)
```

### Stage 1 — pose extraction

`extract_pose_frames()` opens the video with OpenCV, and for each frame runs
MediaPipe's `PoseLandmarker`, which returns 33 named body landmarks as
**normalised** coordinates (0–1 fractions of frame width/height) plus a
visibility confidence per landmark.

Three details worth noticing:

**`RunningMode.VIDEO`, not `IMAGE`.** Video mode lets the tracker use temporal
context — it knows where the wrist was last frame, so it tracks rather than
re-detecting from scratch. Cheaper and much smoother. This is why the code
computes and passes `timestamp_ms` rather than just looping.

**Coordinates are normalised, not pixels.** A 4K video and a 720p video of the
same swing produce the same numbers. Resolution independence for free.

**Output is JSON-serialisable dicts.** `extract_pose_frames` returns plain
dicts; `extract_pose_sequence` wraps it and converts to the analysis structure.
That separation is what let the CV spike dump landmarks to a file, iterate on
metrics offline against a fixed input, and never re-run the slow extraction.

> **Principle.** Put a serialisable boundary between an expensive stage and the
> logic that consumes it. It converts a 30-second feedback loop into a
> 30-millisecond one, which is the difference between iterating and not.

### Stage 2 — the data structure

`PoseSequence` transposes the data. Instead of *a list of frames, each holding
all landmarks*, you get *a dict of landmarks, each holding a `(n_frames, 2)`
numpy array*:

```python
pose.xy["right_wrist"]          # (n_frames, 2) — the whole trajectory
pose.visibility["right_wrist"]  # (n_frames,)   — confidence over time
pose.timestamps_s               # (n_frames,)
pose.fps
```

This is the **struct-of-arrays vs. array-of-structs** trade-off. The question to
ask is: *what does the downstream code iterate over?* Every analysis here asks
"how did this one joint move over time", so laying out one joint contiguously is
exactly right — and it means whole-trajectory operations become single
vectorised numpy expressions rather than Python loops.

Note also `fps = 1.0 / np.median(np.diff(timestamps_s))` — the *median* gap, not
the mean. A dropped frame creates one huge gap; a mean is dragged by it, a
median shrugs. Robust statistics, used deliberately.

And a documented limitation: frames with no detection are **dropped**, not
interpolated. The docstring says so. This means frame indices in the analysis
are indices into *detected* frames, not into the video. Fine here — but the kind
of off-by-a-frame-set assumption that causes bugs if it's implicit.

### Stage 3 — finding the swings (the clever bit)

**The insight:** you don't need to see the ball or the racket. A groundstroke has
an unmistakable *speed signature* in the racket hand:

- slow at the top of the backswing (a momentary pause as direction reverses)
- fast, peaking at contact
- slowing through the follow-through

So each swing is **one peak in wrist speed, bounded by two valleys.** Find the
peaks and you've found the swings. No ball tracking, no trained classifier, no
labelled data.

```python
wrist_xy  = pose.xy[f"{hand}_wrist"]
raw_speed = np.linalg.norm(np.diff(wrist_xy, axis=0), axis=1) * pose.fps
```

`np.diff` gives per-frame displacement vectors; `norm` gives magnitude; ×fps
converts per-frame to per-second. One line, fully vectorised.

Then three parameters, each of which fixes a specific real failure:

**Smoothing (0.25s window).** A single swing's speed profile is often *bimodal*
— a shoulder-driven acceleration, then a distinct wrist-snap pulse near contact.
Unsmoothed, that's two peaks, i.e. one swing counted as two. A ~0.25s moving
average merges them while still resolving genuinely distinct swings.

**Minimum separation (1.2s).** Peaks are selected greedily, strongest first, and
a peak is rejected if it's within 1.2s of an already-accepted one. That's
domain knowledge: humans don't hit two groundstrokes 0.4s apart.

**Minimum height (50% of max speed).** Only peaks at least half the fastest
motion in the clip count. This filters warm-up movement, walking, and adjusting
your grip. Note it's **relative to the clip's own maximum**, not an absolute
threshold — so it adapts to a hard hitter and a gentle one alike.

Then valley-finding, and here's a lesson:

```python
def _lowest_point_between(signal, start, end):
    lo, hi = min(start, end), max(start, end)
    return lo + int(np.argmin(signal[lo:hi + 1]))
```

The docstring records the bug this replaced. The intuitive approach — "walk
backwards from the peak until speed stops decreasing" — is a *local* search, and
it gets trapped in the shallow notch between the two bumps of a bimodal swing.
Taking the **global minimum over the whole inter-peak window** is immune to that.

> **Principle.** Local greedy searches get stuck in local minima. When the
> search window is small and bounded, just take the global optimum over it —
> it's both simpler and more robust. Also: that bug is now documented in the
> function that fixes it, so the next person doesn't "simplify" it back.

### Stage 4 — turning positions into measurements

`metrics.py` computes the numbers. Four ideas here, each generalisable.

**(a) Normalise against something stable.** Every distance is divided by
shoulder width, so a player filmed from 3m and one filmed from 10m produce
comparable numbers. But *which* shoulder width?

```python
def _reference_shoulder_width(pose, min_visibility=0.7):
    vis_ok = (pose.visibility["left_shoulder"] >= min_visibility) & ...
    widths = np.linalg.norm(pose.xy["left_shoulder"][vis_ok] - ...)
    return float(np.median(widths))
```

Not the width at the frame you're measuring — as the torso rotates through the
swing, the shoulder line foreshortens in 2D and can shrink toward zero right at
contact. Dividing by a near-zero number at exactly the moment you care most is a
catastrophic failure. So: **median across all well-tracked frames in the video**.
Filtered by confidence, then median for outlier resistance.

**(b) Detect when your own measurement is unreliable, and say so.**

```python
min_reliable_line_length = 0.4 * shoulder_width

def rotation_deg(frame):
    left, right = pose.xy["left_shoulder"][frame], pose.xy["right_shoulder"][frame]
    if _dist(left, right) < min_reliable_line_length:
        return None
    return _angle_deg(left, right)
```

When the torso is nearly edge-on, the two shoulder points nearly coincide, and
`arctan2` on two nearly-identical points amplifies a pixel of jitter into a huge
angle swing. The code detects that geometry and returns `None` rather than a
confident garbage number — and appends a human-readable explanation to `notes`,
which is passed to the LLM, which is instructed never to speculate about nulls.

> **Principle.** A measurement system should know the conditions under which it
> is invalid, and report *no answer* rather than a wrong one. Propagating "I
> don't know" through the whole stack — metric → JSON → prompt → output — is
> what makes the final feedback trustworthy.

**(c) Clamp windows with domain knowledge.** The valleys from `phases.py` are
good for *locating* a swing, but between shots in a rally that window also
contains recovery footwork. A real backswing-to-contact takes ~0.3–0.6s, so
metrics clamp to `max_backswing_s=0.6` and `max_follow_through_s=0.5` around
contact. Two different windows for two different purposes — detection wants
generosity, measurement wants precision.

**(d) Encode the coordinate system in a comment.**

```python
# image y grows downward, so a negative value means the wrist finished above the shoulder
finish_height = float(finish_wrist_y - finish_shoulder_y) / shoulder_width
```

Screen coordinates have y increasing downward. This is the single most common
source of sign errors in vision code. One comment at the point of use.

### What this layer buys you

Everything here is **deterministic**: same video in, same numbers out, every
time. That's the whole reason it exists alongside an LLM. It gives the system a
layer that can be tested, regression-checked, reasoned about, and *trusted* —
which is exactly what Part 3 builds on.

---

<a name="part-3"></a>
## Part 3 — The AI layer, and why there are two models

### The problem: neither model can do the job alone

| | Pose metrics | Claude | Gemini |
|---|---|---|---|
| Precise angles/distances | ✅ exact | ❌ | ❌ |
| Reproducible | ✅ | ❌ | ❌ |
| Knee bend, balance, depth | ❌ blind | ❌ | ✅ |
| Timing, rhythm, footwork between shots | ❌ | ❌ | ✅ |
| Judgement, prioritisation, coaching prose | ❌ | ✅ | ~ |
| Accepts video input | — | ❌ | ✅ |

The pose pipeline measures a projected 2D skeleton. It structurally *cannot*
see depth or quality: whether the knees are loaded, whether the player is
balanced, whether contact was rushed. Claude — the strongest reasoner here —
cannot take video at all. Gemini can watch video but shouldn't be trusted to
produce precise numbers.

So the architecture is a straightforward reading of each component's real
capability:

```
video ──┬─→ pose pipeline ──→ metrics (exact, reproducible) ──┐
        │                                                      ├─→ Claude ─→ feedback
        └─→ Gemini ────────→ observations (perceptual)  ───────┘
```

> **Principle.** When you have several models, don't pick a winner — decompose
> the task by what each one is actually *good at*, and give each the job it
> can't be beaten at.

### The division of labour is enforced in the prompt

This is the most important idea in the AI layer. It isn't enough to *hope* the
models stay in their lanes; the system prompt in `perception.py` tells Gemini
exactly what's already covered:

> You are the PERCEPTION half of a two-part system. A separate deterministic
> pose-tracking pipeline already measures, reliably and reproducibly: [list].
> **Do not report those.** They are covered, and measured more precisely than
> you can see them. Your job is everything that pipeline is blind to […]

Then the matching half of the contract, in `prompts.py`, tells Claude how to
weigh the two sources:

- For anything the metrics measure, **the metrics win.** A visual impression
  never overrides a number.
- For anything they don't cover, observations are the only source — use them,
  but don't inflate them into precision they don't have.
- **If the two disagree, say so rather than smoothing it over.** A conflict
  means either the tracking is wrong or the observer is confabulating, and
  either is worth knowing.
- Prefer synthesis: an observation may *explain* an anomaly in the metrics.

> **Principle.** In a multi-model system, the contract between components is the
> prompt. Write it explicitly, on both sides, the way you'd write an interface.
> Overlapping responsibility between two non-deterministic components is where
> hallucinated consensus comes from.

That last bullet — surfacing disagreement instead of averaging it — is an
unusually good instinct. Two independent estimates that conflict carry
information; blending them destroys it.

### Structured output, and what deliberately isn't structured

Gemini is called with a JSON Schema (`RESPONSE_SCHEMA`), so the result parses
reliably instead of needing to be scraped out of prose. But look at what the
fields *are*: `lower_body`, `balance`, `timing_and_preparation`,
`spacing_to_ball`, `contact_point` — all **free-text strings**, not scores.

The comment explains: these are qualitative perceptions feeding a reasoning
step, and forcing them onto a 1–10 scale would invent precision that isn't
there. A "6/10 balance" looks rigorous and means nothing.

And there's a `not_visible` field, present on every swing, marked required. It
exists to give the model **somewhere to put uncertainty other than into a
confident-sounding observation.** If the only available fields demand an
answer, a model will produce one. Give it a legitimate place to say "couldn't
tell" and you get honest output.

> **Principle.** Structure the *shape* of a model's output, not the *certainty*
> of it. And always provide an explicit channel for "I don't know", or you'll
> get confabulation by construction.

### Aligning the two models' indexing

A subtle but essential detail in `analyze_video()`: the CV-derived contact times
are injected into Gemini's prompt.

```python
prompt = (
    f"The clip contains {len(contact_times)} groundstrokes. Ball contact happens at "
    f"{', '.join(f'{t:.2f}s' for t in contact_times)}. "
    f"Number them 1 to {len(contact_times)} in that order and analyze each. ..."
)
```

Without this, Gemini would count swings itself and might disagree with the
metrics pipeline about how many there were or where they started. Then "swing 2"
in the observations and "swing 2" in the metrics would be *different swings*,
and every downstream join — the LLM's synthesis, the UI's breakdown, the chat
tools — would be silently wrong.

> **Principle.** When two independent components produce records that will later
> be joined, one of them must own the key and tell the other. Don't let both
> derive it and hope they agree.

### Empirical tuning, recorded in the code

This is the part most projects skip, and it's the most valuable thing here.

**Frame rate and windowing** (`perception.py`):

```
whole clip @ 1 FPS    1.4k tokens, 14s  — misjudged fast motion at contact
clipped   @ 10 FPS     13k tokens, 31s  — missed the split-step entirely,
                                          because clipping cuts out the
                                          between-shot window
whole clip @ 6 FPS      8k tokens, 17s  — caught both, cheaper and faster
```

Conclusion recorded in one line: *frame rate is what mattered; windowing
actively hurt.* The intuitive optimisation — "only send the parts with swings in
them" — was **worse**, because the between-shot footage is itself signal.

**Downscaling before upload** (`_downscaled`): re-encode to 640px tall with
ffmpeg first. Measured: 6.3MB → 0.09MB, call 13.5s → 10.3s, for 0.8s of encode
cost, with no loss in observation quality — because Gemini downsamples
internally anyway. A ~70× bandwidth reduction for free.

And it **falls back to the original file** if ffmpeg is missing or fails. Which
is exactly why the Dockerfile installs ffmpeg with a comment saying so: without
it nothing crashes, you just silently send 70× more data. The most dangerous
class of bug is the one that doesn't announce itself.

> **Principle.** Measure before you optimise, and then *write the measurement
> down next to the code*. "6 FPS" with no context is a magic number someone will
> change; "6 FPS, here are the three configurations I tried and what each cost"
> is an engineering decision that survives.

### The single-call collapse

`analyst.py` and `prompts.py` both record the same finding, and it's the most
counter-cultural decision in the project:

> This was a three-node LangGraph (diagnose → prioritize → synthesize); measured
> at ~46s for output no better than a single ~8s call. The chain was a linear
> sequence with no branching, no loops, no tool use and no persisted state —
> i.e. a function call chain, which is what it is now.

And the reasoning about *why*:

> The decomposition was a pattern for weaker models; Sonnet 5 does it internally
> via adaptive thinking, and forcing it through two English handoffs lost more
> than it gained.

Both halves matter. **The engineering half:** if your "graph" has no branches, no
cycles, no tools and no persisted state, it is a function, and dressing it as a
graph buys you framework overhead and nothing else. **The AI half:** prompt-chain
decomposition existed because early models couldn't hold a multi-step reasoning
task in one pass. Modern reasoning models can, and each handoff between steps is
lossy — you're forcing rich internal state to round-trip through a paragraph of
English. Every intermediate step is also a serial latency cost.

> **Principle.** Frameworks earn their place by the features you actually use.
> Re-examine patterns you adopted under older constraints — a technique that was
> essential two model generations ago can be actively harmful now.

### Observability is wired in, not bolted on

Langfuse `@observe` decorators wrap every model call, and each one manually
reports model, input, output and token usage.

```python
get_client().update_current_generation(
    model=MODEL,
    input=[...],
    output=observations,
    usage_details={"input": usage.total_input_tokens, "output": usage.total_output_tokens},
    metadata={"fps": FPS, "resolution": "high"},
)
```

Two subtleties. First, the manual reporting is necessary because Langfuse can't
auto-instrument a non-Anthropic SDK. Second — and this is the good bit — **the
video is deliberately omitted from the traced input.** 6MB of base64 in every
trace is useless and expensive. What's traced instead is the prompt plus
`metadata={"fps", "resolution"}`: the *parameters* of the call, which is what
you'd actually want when debugging a bad result.

`lifespan` calls `get_client().flush()` on shutdown, because traces are batched
and buffered — without an explicit flush, the last traces before a deploy are
lost, which is exactly when you most want them.

Note also the `@observe(name="analyze_swing_video")` on `_process_video` itself.
That makes the whole pipeline one **trace**, with the Gemini and Claude calls as
nested **spans** inside it. That parent-child structure is what lets you ask
"which stage was slow for this specific upload", rather than staring at two
unrelated call logs.

> **Principle.** With non-deterministic components, tracing isn't optional —
> it's the only way to answer "why did it say that?" after the fact. Trace the
> *decision-relevant* inputs, not the raw payload, and keep the span hierarchy
> matching the logical pipeline.

### Client reuse

Both `analyst._client()` and `perception._client()` are `@lru_cache(maxsize=1)`.
The comments say why: constructing a client per call meant a fresh TLS handshake
every time. Both SDK clients are thread-safe and designed to be reused, and
`lru_cache` on a zero-argument function is the smallest possible way to express
"one lazily-created singleton" in Python — lazily, so `os.environ` is read after
`load_dotenv()` rather than at import time.

> **Principle.** Objects that own a connection pool (HTTP clients, DB engines)
> should be created once per process, not once per call. Watch for this
> everywhere; it's one of the most common silent performance bugs.

---

<a name="part-4"></a>
## Part 4 — The agent layer: when a function, when a graph

The project uses LangGraph for **one** of its two AI paths and a plain function
for the other. Understanding why is the best available lesson in choosing a
framework.

### The analysis path is a function

`analyst.analyze()` is: build a string, call Claude once, extract text, log,
return. That's it. As covered in Part 3, it *used* to be a three-node graph and
was measured to be slower with no quality gain.

### The chat path is genuinely a graph

`chat.py` opens with the justification, and each line maps to a LangGraph
feature that is actually used:

```
START → agent ──(tool calls?)──→ tools ─┐
          ↑                             │
          └─────────────────────────────┘
          │
          └──(no)──→ END
```

- **LOOPS** — agent → tools → agent → … until the model stops requesting tools.
  A plain function can't express "repeat an unknown number of times, decided by
  the model."
- **BRANCHES** — a conditional edge picks tool-call vs. done, per turn.
- **PERSISTENT STATE** — the conversation survives between HTTP requests,
  checkpointed in Postgres, keyed by `thread_id`.
- **MODEL-DRIVEN CONTROL FLOW** — the model decides what happens next, not the
  code.

> **The test.** Does the work have cycles, conditional branching, tool use, or
> state that must outlive the call? If none of those: it's a function. If
> several: it's a graph. Don't let the framework decide the shape of your
> problem.

### Reducers: how state accumulates

```python
class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    job_id: str
```

`Annotated[..., add_messages]` attaches a **reducer**. Without it, a node
returning `{"messages": [x]}` would *replace* the list. With it, LangGraph
*appends*. That means each node returns only what it newly produced, and the
framework handles accumulation.

This is the same idea as a Redux reducer or a fold: define how to combine old
state with an update, once, at the schema level, rather than making every
producer responsible for reconstructing the whole.

### Dependency injection into tools

The nicest piece of design in this file:

```python
@tool
def get_swing_metrics(swing_number: int, state: Annotated[dict, InjectedState]) -> str:
    ...
```

`InjectedState` is filled in by LangGraph from graph state and is **excluded
from the schema the model sees**. So:

- The model sees a tool with exactly one parameter: `swing_number`.
- The tool body reads `state["job_id"]` and fetches the right database row.
- The model never knows a `job_id` exists and *cannot* pass a different one.

That's not just ergonomics — it's a **security boundary**. If `job_id` were a
model-visible parameter, a prompt injection in the video's observations could
plausibly steer the model into reading another user's analysis. Making it
uninjectable removes the class of attack rather than defending against it.

> **Principle.** Never expose an identifier to a model that the model shouldn't
> be able to choose. Bind authority in the code path, not in the prompt.

### The tools are a retrieval hierarchy

```python
CHAT_TOOLS = [get_swing_metrics, get_observations, rewatch_swing]
```

Three tools, in increasing cost, and the docstrings tell the model the ordering:

| Tool | Cost | Source |
|---|---|---|
| `get_swing_metrics` | a DB read | stored metrics |
| `get_observations` | a DB read | stored observations |
| `rewatch_swing` | a fresh Gemini video call | the video itself |

`rewatch_swing`'s docstring literally says *"Slower than the other tools, so
prefer them when they suffice."* Because tool docstrings are the model's API
documentation, cost guidance placed there is how you get sensible tool choice
without hard-coding a policy.

And `rewatch_swing` is the feature that makes the whole thing feel alive: the
coach can **go back and look again** at a specific swing to answer a question
that wasn't anticipated during the original analysis. It clips to just that
swing (`pre_s=1.2`, `post_s=0.9` around contact) — unlike the whole-clip
analysis pass, here you already know which swing is at issue, so extra frames
are pure cost. That's the same windowing that *hurt* in the analysis pass,
correctly applied in the one case where it helps.

Every tool also degrades to a **plain English sentence** on failure — "The video
file for this session is no longer on disk." — rather than raising. A tool that
raises breaks the agent loop; a tool that returns an explanation lets the model
recover and tell the user something useful.

> **Principle.** Tool docstrings are prompt engineering. Tool return values on
> the error path are too.

### Persistence: the checkpointer

```python
@lru_cache(maxsize=1)
def _checkpointer() -> PostgresSaver:
    dsn = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://")
    pool = ConnectionPool(conninfo=dsn, max_size=5, kwargs={"autocommit": True})
    saver = PostgresSaver(pool)
    saver.setup()
    _pools.append(pool)
    return saver
```

Because state is checkpointed under `thread_id = job_id`, the HTTP layer is
beautifully thin: `POST /chat` sends **only the new message**, and history is
loaded and saved by the framework. The client holds no transcript; a page
refresh or a different device resumes the same conversation.

Three real-world details the comments preserve:

1. **`from_conn_string` is a context manager** — designed for a scoped block,
   not for an object that must outlive a request. So the code owns the pool and
   constructs `PostgresSaver` directly.
2. **The DSN needs rewriting.** `DATABASE_URL` is SQLAlchemy-flavoured
   (`postgresql+psycopg://`); psycopg wants a raw libpq URL. Two libraries, two
   dialects of the same string.
3. **Pools must be closed.** `_pools` + `close_pools()`, called from `lifespan`,
   exist because psycopg_pool otherwise logs *"couldn't stop thread … within 5.0
   seconds"* at exit. Small, but it's the difference between a clean shutdown
   and a container that takes 5s longer to stop on every deploy.

### Content blocks, and not leaking reasoning

```python
def _text_of(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )
```

Sonnet 5 runs adaptive thinking by default, so a reply often arrives as
`[{'type': 'thinking', ...}, {'type': 'text', ...}]` rather than a plain string.
Returning that raw would both fail response validation *and* leak the model's
reasoning to the user. This helper is used in both `ask()` and `history()`.

`history()` additionally filters to `human` / `ai` messages, dropping tool calls
and tool results — those are machinery, not conversation.

> **Principle.** There is a difference between a model's *internal* output and
> its *user-facing* output. Have exactly one function that makes that
> distinction, and route everything through it.

### Blocking work off the event loop

```python
reply = await run_in_threadpool(chat_ask, job_id, body.message)
```

FastAPI runs on an async event loop, which is single-threaded. A synchronous
call that blocks — and the agent loop makes several blocking LLM and DB calls —
would stall *every* other request in the process for its whole duration.
`run_in_threadpool` moves it to a worker thread so the loop stays free.

> **Principle.** In an async server, blocking the event loop is a
> whole-application outage, not a slow endpoint. Any sync call that touches the
> network or the disk needs a thread.

### The system prompt is prepended, not stored

```python
response = model.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
```

The comment explains: keeping the system prompt out of checkpointed state means
editing it doesn't require rewriting every stored conversation. Old
conversations immediately pick up the new prompt.

That's a schema-migration decision in disguise. **Persist the data; recompute
the configuration.** Anything you bake into persisted state, you must later
migrate.

---

<a name="part-5"></a>
## Part 5 — Persistence, migrations, and the state model

### One table, and what each column is for

```python
class JobModel(Base):
    __tablename__ = "jobs"
    id:           Mapped[str]              # uuid, primary key
    status:       Mapped[str]              # pending | processing | done | error
    stage:        Mapped[str | None]       # tracking | watching | coaching
    created_at:   Mapped[datetime]         # timezone-aware
    feedback:     Mapped[str | None]       # the coaching prose
    error:        Mapped[str | None]
    metrics:      Mapped[list | None]      # JSONB — the evidence
    observations: Mapped[dict | None]      # JSONB — the evidence
    video_key:    Mapped[str | None]
```

The comments record that `metrics` and `observations` were once local variables
in `_process_video`, discarded when it returned. Persisting them changed what
the system *is*, in two ways:

1. **The analysis became auditable.** You can go back and see the evidence
   behind a conclusion.
2. **Chat became possible.** Without stored evidence, follow-up questions would
   have nothing to ground in, and the model would have to invent answers.

> **Principle.** Store the evidence, not just the conclusion. Anything derived
> from an expensive or non-reproducible process should be persisted alongside
> its output — you will want it later, and by then you can't recompute it.

### Why JSONB and not more tables

`metrics` is a list of ~10 fields per swing; `observations` is a nested object.
You could normalise that into `swings` and `observations` tables. JSONB was
chosen instead, and that's right here because:

- the shape is still changing as metrics get added — a schema change per metric
  would be painful
- nothing queries *into* these fields yet; they're read whole, by job id
- the structure is genuinely document-shaped

`JSONB` (not `JSON`) is the right variant: it's stored parsed and binary, so it
supports indexing and containment operators if you later need `WHERE metrics @>
...`. The column comment says exactly this — "JSONB rather than Text so we can
query into them later."

> **Principle.** Normalise what you query and join on. Keep document-shaped,
> read-whole, still-evolving data as JSONB. It's a reversible decision; picking
> JSONB now doesn't stop you extracting columns later.

### Timezone-aware timestamps, always

```python
created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
)
```

`DateTime(timezone=True)` plus `datetime.now(timezone.utc)`. Naive datetimes are
a category of bug that reliably ships to production and surfaces at a timezone
boundary. Store UTC, store the offset, convert at the edges only.

Note `default=lambda: ...` rather than `default=datetime.now(timezone.utc)` —
the latter evaluates **once, at import time**, and would stamp every row with
the moment the process started. A classic Python mutable/evaluated-default trap.

### The engine singleton

```python
@lru_cache
def get_engine():
    return create_engine(os.environ["DATABASE_URL"])
```

Same pattern as the API clients, for the same reason plus one more: a SQLAlchemy
`Engine` owns a **connection pool**. Creating one per request means a new TCP
connection and auth handshake per request, and defeats pooling entirely. Lazy
via `lru_cache` so `DATABASE_URL` is read on first use, after `load_dotenv()`,
regardless of import order.

The session pattern throughout is explicit:

```python
session = get_session()
try:
    ...
finally:
    session.close()
```

Verbose but correct: a leaked session holds a pooled connection, and a pool of
leaked connections is an outage. (FastAPI's `Depends` with a yield-dependency is
the more idiomatic form and would remove the repetition — a worthwhile future
refactor.)

### Migrations, and a real production bug

Alembic keeps the schema in version control: three migrations, applied
automatically at container start (`alembic upgrade head && uvicorn ...`).

The interesting one is the third. Alembic's `--autogenerate` diffs your ORM
models against the live database and writes the delta. But **LangGraph's
`PostgresSaver.setup()` creates its own tables at runtime** — `checkpoints`,
`checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`. Alembic sees
tables that aren't in `Base.metadata`, concludes they're stale, and emits
`DROP TABLE` for them.

That's catastrophic twice over: it would delete every stored conversation, and —
as the comment records — the generated migration *crash-looped on any database
where those tables didn't exist yet*, because you can't drop what isn't there.
Since migrations run at container start, that's a boot loop.

The fix is a filter hook in `alembic/env.py`:

```python
EXTERNALLY_MANAGED_TABLES = {
    "checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations",
}

def include_object(object, name, type_, reflected, compare_to):
    if type_ == "table" and name in EXTERNALLY_MANAGED_TABLES:
        return False
    if type_ == "index" and getattr(object, "table", None) is not None:
        return object.table.name not in EXTERNALLY_MANAGED_TABLES
    return True
```

Note it filters **indexes too** — otherwise autogenerate would still try to drop
indexes belonging to those tables. And it's symmetric ("in both directions"), so
Alembic neither creates nor drops them.

> **Principle.** When two systems write to one database, you must explicitly
> partition ownership. Autogenerate is a *suggestion engine*, not an authority —
> always read the migration it produces before committing it.

### There are three storage systems, and they can disagree

Worth naming plainly:

| Store | Holds | Lifetime |
|---|---|---|
| Postgres `jobs` | job state, metrics, observations, feedback | forever |
| Postgres `checkpoints` (LangGraph) | chat transcripts | forever |
| Filesystem `storage/videos/` | the uploaded video | 7 days |

The video is retained *specifically* so `rewatch_swing` can work — that's a
feature requirement driving a storage decision. But retention differs from the
DB rows, so an old job will have metrics and a transcript and **no video**.
`rewatch_swing` handles exactly that case with a plain-English return value.

> **Principle.** When components of one logical record have different
> lifetimes, every reader must handle the partially-expired case. Write that
> down, and test the path.

---

<a name="part-6"></a>
## Part 6 — The frontend

React 19, TypeScript, Vite, Tailwind 4, TanStack Query. Five components, ~650
lines total. Small, but several decisions are worth studying.

### Server state vs. client state

The single most important frontend idea here:

```typescript
const [jobId, setJobId] = useState<string | null>(null)   // client state

const jobQuery = useQuery({                               // server state
  queryKey: ['job', jobId],
  queryFn:  () => getJob(jobId!),
  enabled:  jobId !== null,
  refetchInterval: (query) => {
    const status = query.state.data?.status
    return status === 'done' || status === 'error' ? false : 1500
  },
})
```

`jobId` is genuinely local — it's "what is the user looking at". Everything else
is a **cache of server state**, and TanStack Query owns it: fetching, caching,
loading and error flags, invalidation, polling. There is no `useState` for job
data, no `useEffect` fetch, no manual `isLoading` boolean.

> **Principle.** Server state and client state are different things. Server
> state is *cached remote data* — it goes stale, it can refetch, it can be
> invalidated. Trying to model it with `useState` + `useEffect` is how you get
> race conditions and stale closures.

### Self-terminating polling

`refetchInterval` is a **function of the current data**, so the poll stops
itself when `status` becomes `done` or `error`. No timers to clear, no cleanup
function, no "component unmounted but the interval kept running" bug. The stop
condition lives right next to the start condition.

The comment records a tuning change: 3s felt laggy once stage transitions became
the point of the progress view, so it moved to 1.5s. A UX-driven parameter
change, documented.

### Optimistic updates in chat

`ChatPanel` implements the full optimistic-update triad:

```typescript
onMutate: async (message) => {
  await queryClient.cancelQueries({ queryKey: ['chat', jobId] })
  const previous = queryClient.getQueryData<ChatMessage[]>(['chat', jobId]) ?? []
  queryClient.setQueryData(['chat', jobId], [...previous, { role: 'user', content: message }])
  return { previous }                                   // rollback snapshot
},
onError:   (_e, _m, context) => queryClient.setQueryData(['chat', jobId], context?.previous ?? []),
onSettled: () => queryClient.invalidateQueries({ queryKey: ['chat', jobId] }),
```

1. `onMutate` — show the user's message *immediately*, before the round trip.
   Essential here, because the coach may spend several seconds re-watching video.
2. `onError` — roll back to the captured snapshot.
3. `onSettled` — invalidate so the server's version becomes truth, success or
   failure.

The `cancelQueries` call is the subtle one: without it, an in-flight refetch
could land *after* your optimistic write and clobber it.

> **Principle.** Optimistic UI is a three-part contract: apply, roll back,
> reconcile. Implementing only the first is how you get a UI that lies.

### Deriving state from the server instead of duplicating it

`ProcessingView` takes only `stage` and computes everything from it:

```typescript
const activeIndex = Math.max(0, STEPS.findIndex((s) => s.id === stage))
```

Ordering lives in a `STEPS` constant, so "which steps are done" is derived, not
tracked. And `Math.max(0, ...)` handles the `findIndex → -1` case for the window
before the first stage lands, deliberately showing step 1 rather than nothing —
the upload has finished, so work really is underway.

### The event-listener bug, and the fix

```typescript
// Props rather than addEventListener in an effect: the effect ran on mount,
// before the data arrived and before this element existed, so the listeners
// never attached and duration stayed 0.
onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
onTimeUpdate={(e) => setPlayhead(e.currentTarget.currentTime)}
```

A perfect illustration of a React trap. The `<video>` element renders
conditionally on query data; an effect running on mount finds `videoRef.current
=== null` and silently attaches nothing. React's own event props attach when the
element actually renders. **Prefer declarative props over imperative ref wiring;
reach for a ref only for things React doesn't model** — which is exactly what
`seekTo` does, setting `currentTime` and calling `play()`.

Note `void v.play().catch(() => {})` — browsers block autoplay in various
conditions, and a rejected promise here is expected and harmless.

### The honesty principle, carried into the UI

The backend exposes a `Swing` model whose docstring is the clearest statement of
the project's philosophy:

> Metrics are deliberately NOT sent as raw values. Our rotation figures are
> angles of a 2D projected shoulder line from a back view — the absolute number
> isn't in units a player thinks in, and quoting it implies precision we don't
> have. What IS trustworthy is how a quantity compares across the swings in one
> video.

So `_relative()` converts each metric to a **0–1 position within this video's
own range**, and the UI renders bars labelled `most` / `mid` / `least`, with the
caption *"Relative to your own swings in this video — not a score."* The same
rule is enforced in both system prompts: no degrees, no ratios, no normalized
units.

This is a single conviction — *we know the direction, not the magnitude* —
enforced consistently across the metric layer, the API shape, both prompts, and
the visual design. That consistency is rare and is the mark of a system with an
actual point of view.

> **Principle.** Only expose precision you actually have. The interface should
> make it impossible to over-read the data.

Note also `_relative()` returns `None` when there are fewer than two comparable
values — with one swing there is no "relative", and the bar is simply omitted
rather than shown at a meaningless 50%.

### Small things that show care

- **44px touch targets.** The timeline markers are 28px visually but wrapped in
  a 44×44 button — Apple's minimum. Smaller targets are genuinely hard to hit.
- **`aria-label`, `title`, `role="button"`, `tabIndex`, Enter/Space handling**
  on the drop zone: a `<div>` used as a button needs all of it to be usable by
  keyboard and screen reader.
- **`text-base sm:text-sm` on the chat input.** iOS Safari auto-zooms when you
  focus an input with font-size under 16px. Bumping to 16px on mobile only
  prevents that jarring zoom.
- **Portrait video handling.** Phone footage is tall; unconstrained it becomes a
  strip with black bars. `max-w-sm` + `max-h-[60vh]` + `object-contain`.
- **`pointer-events-none`** on the decorative background gradient, so it can
  never intercept a click.
- **Suggested prompts** in the empty chat state — solving the blank-page problem
  by showing what kinds of question work.
- **Typed API boundary.** `api.ts` holds every `fetch`, every response
  interface, and the base URL. Components never touch `fetch`. One file to
  change when the API changes.

---

<a name="part-7"></a>
## Part 7 — Deployment and operations

### The multi-stage Docker build

```dockerfile
FROM node:22-slim AS frontend      # stage 1: build the UI
...
FROM python:3.13-slim              # stage 2: the API, which also serves the UI
...
COPY --from=frontend /ui/dist ./static
```

Stage 1 needs Node, npm, and the whole `node_modules` tree — hundreds of MB. The
final image needs none of it, only the compiled `dist/`. Multi-stage builds let
you use heavy tooling during the build and ship only the output.

### Single origin, so CORS disappears

This is an elegant piece of design. The frontend is built with
`VITE_API_BASE_URL=""`, so the client calls `/videos` rather than
`http://host/videos` — relative to whatever origin served the page. FastAPI then
serves the built assets itself.

The result: in production, the UI and the API are the **same origin**, so CORS
doesn't apply at all. The `CORSMiddleware` is still configured, but only for
local development, where Vite runs on :5173 and the API on :8000.

> **Principle.** The best way to handle a class of problem is often to arrange
> for it not to exist. Same-origin removes CORS, preflight requests, and cookie
> `SameSite` headaches in one move.

### Mount order matters

```python
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
```

Registered **last, on purpose.** A mount at `/` matches everything, so
registering it before the API routes would shadow every one of them. And it's
**conditional**, because in local dev the directory doesn't exist and the
frontend runs on Vite. One file that works in both environments.

### Layer caching

```dockerfile
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY backend/ .
RUN uv sync --frozen --no-dev
```

Docker caches each layer and invalidates everything after the first change. Copy
the dependency manifests alone, install, *then* copy source — so editing a
Python file doesn't reinstall every dependency. The `--no-install-project` on
the first pass is what makes that split possible. `--frozen` means "use the
lockfile exactly", which is what makes builds reproducible.

### The invisible dependencies

```dockerfile
RUN apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg curl
```

Each one has a comment. `libgl1` and friends are **system** libraries that
opencv and mediapipe link against at import time — `pip install opencv-python`
succeeds and then `import cv2` crashes in a slim container with a
missing-shared-object error. This is one of the most common and most confusing
Docker failures in the Python CV world.

`ffmpeg` is the one to internalise: without it nothing breaks, the code just
falls back to sending 70× more data to Gemini. **The comment calls it "the kind
of regression you'd never notice."** Silent degradation is more dangerous than a
crash, and the right response is to make the requirement explicit where it's
installed.

### Fetching the model at build time

```dockerfile
RUN mkdir -p cv_spike/models && \
    curl -fsSL -o cv_spike/models/pose_landmarker_full.task "https://..."
```

The MediaPipe pose model is a 9.4MB binary, deliberately not committed to git.
Fetching it during the build keeps the repo light **and** keeps the build
self-contained — relying on a local file meant the image only built from a
working copy that happened to have it, not from a clean clone.

> **Principle.** A build should work from a fresh clone on a machine that has
> never seen the project. Anything that only works on your laptop is a trap for
> CI and for the next person.

### Migrate-then-start

```dockerfile
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
```

Schema changes ship with the code that needs them, with no separate manual step.
`&&` means a failed migration prevents the app from starting — which is correct:
better to fail the deploy loudly than to run new code against an old schema.

The caveat to know: with multiple instances, several containers race to migrate
at once. Alembic takes a lock so it's usually fine, but at real scale migrations
become their own deploy step (a Fly `release_command`, a Kubernetes job).

### The Fly configuration, line by line

```toml
auto_stop_machines = "stop"
min_machines_running = 0
```

Scale to zero when idle. This is a personal project, not a service that must be
warm — the first request after a sleep pays a cold start, and that's an accepted
trade for near-zero idle cost.

```toml
[http_service.concurrency]
  soft_limit = 4
  hard_limit = 8
```

Low, because analysis is CPU-heavy and holds a video in memory. Concurrency
limits make a burst **queue** rather than OOM the machine. Backpressure is a
feature: a slow response beats a dead process.

```toml
[[vm]]
  memory = "1gb"   # measured: 512MB OOMs during frame-by-frame extraction
```

The comment says "measured need, not a guess" — exactly the right way to justify
a resource number.

```toml
[[mounts]]
  source = "tennis_storage"
  destination = "/app/storage"
```

Fly machine filesystems are **ephemeral** — they reset on every deploy. Without
a volume, every uploaded video would vanish on release, and `rewatch_swing`
would break for every existing job. Knowing which parts of your container are
durable is fundamental to deploying statefully.

### The health check

```python
@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Deliberately does not touch the DB ..."""
    return {"status": "ok"}
```

This is a **liveness** probe: "is this process up and serving?" It deliberately
doesn't check the database, because the platform kills machines that fail their
health check — and a slow database would then take down an otherwise-healthy
app, turning a degradation into an outage. Worse, restarting wouldn't help.

A **readiness** probe (should I send traffic here?) *would* check dependencies.
Knowing which kind you're writing, and why, is the whole lesson.

### Retention sweep

```python
def _purge_old_videos() -> None:
    cutoff = time.time() - VIDEO_RETENTION_DAYS * 86400
    for path in STORAGE_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                ...
        except OSError:
            logger.warning("Could not purge %s", path.name, exc_info=True)
```

Videos stopped being deleted after analysis (chat needs them), which left
storage growing without bound. This sweep at startup is the replacement — honest
about being "enough for a single-instance deployment", with the note that object
storage lifecycle rules replace it when this moves to S3.

Two good habits visible here: the per-file `try/except` means one undeletable
file doesn't abort the whole sweep, and the retention window is an env var so it
can be tuned without a code change.

### Configuration through the environment

`CORS_ORIGINS`, `VIDEO_RETENTION_DAYS`, `DATABASE_URL`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY` — all environment variables, with sensible local defaults where
one exists:

```python
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
```

Secrets via `fly secrets set`, never in the image. `.env.example` is committed,
`.env` is not. This is the twelve-factor "config in the environment" rule, and
the reason it matters is that the *same artifact* must run in dev and prod with
only its environment differing.

---

<a name="part-8"></a>
## Part 8 — Cross-cutting lessons

If you take nothing else from this codebase, take these.

### 1. Comments explain *why*, never *what*

Almost every non-obvious line here carries a comment, and they are near-uniformly
about **reasoning, alternatives tried, and consequences** — not restatement:

- "Poll faster than before: stage changes are the point of the progress view,
  and a 3s interval made the steps feel laggy."
- "Props rather than addEventListener in an effect: the effect ran on mount,
  before the data arrived…"
- "Mounted LAST, on purpose."
- "1GB: the 9.4MB pose model plus frame-by-frame video decoding OOMs a 512MB
  machine during extraction. Measured need, not a guess."

The pattern to copy: **when you fix a bug or reject an approach, write the
rejected approach into the comment.** Code shows what you did; only a comment
can show what you tried and why it failed. That is what stops someone
"simplifying" your fix back into the bug.

### 2. Calibrate confidence, and propagate uncertainty

This is the spine of the whole system, and it recurs at every layer:

- `metrics.py` returns `None` when an angle is geometrically unreliable, plus a
  `notes` string explaining why
- `perception.py` gives the model a required `not_visible` field so uncertainty
  has somewhere to go
- both system prompts forbid speculating about nulls and forbid quoting raw
  numbers
- the API ships 0–1 relative positions instead of absolute values
- the UI labels them `most` / `mid` / `least` with "not a score"

One conviction — *we know direction, not magnitude* — enforced at five layers.
Most systems lose their uncertainty at the first boundary, and everything
downstream then treats a guess as a fact.

### 3. Measure, then record the measurement

Six FPS, 640px, 1GB, 0.25s smoothing, 1.2s separation, 0.4× reliability
threshold — every magic number in this codebase has a comment saying what was
tried and what it cost. A number with a rationale is a decision; a number
without one is a liability nobody dares change.

### 4. Delete the framework when it isn't earning its place

A three-node graph became one function because it was measured at ~46s vs. ~8s
for no quality gain. Meanwhile LangGraph *stayed* for chat, where cycles,
branching, tools and checkpointing are all genuinely used. Same library, two
opposite verdicts, both correct — because the verdict came from the requirements,
not from the library's marketing.

### 5. Classify your dependencies

Load-bearing (metrics) vs. supplementary (perception), and encode it: one raises,
the other returns `None` and the prompt is told. Do this consciously for every
external call and your system degrades instead of falling over.

### 6. Design for the boundary, not the happy path

Streamed uploads, path-traversal checks, `InjectedState` so the model can't
choose a `job_id`, tools that return English on failure, per-file `try/except` in
the sweep, `Math.max(0, findIndex(...))`, `_relative()` returning `None` for a
single swing. None of these matter when everything works. All of them are why it
keeps working.

### 7. Leave the honest TODOs in

"Retention is unbounded for now." "Enough for a single-instance deployment."
"Becomes an S3 key when storage moves off local disk." Known limitations,
written down where the code is, are worth more than a clean-looking codebase
whose constraints live only in someone's head.

---

## The known gaps, collected

An honest register of what a next iteration should address:

| Gap | Impact | Fix |
|---|---|---|
| No automated tests | Regressions in CV metrics are invisible | Golden-file tests: fixed landmark JSON → asserted metrics. The `cv_spike/` outputs are ready-made fixtures. |
| Stranded `processing` jobs | A crash mid-analysis leaves a job hung forever | Startup sweep marking stale `processing` → `error`; eventually a real queue |
| `BackgroundTasks` in-process | CPU-bound work competes with request serving; no retries | Task queue + separate workers |
| No auth | Any job id is readable by anyone who has it | Sessions/accounts, and scope `job_id` to a user |
| No upload size/type limit | A huge or non-video upload wastes a full pipeline run | Check `Content-Length` and sniff the container before enqueueing |
| Hardcoded `hand="right"` | Left-handed players get wrong-arm metrics | Infer from which wrist has the dominant speed peak, or ask |
| Repeated session boilerplate | Easy to leak a connection in new code | FastAPI `Depends` yield-dependency |
| Local disk storage | Ties the app to one machine with a volume | S3 + lifecycle policy (already anticipated in the comments) |

None of these are wrong for the project's current scope. They're the map of
where it goes next.
