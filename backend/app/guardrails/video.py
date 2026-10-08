"""Guardrails on uploaded video: what we accept, and how much of it.

Checks run cheapest-first and, where possible, BEFORE a job row and background
task exist -- so a rejected upload is an immediate, specific HTTP error rather
than a job the client discovers has failed half a minute later behind a polling
spinner.

    1. extension allowlist  -- no bytes written yet
    2. size cap             -- enforced mid-stream, not after
    3. ffprobe gate         -- duration, resolution, codec, frame count
    4. frame budget         -- backstop inside pose extraction, see
                               pose_extraction.extract_pose_frames

Why a frame budget and not just a duration cap: pose extraction runs MediaPipe
inference on *every* frame with delegate=CPU. Measured ~59 fps locally on an
M3, but that figure comes from Metal-backed GL; Fly runs shared-cpu-1x with no
GPU, where it's realistically 5-15 fps. So frame count, not file size, is what
actually bounds both CPU burn and how long a user waits.

Limits are deliberately generous against real phone footage -- the sample clips
in cv_spike/ run 5-47s, 1.5-11MB, 162-1407 frames -- because the job here is to
stop abuse and runaway cost, not to second-guess a player's camera.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_MB", 100) * 1024 * 1024
MAX_DURATION_S = _env_int("MAX_VIDEO_DURATION_S", 90)
MAX_FRAMES = _env_int("MAX_VIDEO_FRAMES", 3000)
MAX_LONG_EDGE_PX = _env_int("MAX_VIDEO_LONG_EDGE_PX", 3840)

# Container extensions we accept. Kept as a subset of perception's
# _MIME_BY_SUFFIX so every accepted upload has a MIME type to send onward --
# tests/test_video_guardrails.py asserts that invariant rather than importing
# across the two modules, which would couple validation to the Gemini client.
# .wmv and .flv are dropped from that set on purpose: nothing films in them.
ALLOWED_SUFFIXES = frozenset({".mp4", ".mov", ".webm", ".avi", ".mpeg", ".mpg", ".3gp"})

# hevc is NOT optional here -- it's what iPhones record by default, so an
# h264-only allowlist would reject the single most likely upload we'll see.
ALLOWED_CODECS = frozenset({"h264", "hevc", "vp8", "vp9", "av1", "mpeg4"})

_PROBE_TIMEOUT_S = 30


class VideoRejected(Exception):
    """An upload failed validation.

    Carries the HTTP status the route should return, so the policy decision of
    "which failure is this" lives here with the check rather than at the route.
    """

    def __init__(self, detail: str, status: int = 422) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status = status


@dataclass(frozen=True)
class VideoInfo:
    """What ffprobe could tell us. `duration_s` and `frames` are 0 when the
    container didn't say -- see validate_video_file for how that's handled."""

    codec: str
    width: int
    height: int
    duration_s: float
    fps: float
    frames: int


def check_filename(filename: str | None) -> str:
    """Validate the upload's extension; returns a safe, normalized suffix.

    Also the reason the returned value is used to build the stored path instead
    of the raw filename: the result is always one of ALLOWED_SUFFIXES, so no
    part of a client-supplied name reaches the filesystem.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        shown = suffix or "(no extension)"
        raise VideoRejected(
            f"{shown} isn't a video format we can read. Try MP4, MOV or WebM.",
            status=415,
        )
    return suffix


def _as_size(num_bytes: int) -> str:
    """Human-readable byte count. Integer-divides to 0 for anything under a
    megabyte, which is what a test-sized cap looks like, so fall back to bytes
    rather than telling someone the limit is '0 MB'."""
    mb = num_bytes / (1024 * 1024)
    return f"{mb:.0f} MB" if mb >= 1 else f"{num_bytes} bytes"


async def stream_to_disk(
    file,
    dest: Path,
    max_bytes: int = MAX_UPLOAD_BYTES,
    chunk_size: int = 1024 * 1024,
) -> int:
    """Stream an UploadFile to disk, aborting if it exceeds max_bytes.

    The cap is checked inside the read loop, which is the whole point: testing
    the size after writing means the disk has already been consumed, so the
    check would run exactly too late to prevent what it's for.

    Cleans up the partial file on any exception -- including cancellation, which
    is the common case when a client hangs up mid-upload.
    """
    written = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await file.read(chunk_size):
                written += len(chunk)
                if written > max_bytes:
                    raise VideoRejected(
                        f"That video is over the {_as_size(max_bytes)} limit. "
                        "A few seconds of footage is all we need.",
                        status=413,
                    )
                f.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return written


def _as_float(value: object) -> float:
    """ffprobe reports numbers as strings, and 'N/A' when it doesn't know."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _as_fps(value: object) -> float:
    """r_frame_rate arrives as a fraction string: '30/1', or '0/0' when the
    stream has no fixed rate."""
    if not isinstance(value, str) or "/" not in value:
        return _as_float(value)
    num, _, den = value.partition("/")
    denominator = _as_float(den)
    return _as_float(num) / denominator if denominator else 0.0


def probe(path: Path) -> VideoInfo:
    """Read stream metadata with ffprobe.

    Fails closed. ffmpeg is a declared dependency (installed in the Dockerfile
    for the Gemini downscale), and a guardrail that quietly turns itself off
    when a binary goes missing is the exact failure mode guardrails exist to
    prevent. The cost of that choice is bounded: the size cap above and the
    frame budget in pose extraction are independent of ffprobe, so they still
    hold even if this check can't run.
    """
    if not shutil.which("ffprobe"):
        raise VideoRejected(
            "Can't validate video right now -- ffprobe is unavailable on the server.",
            status=503,
        )

    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                # ffprobe separates entry groups with ':' -- one string, not two.
                "-show_entries",
                "stream=codec_name,width,height,duration,r_frame_rate,nb_frames:format=duration",
                "-print_format", "json",
                str(path),
            ],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        # A deliberately malformed container can make ffprobe spin.
        raise VideoRejected("That file took too long to read. Is it a valid video?") from e

    if proc.returncode != 0:
        logger.info("ffprobe rejected %s: %s", path.name, proc.stderr.decode()[:200])
        raise VideoRejected("That file isn't a video we can read.", status=415)

    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as e:
        raise VideoRejected("That file isn't a video we can read.", status=415) from e

    streams = data.get("streams") or []
    if not streams:
        raise VideoRejected(
            "That file has no video track -- an audio file or image won't work here.",
            status=415,
        )
    stream = streams[0]

    # Duration lives on the stream for some containers and only on the format
    # for others (MOV in particular), so fall back rather than assume.
    duration = _as_float(stream.get("duration")) or _as_float(
        (data.get("format") or {}).get("duration")
    )
    fps = _as_fps(stream.get("r_frame_rate"))

    # nb_frames is absent or 'N/A' in plenty of containers; derive it when so.
    # Either way it's an estimate: ffprobe said 206 frames for a sample clip
    # that OpenCV then read 193 of, which is why the real enforcement point is
    # the counter inside the extraction loop.
    frames = int(_as_float(stream.get("nb_frames")) or round(duration * fps))

    return VideoInfo(
        codec=str(stream.get("codec_name") or "").lower(),
        width=int(_as_float(stream.get("width"))),
        height=int(_as_float(stream.get("height"))),
        duration_s=duration,
        fps=fps,
        frames=frames,
    )


def validate_video_file(
    path: Path,
    max_duration_s: int = MAX_DURATION_S,
    max_frames: int = MAX_FRAMES,
    max_long_edge_px: int = MAX_LONG_EDGE_PX,
) -> VideoInfo:
    """Probe a stored upload and reject it if it's outside what we'll process."""
    info = probe(path)

    if info.codec not in ALLOWED_CODECS:
        raise VideoRejected(
            f"That video uses the {info.codec or 'unknown'} codec, which we can't decode. "
            "Re-saving it as MP4 usually fixes this.",
            status=415,
        )

    if info.duration_s > max_duration_s:
        raise VideoRejected(
            f"That clip is {info.duration_s:.0f}s long; the limit is {max_duration_s}s. "
            "Trim it to the few swings you want looked at."
        )

    long_edge = max(info.width, info.height)
    if long_edge > max_long_edge_px:
        raise VideoRejected(
            f"That video is {info.width}x{info.height}, larger than the "
            f"{max_long_edge_px}px limit on the long edge. Any phone preset below 4K is fine."
        )

    # frames == 0 means the container reported neither a frame count nor a
    # duration. Rather than reject footage that's merely unusual, let it through
    # -- the size cap already bounds it, and the budget inside the extraction
    # loop will stop it there if it really is enormous.
    if info.frames > max_frames:
        raise VideoRejected(
            f"That clip has about {info.frames} frames; we analyse up to {max_frames}. "
            "A shorter clip, or one at a lower frame rate, will go through."
        )

    logger.info(
        "Accepted upload %s: %s %dx%d, %.1fs, ~%d frames",
        path.name, info.codec, info.width, info.height, info.duration_s, info.frames,
    )
    return info
