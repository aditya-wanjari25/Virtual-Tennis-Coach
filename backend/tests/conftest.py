"""Shared fixtures.

Test videos are generated with ffmpeg at run time rather than committed, for
the same reason the pose model isn't committed (see the Dockerfile): it keeps
the repo light and the suite self-contained. It also means fixtures can't
silently drift from what they claim to be -- a clip described as 2s at 30fps
really is that, because ffmpeg just made it.

Everything here is small and short on purpose. The guardrail functions take
their limits as arguments, so a test can pass max_duration_s=1 against a 2s
clip instead of generating a 2-minute one. The suite stays fast, and the
limits under test stay independent of the production defaults.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Keep the suite out of the real Langfuse project. Set before any app import,
# because the client reads this when it is first constructed -- without it,
# running the tests files spans against production observability data, which
# is how synthetic job ids like "job-1" end up in a live dashboard.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def _encoder_available(name: str) -> bool:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True)
    return name.encode() in proc.stdout


def _lavfi(dest: Path, source: str, *out_args: str) -> Path:
    """Render a synthetic ffmpeg source to dest."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", source, *out_args, str(dest)],
        check=True,
        capture_output=True,
    )
    return dest


@pytest.fixture(scope="session")
def media(tmp_path_factory) -> Path:
    """Directory of generated test media, built once for the whole session."""
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def clip(media: Path) -> Path:
    """A perfectly ordinary short clip: 2s, 320x240, 30fps, h264."""
    return _lavfi(
        media / "clip.mp4",
        "testsrc=duration=2:size=320x240:rate=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
    )


@pytest.fixture(scope="session")
def hevc_clip(media: Path) -> Path:
    """An HEVC clip -- what an iPhone records by default, and the case an
    h264-only allowlist would wrongly reject."""
    if not _encoder_available("libx265"):
        pytest.skip("libx265 not available in this ffmpeg build")
    return _lavfi(
        media / "clip_hevc.mp4",
        "testsrc=duration=2:size=320x240:rate=30",
        "-c:v", "libx265", "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
    )


@pytest.fixture(scope="session")
def high_frame_clip(media: Path) -> Path:
    """Short but frame-dense: 2s at 240fps. Frame count, not duration, is what
    bounds pose-extraction cost, so the two need separate coverage."""
    return _lavfi(
        media / "dense.mp4",
        "testsrc=duration=2:size=320x240:rate=240",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
    )


@pytest.fixture(scope="session")
def big_clip(media: Path) -> Path:
    """8K, to trip the resolution cap. Low frame rate so it stays cheap."""
    return _lavfi(
        media / "big.mp4",
        "testsrc=duration=1:size=7680x4320:rate=5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
    )


@pytest.fixture(scope="session")
def mjpeg_clip(media: Path) -> Path:
    """A real video in a codec we deliberately don't accept."""
    return _lavfi(
        media / "clip.avi", "testsrc=duration=1:size=320x240:rate=30", "-c:v", "mjpeg",
    )


@pytest.fixture(scope="session")
def audio_only(media: Path) -> Path:
    """An .mp4 with no video track -- the 'I uploaded the wrong thing' case."""
    return _lavfi(media / "audio.mp4", "sine=frequency=440:duration=1", "-c:a", "aac")


@pytest.fixture(scope="session")
def junk_file(media: Path) -> Path:
    """Random bytes wearing a .mp4 extension."""
    path = media / "junk.mp4"
    path.write_bytes(b"\x00\x01\x02\x03" * 1024)
    return path


class FakeUpload:
    """Stands in for starlette's UploadFile over a fixed payload.

    Counts reads, so a test can assert the size cap stops consuming the stream
    rather than reading it all and complaining afterwards.
    """

    def __init__(self, total_bytes: int, chunk_cap: int = 1024) -> None:
        self.remaining = total_bytes
        self.chunk_cap = chunk_cap
        self.reads = 0

    async def read(self, size: int) -> bytes:
        self.reads += 1
        take = min(size, self.chunk_cap, self.remaining)
        self.remaining -= take
        return b"x" * take
