"""Tests for the upload guardrails.

The thing these are really for: a validator that silently stops validating
looks exactly like one that works. Every test here asserts a specific
rejection, not just "no exception" -- otherwise a guardrail that accepted
everything would pass the suite.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.analysis.perception import _MIME_BY_SUFFIX
from app.guardrails.video import (
    ALLOWED_SUFFIXES,
    MAX_DURATION_S,
    MAX_FRAMES,
    MAX_UPLOAD_BYTES,
    VideoRejected,
    check_filename,
    probe,
    stream_to_disk,
    validate_video_file,
)
from tests.conftest import FakeUpload


# --- policy invariants ------------------------------------------------------

def test_every_accepted_suffix_has_a_mime_type():
    """The drift guard.

    perception sends uploads onward with a MIME type looked up by suffix, so
    accepting a suffix it can't label would mean a file that passes validation
    and then gets mislabelled as video/mp4 at the Gemini call. Asserting the
    subset here keeps the two lists honest without making the guardrail import
    the Gemini client.
    """
    assert ALLOWED_SUFFIXES <= set(_MIME_BY_SUFFIX)


def test_defaults_are_above_what_the_app_asks_users_for():
    """The UI tells players to film "five to fifteen seconds" and the sample
    clips run to 47s. Defaults that undercut the app's own guidance would
    reject ordinary footage, so the floor is worth pinning down."""
    assert MAX_DURATION_S >= 60
    assert MAX_FRAMES >= 1500          # 47s at 30fps is ~1400 frames
    assert MAX_UPLOAD_BYTES >= 50 * 1024 * 1024


# --- extension allowlist ----------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("swing.mp4", ".mp4"),
    ("swing.mov", ".mov"),
    ("IMG_4021.MOV", ".mov"),        # iPhone casing
    ("clip.webm", ".webm"),
    ("a.b.c.mp4", ".mp4"),           # only the final suffix counts
])
def test_check_filename_accepts_known_video_extensions(filename, expected):
    assert check_filename(filename) == expected


@pytest.mark.parametrize("filename", [
    "notes.txt",
    "noextension",
    "payload.exe",
    "clip.mp4.exe",                  # double extension, executable last
    "",
    None,
])
def test_check_filename_rejects_everything_else(filename):
    with pytest.raises(VideoRejected) as exc:
        check_filename(filename)
    assert exc.value.status == 415


def test_check_filename_discards_path_components():
    """Not a traversal guard -- a demonstration that none is needed.

    The stored filename is built from a server-generated uuid plus this return
    value, which is always a member of ALLOWED_SUFFIXES. So no part of a
    client-supplied name can reach the filesystem, however it's spelled.
    """
    assert check_filename("../../../etc/passwd.mp4") == ".mp4"
    assert check_filename("/absolute/path/clip.mov") == ".mov"


# --- streaming size cap -----------------------------------------------------

def test_stream_to_disk_writes_a_file_under_the_cap(tmp_path: Path):
    dest = tmp_path / "out.bin"
    written = asyncio.run(stream_to_disk(FakeUpload(5_000), dest, max_bytes=10_000))
    assert written == 5_000
    assert dest.stat().st_size == 5_000


def test_stream_to_disk_rejects_over_the_cap(tmp_path: Path):
    with pytest.raises(VideoRejected) as exc:
        asyncio.run(stream_to_disk(FakeUpload(100_000), tmp_path / "out.bin", max_bytes=10_000))
    assert exc.value.status == 413


def test_stream_to_disk_removes_the_partial_write(tmp_path: Path):
    """The point of the guardrail is that the disk isn't consumed. A rejection
    that leaves the bytes behind would have achieved nothing."""
    dest = tmp_path / "out.bin"
    with pytest.raises(VideoRejected):
        asyncio.run(stream_to_disk(FakeUpload(100_000), dest, max_bytes=10_000))
    assert not dest.exists()


def test_stream_to_disk_stops_reading_at_the_cap(tmp_path: Path):
    """Enforced mid-stream, not after.

    Reading the whole upload and then measuring it would mean a 10GB request
    is fully transferred before being refused -- the cap has to end the read
    loop, which is what the read count proves.
    """
    upload = FakeUpload(10_000_000, chunk_cap=1024)
    with pytest.raises(VideoRejected):
        asyncio.run(stream_to_disk(upload, tmp_path / "out.bin", max_bytes=10_000, chunk_size=1024))
    assert upload.reads < 20          # ~10 to hit the cap; ~9766 to drain it
    assert upload.remaining > 9_000_000


def test_stream_to_disk_cleans_up_when_the_client_hangs_up(tmp_path: Path):
    """A dropped connection raises CancelledError, which inherits from
    BaseException rather than Exception -- so an `except Exception` cleanup
    would miss the single most common way an upload fails."""
    class Hangup(FakeUpload):
        async def read(self, size: int) -> bytes:
            if self.reads > 2:
                raise asyncio.CancelledError()
            return await super().read(size)

    dest = tmp_path / "out.bin"
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(stream_to_disk(Hangup(10_000_000), dest, max_bytes=10_000_000))
    assert not dest.exists()


# --- the ffprobe gate -------------------------------------------------------

def test_accepts_an_ordinary_clip(clip: Path):
    info = validate_video_file(clip)
    assert info.codec == "h264"
    assert info.frames > 0
    assert info.duration_s == pytest.approx(2.0, abs=0.3)


def test_accepts_hevc(hevc_clip: Path):
    """iPhones record HEVC by default, so this is the most likely real upload.
    An h264-only codec allowlist would reject it -- hence a test of its own."""
    assert validate_video_file(hevc_clip).codec == "hevc"


def test_rejects_a_clip_over_the_duration_limit(clip: Path):
    with pytest.raises(VideoRejected) as exc:
        validate_video_file(clip, max_duration_s=1)
    assert exc.value.status == 422
    assert "limit is 1s" in exc.value.detail


def test_rejects_a_clip_over_the_frame_budget(high_frame_clip: Path):
    """2s at 240fps is well inside any duration cap but ~480 frames of CPU
    inference, which is the cost that actually matters."""
    assert validate_video_file(high_frame_clip, max_duration_s=90).frames > 400
    with pytest.raises(VideoRejected) as exc:
        validate_video_file(high_frame_clip, max_frames=100)
    assert exc.value.status == 422


def test_rejects_a_clip_over_the_resolution_limit(big_clip: Path):
    with pytest.raises(VideoRejected) as exc:
        validate_video_file(big_clip, max_long_edge_px=3840)
    assert exc.value.status == 422
    assert "7680x4320" in exc.value.detail


def test_rejects_a_codec_we_dont_accept(mjpeg_clip: Path):
    with pytest.raises(VideoRejected) as exc:
        validate_video_file(mjpeg_clip)
    assert exc.value.status == 415
    assert "mjpeg" in exc.value.detail


def test_rejects_a_file_with_no_video_track(audio_only: Path):
    with pytest.raises(VideoRejected) as exc:
        validate_video_file(audio_only)
    assert exc.value.status == 415


def test_rejects_bytes_that_arent_video(junk_file: Path):
    """A .mp4 extension is a claim, not evidence. This is why the probe exists
    on top of the extension allowlist."""
    with pytest.raises(VideoRejected) as exc:
        validate_video_file(junk_file)
    assert exc.value.status == 415


def test_rejects_a_missing_file(tmp_path: Path):
    with pytest.raises(VideoRejected):
        validate_video_file(tmp_path / "nothing.mp4")


def test_probe_reports_frames_and_fps(clip: Path):
    info = probe(clip)
    assert info.fps == pytest.approx(30.0, abs=0.1)
    assert info.width == 320 and info.height == 240
    # ~60 frames for 2s at 30fps; container metadata is an estimate, so this
    # asserts the ballpark rather than an exact count.
    assert 50 <= info.frames <= 70
