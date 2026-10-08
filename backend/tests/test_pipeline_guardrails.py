"""Tests for the two guardrails that sit inside the pipeline rather than at the
HTTP boundary: the frame budget in pose extraction, and the storage-path guard
the chat tools share with the video route.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent import chat_tools
from app.analysis.pose_extraction import DEFAULT_MODEL_PATH, extract_pose_frames


# --- the frame budget -------------------------------------------------------

requires_pose_model = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="pose model not present (fetched at Docker build time, not committed)",
)


@requires_pose_model
def test_extraction_stops_at_the_frame_budget(clip: Path):
    """The backstop for the probe being an estimate.

    ffprobe reads frame counts from container metadata, which can disagree with
    what OpenCV actually decodes (206 vs 193 on one of the sample clips) or be
    absent entirely. So the budget is re-enforced where the frames are real --
    without this, a clip with lying metadata would get the full CPU cost the
    upload guardrail was supposed to prevent.
    """
    assert len(extract_pose_frames(clip, max_frames=10)) == 10


@requires_pose_model
def test_extraction_is_unaffected_when_under_the_budget(clip: Path):
    """A budget that truncated normal clips would quietly degrade every
    analysis, which is worse than the DoS it prevents."""
    full = extract_pose_frames(clip, max_frames=10_000)
    assert 50 <= len(full) <= 70          # ~60 frames for 2s at 30fps
    assert len(full) == len(extract_pose_frames(clip))


# --- the storage path guard -------------------------------------------------

def test_stored_video_resolves_a_real_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(chat_tools, "STORAGE_DIR", tmp_path)
    (tmp_path / "clip.mp4").write_bytes(b"data")
    assert chat_tools._stored_video("clip.mp4") == tmp_path / "clip.mp4"


def test_stored_video_returns_none_for_a_missing_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(chat_tools, "STORAGE_DIR", tmp_path)
    assert chat_tools._stored_video("gone.mp4") is None


@pytest.mark.parametrize("key", [
    "../../../etc/passwd",
    "../outside.mp4",
    "/etc/passwd",
    "subdir/../../escape.mp4",
])
def test_stored_video_refuses_to_escape_the_storage_directory(key, tmp_path: Path, monkeypatch):
    """video_key is server-generated today, so this isn't reachable now.

    It's tested because the video route already guarded this same join while
    the chat tool didn't, and one of two paths to the same directory being
    unguarded is exactly how that stops being true.
    """
    monkeypatch.setattr(chat_tools, "STORAGE_DIR", tmp_path)
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"data")
    try:
        assert chat_tools._stored_video(key) is None
    finally:
        outside.unlink(missing_ok=True)
