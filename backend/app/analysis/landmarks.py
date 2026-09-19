"""Load pose landmark JSON (produced by cv_spike/pose_spike.py) into a
convenient numpy-backed structure for downstream phase/metric code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


@dataclass
class PoseSequence:
    """Landmark positions for every detected frame of a video.

    xy[landmark_name] -> (n_frames, 2) array of normalized (x, y) coords.
    visibility[landmark_name] -> (n_frames,) array of confidence scores.
    timestamps_s -> (n_frames,) array of seconds from video start.
    fps -> approximate frame rate, derived from consecutive timestamps.
    """

    xy: dict[str, np.ndarray]
    visibility: dict[str, np.ndarray]
    timestamps_s: np.ndarray
    fps: float

    def __len__(self) -> int:
        return len(self.timestamps_s)


def pose_sequence_from_frames(raw_frames: list[dict]) -> PoseSequence:
    """Build a PoseSequence from the raw per-frame dicts produced by pose
    extraction: [{"frame": int, "timestamp_ms": int, "landmarks": dict|None}, ...]
    """
    # Only keep frames where a pose was actually detected -- gaps (missed
    # detections) get silently dropped rather than interpolated for now.
    detected = [f for f in raw_frames if f["landmarks"] is not None]
    if len(detected) < 2:
        raise ValueError("Too few detected frames to analyze")

    timestamps_s = np.array([f["timestamp_ms"] / 1000 for f in detected])

    xy: dict[str, np.ndarray] = {}
    visibility: dict[str, np.ndarray] = {}
    for name in LANDMARK_NAMES:
        xy[name] = np.array([[f["landmarks"][name]["x"], f["landmarks"][name]["y"]] for f in detected])
        visibility[name] = np.array([f["landmarks"][name]["visibility"] for f in detected])

    fps = 1.0 / np.median(np.diff(timestamps_s))

    return PoseSequence(xy=xy, visibility=visibility, timestamps_s=timestamps_s, fps=fps)


def load_pose_sequence(json_path: str | Path) -> PoseSequence:
    with open(json_path) as f:
        raw_frames = json.load(f)
    return pose_sequence_from_frames(raw_frames)
