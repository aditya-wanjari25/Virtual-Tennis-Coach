"""Extract pose landmarks from a video file using MediaPipe's
PoseLandmarker. Promoted from cv_spike/pose_spike.py once the CV spike
confirmed this approach works (see project notes) -- this is the real,
reusable version the FastAPI backend calls per upload.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from app.analysis.landmarks import LANDMARK_NAMES, PoseSequence, pose_sequence_from_frames

# backend/app/analysis/pose_extraction.py -> backend/cv_spike/models/
DEFAULT_MODEL_PATH = Path(__file__).parents[2] / "cv_spike" / "models" / "pose_landmarker_full.task"


def extract_pose_frames(video_path: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH) -> list[dict]:
    """Run pose detection on every frame of a video. Returns raw per-frame
    dicts (JSON-serializable) -- see pose_sequence_from_frames() to turn
    these into a PoseSequence for analysis."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    frames_out = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_record = {"frame": frame_idx, "timestamp_ms": timestamp_ms, "landmarks": None}
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]  # first (only) detected person
                frame_record["landmarks"] = {
                    name: {"x": round(lm.x, 4), "y": round(lm.y, 4), "visibility": round(lm.visibility, 3)}
                    for name, lm in zip(LANDMARK_NAMES, landmarks)
                }

            frames_out.append(frame_record)
            frame_idx += 1

    cap.release()
    return frames_out


def extract_pose_sequence(video_path: str | Path, model_path: str | Path = DEFAULT_MODEL_PATH) -> PoseSequence:
    frames = extract_pose_frames(video_path, model_path)
    return pose_sequence_from_frames(frames)
