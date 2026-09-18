"""
Throwaway spike: run MediaPipe's PoseLandmarker on a tennis swing video
and dump per-frame keypoints to JSON, plus a video with the skeleton
drawn on top so we can sanity-check the output visually.

This is NOT production code -- just getting hands-on with the library
and confirming pose detection actually works on real swing footage
filmed from behind the baseline before we build metrics on top of it.

Usage:
    .venv/bin/python pose_spike.py path/to/video.mp4
"""

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "models" / "pose_landmarker_full.task"
OUTPUT_DIR = SCRIPT_DIR / "output"

# The 33 landmark names MediaPipe Pose returns, in index order.
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
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

# Bones to draw for the visual sanity check.
POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def run(video_path: Path):
    OUTPUT_DIR.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {frame_count} frames")

    out_video_path = OUTPUT_DIR / f"{video_path.stem}_pose.mp4"
    out_json_path = OUTPUT_DIR / f"{video_path.stem}_landmarks.json"
    writer = cv2.VideoWriter(
        str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    frames_out = []
    detected_count = 0

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
                detected_count += 1
                landmarks = result.pose_landmarks[0]  # first (only) detected person
                points_px = {}
                for name, lm in zip(LANDMARK_NAMES, landmarks):
                    px, py = int(lm.x * width), int(lm.y * height)
                    points_px[name] = {
                        "x": round(lm.x, 4), "y": round(lm.y, 4),
                        "visibility": round(lm.visibility, 3),
                    }
                    cv2.circle(frame_bgr, (px, py), 4, (0, 255, 0), -1)

                for a, b in POSE_CONNECTIONS:
                    ax, ay = int(landmarks[LANDMARK_NAMES.index(a)].x * width), int(landmarks[LANDMARK_NAMES.index(a)].y * height)
                    bx, by = int(landmarks[LANDMARK_NAMES.index(b)].x * width), int(landmarks[LANDMARK_NAMES.index(b)].y * height)
                    cv2.line(frame_bgr, (ax, ay), (bx, by), (255, 200, 0), 2)

                frame_record["landmarks"] = points_px

            writer.write(frame_bgr)
            frames_out.append(frame_record)
            frame_idx += 1

    cap.release()
    writer.release()

    with open(out_json_path, "w") as f:
        json.dump(frames_out, f, indent=2)

    detection_rate = detected_count / max(frame_idx, 1) * 100
    print(f"Processed {frame_idx} frames, pose detected in {detected_count} ({detection_rate:.1f}%)")
    print(f"Skeleton video -> {out_video_path}")
    print(f"Landmarks JSON -> {out_json_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: pose_spike.py path/to/video.mp4")
    run(Path(sys.argv[1]))
