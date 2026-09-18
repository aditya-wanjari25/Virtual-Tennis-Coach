"""
Throwaway: plot wrist trajectory + speed over time from the captured
landmarks JSON, so we can visually pick out swing events before writing
phase-segmentation logic. Not production code.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path(__file__).parent / "output" / "forehand_landmarks.json"
OUT_PATH = Path(__file__).parent / "output" / "wrist_trajectory.png"

with open(DATA_PATH) as f:
    frames = json.load(f)

frames = [f for f in frames if f["landmarks"] is not None]

t = [f["timestamp_ms"] / 1000 for f in frames]

# Player faces the wall (away from camera), so we don't know handedness yet --
# plot both wrists and see which one has the bigger swing-like motion.
right_wrist_x = [f["landmarks"]["right_wrist"]["x"] for f in frames]
right_wrist_y = [f["landmarks"]["right_wrist"]["y"] for f in frames]
left_wrist_x = [f["landmarks"]["left_wrist"]["x"] for f in frames]
left_wrist_y = [f["landmarks"]["left_wrist"]["y"] for f in frames]

right_shoulder_x = [f["landmarks"]["right_shoulder"]["x"] for f in frames]
left_shoulder_x = [f["landmarks"]["left_shoulder"]["x"] for f in frames]

# crude speed = frame-to-frame displacement of normalized coords
def speed(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    dx, dy = np.diff(xs), np.diff(ys)
    return np.sqrt(dx**2 + dy**2)

right_speed = speed(right_wrist_x, right_wrist_y)
left_speed = speed(left_wrist_x, left_wrist_y)

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes[0].plot(t, right_wrist_x, label="right_wrist.x")
axes[0].plot(t, left_wrist_x, label="left_wrist.x")
axes[0].plot(t, right_shoulder_x, "--", label="right_shoulder.x", alpha=0.5)
axes[0].plot(t, left_shoulder_x, "--", label="left_shoulder.x", alpha=0.5)
axes[0].set_ylabel("normalized x (0=left, 1=right of frame)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_title("Wrist & shoulder horizontal position over time")

axes[1].plot(t, right_wrist_y, label="right_wrist.y")
axes[1].plot(t, left_wrist_y, label="left_wrist.y")
axes[1].invert_yaxis()  # image y grows downward; flip so "up" reads as up
axes[1].set_ylabel("normalized y (inverted: up=up)")
axes[1].legend(loc="upper right", fontsize=8)

axes[2].plot(t[1:], right_speed, label="right_wrist speed")
axes[2].plot(t[1:], left_speed, label="left_wrist speed")
axes[2].set_ylabel("frame-to-frame speed")
axes[2].set_xlabel("time (s)")
axes[2].legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120)
print(f"Saved -> {OUT_PATH}")
print(f"Frames with landmarks: {len(frames)}, duration: {t[-1]:.2f}s")
