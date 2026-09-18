"""Throwaway: visualize detected swing phases against wrist speed to
sanity-check segment_swings() before trusting it for metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from app.analysis.landmarks import load_pose_sequence
from app.analysis.phases import segment_swings, _moving_average

DATA_PATH = Path(__file__).parent / "output" / "forehand_landmarks.json"
OUT_PATH = Path(__file__).parent / "output" / "segmentation_check.png"

pose = load_pose_sequence(DATA_PATH)
swings = segment_swings(pose, hand="right")

print(f"Detected {len(swings)} swing(s):")
for i, s in enumerate(swings):
    t0 = pose.timestamps_s[s.backswing_start_frame]
    tc = pose.timestamps_s[s.contact_frame]
    t1 = pose.timestamps_s[s.follow_through_end_frame]
    print(f"  Swing {i+1}: backswing@{t0:.2f}s -> contact@{tc:.2f}s -> follow-through end@{t1:.2f}s")

wrist_xy = pose.xy["right_wrist"]
raw_speed = np.linalg.norm(np.diff(wrist_xy, axis=0), axis=1) * pose.fps
window = max(1, int(round(0.1 * pose.fps)))
speed = _moving_average(raw_speed, window)
t_speed = pose.timestamps_s[1:]

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_speed, speed, label="smoothed right_wrist speed")
colors = ["tab:red", "tab:green", "tab:purple", "tab:orange"]
for i, s in enumerate(swings):
    c = colors[i % len(colors)]
    ax.axvline(pose.timestamps_s[s.backswing_start_frame], color=c, linestyle="--", alpha=0.6)
    ax.axvline(pose.timestamps_s[s.contact_frame], color=c, linestyle="-", alpha=0.9, label=f"swing {i+1} contact")
    ax.axvline(pose.timestamps_s[s.follow_through_end_frame], color=c, linestyle=":", alpha=0.6)

ax.set_xlabel("time (s)")
ax.set_ylabel("wrist speed")
ax.set_title("Detected swing phases (dashed=backswing start, solid=contact, dotted=follow-through end)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120)
print(f"Saved -> {OUT_PATH}")
