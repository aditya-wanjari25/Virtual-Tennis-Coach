"""Throwaway: compute metrics for each detected swing and print them so
we can sanity-check values are plausible/consistent across 3 repeated
forehands of the same shot type."""

from pathlib import Path

from app.analysis.landmarks import load_pose_sequence
from app.analysis.metrics import compute_swing_metrics
from app.analysis.phases import segment_swings

DATA_PATH = Path(__file__).parent / "output" / "forehand_landmarks.json"

pose = load_pose_sequence(DATA_PATH)
swings = segment_swings(pose, hand="right")

def fmt(x, unit=""):
    return f"{x:7.1f}{unit}" if x is not None else "    n/a"

for i, swing in enumerate(swings):
    m = compute_swing_metrics(pose, swing, hand="right")
    print(f"\n--- Swing {i+1} (contact @ {m.contact_time_s:.2f}s) ---")
    print(f"  shoulder rotation @ contact:      {fmt(m.shoulder_rotation_at_contact_deg, ' deg')}")
    print(f"  hip rotation @ contact:           {fmt(m.hip_rotation_at_contact_deg, ' deg')}")
    print(f"  hip-shoulder separation:          {fmt(m.hip_shoulder_separation_at_contact_deg, ' deg')}")
    print(f"  max shoulder rotation (backswing):{fmt(m.max_shoulder_rotation_during_backswing_deg, ' deg')}")
    print(f"  swing path width (norm):          {m.swing_path_width:7.2f}")
    print(f"  arm extension @ contact (norm):   {m.arm_extension_at_contact:7.2f}")
    print(f"  stance width @ contact (norm):    {m.stance_width_at_contact:7.2f}")
    print(f"  follow-through finish height:     {m.follow_through_finish_height:7.2f}  (negative = above shoulder)")
    for note in m.notes:
        print(f"  NOTE: {note}")
