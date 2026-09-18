"""Deterministic, back-view groundstroke metrics computed from pose
landmarks -- no LLM involved. Everything here is normalized by shoulder
width at the relevant frame so measurements are comparable across videos
shot at different distances from the camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.analysis.landmarks import PoseSequence
from app.analysis.phases import Swing


def _angle_deg(p_from: np.ndarray, p_to: np.ndarray) -> float:
    """Angle of the vector p_from -> p_to relative to the horizontal, in
    degrees. Used for shoulder/hip line orientation (rotation)."""
    dx, dy = p_to - p_from
    return float(np.degrees(np.arctan2(dy, dx)))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _reference_shoulder_width(pose: PoseSequence, min_visibility: float = 0.7) -> float:
    """A stable body-scale reference for normalizing distances.

    Shoulder width at a single frame is a bad reference: as the torso
    rotates through a swing, the shoulder line foreshortens (can shrink
    toward zero when the torso is edge-on to the camera), which blows up
    anything normalized by it right when it's least reliable. The median
    across all well-tracked frames in the video is far more stable.
    """
    vis_ok = (pose.visibility["left_shoulder"] >= min_visibility) & (pose.visibility["right_shoulder"] >= min_visibility)
    widths = np.linalg.norm(pose.xy["left_shoulder"][vis_ok] - pose.xy["right_shoulder"][vis_ok], axis=1)
    return float(np.median(widths))


@dataclass
class SwingMetrics:
    contact_time_s: float

    # Rotation fields are Optional: near edge-on torso orientation makes
    # the shoulder/hip line angle numerically unreliable (see
    # min_reliable_line_length below), and we'd rather omit a field than
    # report a garbage number for it. `notes` explains any omissions.
    shoulder_rotation_at_contact_deg: float | None
    hip_rotation_at_contact_deg: float | None
    hip_shoulder_separation_at_contact_deg: float | None
    max_shoulder_rotation_during_backswing_deg: float | None

    swing_path_width: float  # normalized by shoulder width
    arm_extension_at_contact: float  # normalized
    stance_width_at_contact: float  # normalized

    follow_through_finish_height: float  # wrist.y - shoulder.y at end, normalized (negative = above shoulder)

    notes: list[str]


def compute_swing_metrics(
    pose: PoseSequence,
    swing: Swing,
    hand: str = "right",
    max_backswing_s: float = 0.6,
    max_follow_through_s: float = 0.5,
) -> SwingMetrics:
    """Compute metrics for one swing.

    `swing.backswing_start_frame`/`follow_through_end_frame` mark the
    low-speed "rest" points used to *locate* the swing (see phases.py),
    but between repeated shots (e.g. a wall rally) that window also
    includes unrelated recovery footwork -- a real backswing-to-contact
    motion only takes ~0.3-0.6s. We cap the window used for metrics to
    that realistic duration so footwork between shots doesn't get
    counted as part of the swing.
    """
    max_backswing_frames = int(round(max_backswing_s * pose.fps))
    max_follow_through_frames = int(round(max_follow_through_s * pose.fps))
    backswing_start_frame = max(swing.backswing_start_frame, swing.contact_frame - max_backswing_frames)
    follow_through_end_frame = min(swing.follow_through_end_frame, swing.contact_frame + max_follow_through_frames)
    swing = Swing(
        backswing_start_frame=backswing_start_frame,
        contact_frame=swing.contact_frame,
        follow_through_end_frame=follow_through_end_frame,
    )

    shoulder_width = _reference_shoulder_width(pose)
    if shoulder_width < 1e-6:
        raise ValueError("Degenerate shoulder width reference -- pose likely unreliable throughout")

    # A shoulder/hip line's angle becomes numerically unstable once the
    # torso rotates close to edge-on to the camera (the two points nearly
    # coincide, so tiny pixel jitter -> huge angle swings). Below this
    # fraction of the reference shoulder width, don't trust the angle.
    min_reliable_line_length = 0.4 * shoulder_width

    def rotation_deg(frame: int) -> float | None:
        # left->right shoulder/hip line angle; 0 deg = facing directly
        # away from camera with shoulders square to it.
        left, right = pose.xy["left_shoulder"][frame], pose.xy["right_shoulder"][frame]
        if _dist(left, right) < min_reliable_line_length:
            return None
        return _angle_deg(left, right)

    def hip_rotation_deg(frame: int) -> float | None:
        left, right = pose.xy["left_hip"][frame], pose.xy["right_hip"][frame]
        if _dist(left, right) < min_reliable_line_length:
            return None
        return _angle_deg(left, right)

    notes: list[str] = []
    contact = swing.contact_frame
    shoulder_rot = rotation_deg(contact)
    hip_rot = hip_rotation_deg(contact)
    separation = shoulder_rot - hip_rot if shoulder_rot is not None and hip_rot is not None else None
    if shoulder_rot is None or hip_rot is None:
        notes.append(
            f"shoulder/hip rotation at contact (frame {contact}) omitted -- torso was too "
            "edge-on to the camera for a reliable angle"
        )

    backswing_window = range(swing.backswing_start_frame, contact + 1)
    shoulder_rots_during_backswing = [r for f in backswing_window if (r := rotation_deg(f)) is not None]
    if shoulder_rots_during_backswing:
        max_rotation = max(shoulder_rots_during_backswing, key=abs)
    else:
        max_rotation = None
        notes.append("max shoulder rotation during backswing omitted -- no reliable frames in window")

    forward_swing_window = slice(swing.backswing_start_frame, contact + 1)
    wrist_x_during_swing = pose.xy[f"{hand}_wrist"][forward_swing_window, 0]
    swing_path_width = float(wrist_x_during_swing.max() - wrist_x_during_swing.min()) / shoulder_width

    wrist_at_contact = pose.xy[f"{hand}_wrist"][contact]
    shoulder_at_contact = pose.xy[f"{hand}_shoulder"][contact]
    arm_extension = _dist(wrist_at_contact, shoulder_at_contact) / shoulder_width

    stance_width = _dist(pose.xy["left_ankle"][contact], pose.xy["right_ankle"][contact]) / shoulder_width

    finish_frame = swing.follow_through_end_frame
    finish_wrist_y = pose.xy[f"{hand}_wrist"][finish_frame, 1]
    finish_shoulder_y = pose.xy[f"{hand}_shoulder"][finish_frame, 1]
    # image y grows downward, so a negative value means the wrist finished above the shoulder
    finish_height = float(finish_wrist_y - finish_shoulder_y) / shoulder_width

    return SwingMetrics(
        contact_time_s=float(pose.timestamps_s[contact]),
        shoulder_rotation_at_contact_deg=shoulder_rot,
        hip_rotation_at_contact_deg=hip_rot,
        hip_shoulder_separation_at_contact_deg=separation,
        max_shoulder_rotation_during_backswing_deg=max_rotation,
        swing_path_width=swing_path_width,
        arm_extension_at_contact=arm_extension,
        stance_width_at_contact=stance_width,
        follow_through_finish_height=finish_height,
        notes=notes,
    )
