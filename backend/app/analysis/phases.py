"""Deterministic swing-phase segmentation from wrist trajectory.

Approach: a groundstroke's racket-hand speed is low at the top of the
backswing (a brief pause/transition point), rises to a peak around
contact, then falls again as the follow-through decelerates. So each
swing shows up as one peak in wrist speed, bounded by two local minima
(valleys) -- no ball tracking or ML classifier needed, just kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.analysis.landmarks import PoseSequence


@dataclass
class Swing:
    backswing_start_frame: int
    contact_frame: int
    follow_through_end_frame: int

    @property
    def frame_slice(self) -> slice:
        return slice(self.backswing_start_frame, self.follow_through_end_frame + 1)


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        return signal
    kernel = np.ones(window) / window
    # 'same' mode keeps array length aligned with the input for easy indexing
    return np.convolve(signal, kernel, mode="same")


def _find_local_maxima(signal: np.ndarray, min_distance: int, min_height: float) -> list[int]:
    candidates = [
        i for i in range(1, len(signal) - 1)
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1] and signal[i] >= min_height
    ]
    selected: list[int] = []
    for idx in sorted(candidates, key=lambda i: -signal[i]):
        if all(abs(idx - s) >= min_distance for s in selected):
            selected.append(idx)
    return sorted(selected)


def _lowest_point_between(signal: np.ndarray, start: int, end: int) -> int:
    """Index of the minimum value of `signal` within [start, end] inclusive.

    Used to find the true rest/pause point between two swings. A naive
    "walk until speed ticks up" approach gets trapped in the shallow
    notch of a bimodal swing (shoulder-drive bump, then wrist-snap bump)
    instead of finding the real low point -- taking the global min over
    the whole inter-peak window is robust to that.
    """
    lo, hi = min(start, end), max(start, end)
    return lo + int(np.argmin(signal[lo:hi + 1]))


def segment_swings(
    pose: PoseSequence,
    hand: str = "right",
    min_swing_gap_s: float = 1.2,
    smoothing_window_s: float = 0.25,
) -> list[Swing]:
    """Find individual swings in a pose sequence via wrist-speed peaks.

    A single swing's speed profile is often bimodal (shoulder-driven
    acceleration, then a wrist-snap pulse near contact) -- a short
    smoothing window can pick up both bumps as separate "swings". A
    ~0.25s window merges those into one peak per real swing while still
    resolving distinct swings that are ~1s+ apart.
    """
    wrist_xy = pose.xy[f"{hand}_wrist"]
    raw_speed = np.linalg.norm(np.diff(wrist_xy, axis=0), axis=1) * pose.fps  # normalized units/sec
    # raw_speed[i] is the speed between pose frames i and i+1

    window = max(1, int(round(smoothing_window_s * pose.fps)))
    speed = _moving_average(raw_speed, window)

    min_distance = int(round(min_swing_gap_s * pose.fps))
    min_height = speed.max() * 0.5  # only the dominant contact-speed peaks, not warm-up noise

    peak_indices = _find_local_maxima(speed, min_distance=min_distance, min_height=min_height)

    swings = []
    for i, peak_idx in enumerate(peak_indices):
        prev_peak = peak_indices[i - 1] if i > 0 else 0
        next_peak = peak_indices[i + 1] if i < len(peak_indices) - 1 else len(speed) - 1

        valley_left = _lowest_point_between(speed, prev_peak, peak_idx)
        valley_right = _lowest_point_between(speed, peak_idx, next_peak)

        # speed[i] sits between pose frames i and i+1 -- map back to pose
        # frame indices (contact ~= the frame right after the speed peak).
        swings.append(Swing(
            backswing_start_frame=valley_left,
            contact_frame=peak_idx + 1,
            follow_through_end_frame=valley_right + 1,
        ))

    return swings
