from __future__ import annotations

from typing import Any

import numpy as np

def detect_impact_window(kinematics: dict[str, Any], *, radius: int = 15) -> dict[str, Any]:
    """
    Detect likely impact frame based on motion/acceleration cues,
    then return a window around that frame for downstream feature extraction.
    """
    frame_indices = list(kinematics["frame_indices"])
    torso_acc_mag = np.asarray(kinematics["torso_acc_mag"], dtype=np.float32)  # (T,)
    torso_decel = np.asarray(kinematics["torso_decel"], dtype=np.float32)  # (T,)
    dt = float(kinematics["dt"])

    T = len(frame_indices)
    if T == 0:
        raise ValueError("No frames in kinematics")

    # Primary impact cue: torso acceleration magnitude peak.
    primary = np.nan_to_num(torso_acc_mag, nan=-np.inf)
    impact_idx = int(np.nanargmax(primary))

    # If there is a strong deceleration spike nearby, snap to the decel peak.
    decel = np.nan_to_num(torso_decel, nan=-np.inf)
    decel_idx = int(np.nanargmax(decel))
    if abs(decel_idx - impact_idx) <= 3:
        impact_idx = decel_idx

    start = max(0, impact_idx - radius)
    end = min(T - 1, impact_idx + radius)

    return {
        "impact_index_in_sequence": impact_idx,
        "impact_frame_index": int(frame_indices[impact_idx]),
        "impact_time_seconds": float(impact_idx * dt),
        "window_index_range": [start, end],
        "window_frame_range": [int(frame_indices[start]), int(frame_indices[end])],
    }

