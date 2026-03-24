from __future__ import annotations

from typing import Any

import numpy as np


def center_and_normalize_skeleton(
    pose3d: dict[str, Any],
    *,
    torso_length_aggregation: str = "median",
) -> dict[str, Any]:
    """
    Normalize 3D joint coordinates into relative units:
      - center skeleton by hip/torso
      - divide by torso length reference (shoulder-to-hip)
    """
    if torso_length_aggregation not in {"median", "per_frame"}:
        raise ValueError("torso_length_aggregation must be 'median' or 'per_frame'")

    xyz = np.asarray(pose3d["keypoints_3d"], dtype=np.float32)  # (T,J,3)
    names = list(pose3d["joint_names"])
    idx = {n: i for i, n in enumerate(names)}

    required_base = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
    ]
    missing = [n for n in required_base if n not in idx]
    if missing:
        raise ValueError(f"Missing required joints for normalization: {missing}")

    T = xyz.shape[0]
    nose = xyz[:, idx["nose"], :]
    l_sh = xyz[:, idx["left_shoulder"], :]
    r_sh = xyz[:, idx["right_shoulder"], :]
    l_hip = xyz[:, idx["left_hip"], :]
    r_hip = xyz[:, idx["right_hip"], :]
    l_knee = xyz[:, idx["left_knee"], :]
    r_knee = xyz[:, idx["right_knee"], :]

    neck = 0.5 * (l_sh + r_sh)
    head = nose
    shoulder_center = neck
    hip_center = 0.5 * (l_hip + r_hip)
    torso_vec = shoulder_center - hip_center
    torso_len = np.linalg.norm(torso_vec, axis=1)

    valid = (~np.isnan(torso_len)) & (torso_len > 1e-6)
    if not np.any(valid):
        torso_scale = np.ones((T,), dtype=np.float32)
    elif torso_length_aggregation == "median":
        s = float(np.nanmedian(torso_len[valid]))
        torso_scale = np.full((T,), max(1e-6, s), dtype=np.float32)
    else:
        torso_scale = np.clip(np.nan_to_num(torso_len, nan=np.nanmedian(torso_len[valid])), 1e-6, None)
        torso_scale = torso_scale.astype(np.float32)

    selected_names = [
        "head",
        "neck",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
    ]
    selected = np.stack([head, neck, l_sh, r_sh, l_hip, r_hip, l_knee, r_knee], axis=1)  # (T,8,3)
    centered = selected - hip_center[:, None, :]
    normalized = centered / torso_scale[:, None, None]

    torso_center_norm = np.zeros((T, 3), dtype=np.float32)
    shoulder_center_norm = (shoulder_center - hip_center) / torso_scale[:, None]

    return {
        "keypoints_3d": normalized.astype(np.float32),
        "joint_names": selected_names,
        "frame_indices": list(pose3d["frame_indices"]),
        "torso_center": torso_center_norm,
        "shoulder_center": shoulder_center_norm.astype(np.float32),
        "torso_scale": torso_scale.astype(np.float32),
        "source_lifting_meta": pose3d.get("lifting_meta", {}),
    }

