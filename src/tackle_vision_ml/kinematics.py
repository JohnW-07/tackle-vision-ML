from __future__ import annotations

from typing import Any

import numpy as np

def compute_kinematics(normalized_smoothed_pose3d: dict[str, Any], *, fps: float) -> dict[str, Any]:
    """
    Compute velocity, acceleration (and optionally jerk) from smoothed normalized 3D positions.

    Input contract (from `smoothing.py`):
      - keypoints_3d: array (T, J, 3), normalized relative coordinates
      - joint_names: list[str] length J
      - frame_indices: list[int]
    """
    xyz = np.asarray(normalized_smoothed_pose3d["keypoints_3d"], dtype=np.float32)  # (T,J,3)
    joints = list(normalized_smoothed_pose3d["joint_names"])
    frame_indices = list(normalized_smoothed_pose3d["frame_indices"])
    dt = 1.0 / max(1e-6, float(fps))

    vel = _grad(xyz, dt=dt)
    acc = _grad(vel, dt=dt)
    jerk = _grad(acc, dt=dt)

    vel_mag = np.linalg.norm(vel, axis=-1)  # (T,J)
    acc_mag = np.linalg.norm(acc, axis=-1)  # (T,J)
    jerk_mag = np.linalg.norm(jerk, axis=-1)  # (T,J)

    # Torso center is anchored at origin after centering, so we use shoulder_center
    # trajectory as the torso dynamics proxy.
    shoulder_center = np.asarray(normalized_smoothed_pose3d.get("shoulder_center"), dtype=np.float32)
    torso_vel = _grad(shoulder_center, dt=dt)
    torso_acc = _grad(torso_vel, dt=dt)
    torso_vel_mag = np.linalg.norm(torso_vel, axis=-1)
    torso_acc_mag = np.linalg.norm(torso_acc, axis=-1)

    # Signed deceleration along the velocity direction:
    # decel > 0 means acceleration opposes current velocity.
    vnorm = np.linalg.norm(torso_vel, axis=-1, keepdims=True)
    vhat = np.divide(torso_vel, np.clip(vnorm, 1e-6, None))
    accel_along_vel = np.sum(torso_acc * vhat, axis=-1)  # positive => speeding up
    torso_decel = -accel_along_vel  # positive => decelerating

    return {
        "joint_names": joints,
        "frame_indices": frame_indices,
        "fps": float(fps),
        "dt": float(dt),
        "pos": xyz,
        "vel": vel,
        "acc": acc,
        "jerk": jerk,
        "vel_mag": vel_mag,
        "acc_mag": acc_mag,
        "jerk_mag": jerk_mag,
        "torso_center": shoulder_center,
        "torso_vel": torso_vel,
        "torso_acc": torso_acc,
        "torso_vel_mag": torso_vel_mag,
        "torso_acc_mag": torso_acc_mag,
        "torso_decel": torso_decel,
    }


def _grad(x: np.ndarray, *, dt: float) -> np.ndarray:
    # np.gradient handles central differences internally and is stable for short clips.
    if x.shape[0] < 2:
        return np.zeros_like(x, dtype=np.float32)
    g = np.gradient(x, dt, axis=0, edge_order=1)
    return np.asarray(g, dtype=np.float32)

