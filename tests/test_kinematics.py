import numpy as np

from tackle_vision_ml.kinematics import compute_kinematics


def test_kinematics_constant_velocity_has_near_zero_acceleration():
    T = 40
    fps = 20.0
    t = np.arange(T, dtype=np.float32) / fps

    # One joint with linear x(t) => constant velocity, near-zero acceleration.
    pos = np.zeros((T, 1, 3), dtype=np.float32)
    pos[:, 0, 0] = 2.0 * t  # x = 2t

    inp = {
        "keypoints_3d": pos,
        "joint_names": ["head"],
        "frame_indices": list(range(T)),
        "shoulder_center": np.zeros((T, 3), dtype=np.float32),
    }
    out = compute_kinematics(inp, fps=fps)

    # Ignore edges where finite-difference boundary effects are largest.
    acc = out["acc"][2:-2, 0, 0]
    assert np.nanmax(np.abs(acc)) < 1e-3

