import numpy as np

from tackle_vision_ml.normalization import center_and_normalize_skeleton


def test_center_and_normalize_uses_hip_center_and_torso_scale():
    # Build minimal 17-joint structure with key names used by normalization.
    names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    T, J = 5, 17
    xyz = np.zeros((T, J, 3), dtype=np.float32)
    idx = {n: i for i, n in enumerate(names)}

    # Constant torso geometry: shoulder center at y=2, hip center at y=0 => torso length=2.
    xyz[:, idx["left_shoulder"], :] = np.array([-0.5, 2.0, 0.0], dtype=np.float32)
    xyz[:, idx["right_shoulder"], :] = np.array([0.5, 2.0, 0.0], dtype=np.float32)
    xyz[:, idx["left_hip"], :] = np.array([-0.5, 0.0, 0.0], dtype=np.float32)
    xyz[:, idx["right_hip"], :] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    xyz[:, idx["left_knee"], :] = np.array([-0.5, -1.0, 0.0], dtype=np.float32)
    xyz[:, idx["right_knee"], :] = np.array([0.5, -1.0, 0.0], dtype=np.float32)
    xyz[:, idx["nose"], :] = np.array([0.0, 3.0, 0.0], dtype=np.float32)

    out = center_and_normalize_skeleton(
        {"keypoints_3d": xyz, "joint_names": names, "frame_indices": list(range(T))},
        torso_length_aggregation="median",
    )

    # Hip center is origin after centering.
    lh = out["joint_names"].index("left_hip")
    rh = out["joint_names"].index("right_hip")
    np.testing.assert_allclose(out["keypoints_3d"][:, lh, 1], 0.0, atol=1e-5)
    np.testing.assert_allclose(out["keypoints_3d"][:, rh, 1], 0.0, atol=1e-5)

    # Neck (shoulder center) y should normalize from 2.0 / torso_len(2.0) => 1.0
    neck = out["joint_names"].index("neck")
    np.testing.assert_allclose(out["keypoints_3d"][:, neck, 1], 1.0, atol=1e-5)

