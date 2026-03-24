from __future__ import annotations

from typing import Any

import numpy as np


def estimate_pose2d_for_track(
    *,
    frames: list[np.ndarray],
    fps: float,
    frame_indices: list[int],
    device: str,
    pose2d_model: str,
    track: dict[str, Any],
) -> dict[str, Any]:
    """
    Estimate 2D keypoints for the selected track.

    Output contract (used by pose3d adapter later):
      - keypoints_2d: float32 array (T, K, 2) in pixel coordinates
      - keypoints_conf: float32 array (T, K) confidence
      - keypoint_names: list[str] names matching K
      - frame_indices: list[int]
    """
    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing vision dependencies. Install with `pip install -e .[vision]`."
        ) from e

    if not frames:
        raise ValueError("frames is empty")
    if len(frame_indices) != len(frames):
        raise ValueError("frame_indices must have the same length as frames")

    T = len(frames)
    K = 17  # COCO

    joint_names = [
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

    bboxes_xyxy = np.asarray(track.get("bboxes_xyxy"), dtype=float)
    if bboxes_xyxy.shape != (T, 4):
        raise ValueError("track['bboxes_xyxy'] must have shape (T,4)")

    device_arg: Any = "cpu"
    if device == "cuda":
        device_arg = 0

    pose_model = YOLO(pose2d_model)

    keypoints_2d = np.full((T, K, 2), np.nan, dtype=np.float32)
    keypoints_conf = np.full((T, K), np.nan, dtype=np.float32)

    conf_thres = 0.25
    iou_thres = 0.5
    pad_frac = 0.25  # padding around bbox crop

    H_img, W_img = frames[0].shape[:2]

    for t, frame in enumerate(frames):
        bbox = bboxes_xyxy[t]
        if np.any(np.isnan(bbox)):
            continue

        x1, y1, x2, y2 = map(float, bbox)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        pad_x = pad_frac * bw
        pad_y = pad_frac * bh

        cx1 = max(0.0, x1 - pad_x)
        cy1 = max(0.0, y1 - pad_y)
        cx2 = min(float(W_img), x2 + pad_x)
        cy2 = min(float(H_img), y2 + pad_y)

        if cx2 <= cx1 + 2 or cy2 <= cy1 + 2:
            continue

        x1i, y1i, x2i, y2i = int(cx1), int(cy1), int(cx2), int(cy2)
        crop = frame[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            continue

        # Run pose estimation only on the crop to reduce false positives and speed up inference.
        try:
            pred_list = pose_model.predict(
                crop,
                verbose=False,
                conf=conf_thres,
                iou=iou_thres,
                device=device_arg,
            )
        except TypeError:
            pred_list = pose_model.predict(crop, verbose=False, conf=conf_thres, device=device_arg)

        if not pred_list:
            continue

        res = pred_list[0]
        kps_obj = getattr(res, "keypoints", None)
        if kps_obj is None:
            continue

        xy = getattr(kps_obj, "xy", None)
        if xy is None:
            continue

        # xy: (N,17,2)
        xy_np = xy.detach().cpu().numpy().astype(np.float32)
        conf_np = getattr(kps_obj, "conf", None)
        if conf_np is not None:
            conf_np = conf_np.detach().cpu().numpy().astype(np.float32)  # (N,17)
        else:
            conf_np = np.ones((xy_np.shape[0], K), dtype=np.float32)

        # Choose the person instance with best keypoint confidence.
        # (Usually N==1 because the crop is tight around the selected track.)
        inst_scores = np.nanmean(conf_np, axis=1)
        best_i = int(np.nanargmax(inst_scores))

        kps_xy_crop = xy_np[best_i]  # (17,2)
        kps_conf = conf_np[best_i]  # (17,)

        # Translate keypoints back into full-frame coordinates.
        kps_xy_full = kps_xy_crop.copy()
        kps_xy_full[:, 0] += float(x1i)
        kps_xy_full[:, 1] += float(y1i)

        keypoints_2d[t] = kps_xy_full
        keypoints_conf[t] = kps_conf

    return {
        "keypoints_2d": keypoints_2d,
        "keypoints_conf": keypoints_conf,
        "keypoint_names": joint_names,
        "frame_indices": list(frame_indices),
        "image_size": {"width": int(W_img), "height": int(H_img)},
    }

