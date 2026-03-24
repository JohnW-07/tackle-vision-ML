from __future__ import annotations

from typing import Any

import numpy as np


def track_and_select_tackling_player(
    *,
    frames: list[np.ndarray],
    fps: float,
    frame_indices: list[int],
    device: str,
    yolo_detect_model: str,
) -> dict[str, Any]:
    """
    Detect people with YOLOv8 and track with ByteTrack, then select the tackling player track.

    Returns a `selection` dictionary consumed by downstream modules.
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
    H, W = frames[0].shape[:2]

    # Ultralytics class list usually includes COCO 'person' at index 0.
    # We avoid hard-failing if class names differ by using both class id and names.
    model = YOLO(yolo_detect_model)
    person_class_ids = {0}
    if getattr(model, "names", None):
        for cid, name in model.names.items():
            if str(name).lower() == "person":
                person_class_ids.add(int(cid))

    # Tracking containers: track_id -> frame_index_in_sequence -> (xyxy, conf)
    track_obs: dict[int, dict[int, tuple[np.ndarray, float]]] = {}

    # Device mapping for ultralytics
    device_arg: Any = "cpu"
    if device == "cuda":
        # Most laptop setups only have a single GPU; ultralytics accepts device ids.
        device_arg = 0

    # Heuristic weights. These can be tuned later; defaults prioritize motion + closeness.
    w_motion = 1.0
    w_bottom = 0.6
    w_size = 0.3
    w_presence = 0.5

    conf_thres = 0.25
    iou_thres = 0.5

    # Run detection + tracking frame-by-frame (simple + robust with CPU).
    for t, frame in enumerate(frames):
        # Ultralytics track expects HWC uint8/float images; frames are already RGB.
        # Different ultralytics versions accept slightly different kwargs, so we try progressively fewer.
        try:
            res_list = model.track(
                frame,
                persist=True,
                verbose=False,
                conf=conf_thres,
                iou=iou_thres,
                tracker="bytetrack.yaml",
                classes=sorted(person_class_ids),
                device=device_arg,
            )
        except TypeError:
            try:
                res_list = model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    conf=conf_thres,
                    iou=iou_thres,
                    tracker="bytetrack.yaml",
                    device=device_arg,
                )
            except TypeError:
                res_list = model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    conf=conf_thres,
                    iou=iou_thres,
                    device=device_arg,
                )

        # res_list is usually length 1 (one image/frame); tolerate other shapes.
        results = res_list[0]
        boxes = getattr(results, "boxes", None)
        if boxes is None:
            continue

        xyxy = boxes.xyxy  # (N,4)
        confs = boxes.conf  # (N,)
        clss = boxes.cls  # (N,)
        ids = boxes.id  # (N,) track ids or None

        if xyxy is None or len(xyxy) == 0:
            continue
        if ids is None:
            continue

        xyxy_np = xyxy.detach().cpu().numpy()
        conf_np = confs.detach().cpu().numpy()
        cls_np = clss.detach().cpu().numpy().astype(int)
        id_np = ids.detach().cpu().numpy().astype(int)

        for i in range(xyxy_np.shape[0]):
            cls_id = int(cls_np[i])
            if cls_id not in person_class_ids:
                continue
            track_id = int(id_np[i])
            if track_id not in track_obs:
                track_obs[track_id] = {}
            track_obs[track_id][t] = (xyxy_np[i].astype(float), float(conf_np[i]))

    if not track_obs:
        raise ValueError("No tracked person trajectories found in the clip.")

    # Score each track using motion + bottom proximity + size + presence.
    def score_track(track_id: int) -> float:
        obs = track_obs[track_id]
        ts_sorted = sorted(obs.keys())
        presence_ratio = len(ts_sorted) / float(T)

        # Compute center positions in the image for motion score.
        centers = []
        conf_weight = []
        for t in ts_sorted:
            bbox, conf = obs[t]
            x1, y1, x2, y2 = bbox
            centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            conf_weight.append(conf)

        centers_np = np.asarray(centers, dtype=float)  # (n,2)
        conf_weight_np = np.asarray(conf_weight, dtype=float)

        # Motion: average step displacement magnitude.
        if centers_np.shape[0] >= 2:
            disps = centers_np[1:] - centers_np[:-1]
            disp_mag = np.linalg.norm(disps, axis=1)
            motion_score = float(np.average(disp_mag, weights=conf_weight_np[1:]))
        else:
            motion_score = 0.0

        # Bottom proximity: how far down the player tends to be.
        y_centers = centers_np[:, 1]
        bottom_score = float(np.average(y_centers / max(1.0, float(H))))

        # Size proxy: bbox area relative to frame.
        areas = []
        for t in ts_sorted:
            bbox, _ = obs[t]
            x1, y1, x2, y2 = bbox
            areas.append(((x2 - x1) * (y2 - y1)) / max(1.0, float(W) * float(H)))
        size_score = float(np.average(areas)) if areas else 0.0

        return (
            w_motion * motion_score
            + w_bottom * bottom_score
            + w_size * size_score
            + w_presence * presence_ratio
        )

    best_id = max(track_obs.keys(), key=score_track)
    best_obs = track_obs[best_id]

    bboxes_xyxy = np.full((T, 4), np.nan, dtype=float)
    bbox_confs = np.full((T,), np.nan, dtype=float)
    for t, (bbox, conf) in best_obs.items():
        bboxes_xyxy[t] = bbox
        bbox_confs[t] = conf

    selected = {
        "track_id": best_id,
        "frame_indices": list(frame_indices),
        "frame_size": {"width": int(W), "height": int(H)},
        # Per-frame bbox aligned to the loaded `frames` sequence; NaN if missing.
        "bboxes_xyxy": bboxes_xyxy,
        "bbox_confs": bbox_confs,
    }
    return selected

