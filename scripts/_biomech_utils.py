"""
Shared biomechanical utilities for all component extractor scripts.
COCO-17 keypoint conventions, angle/distance helpers, aggregation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# COCO-17 keypoint indices
# ---------------------------------------------------------------------------
NOSE           = 0
LEFT_EYE       = 1
RIGHT_EYE      = 2
LEFT_EAR       = 3
RIGHT_EAR      = 4
LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW     = 7
RIGHT_ELBOW    = 8
LEFT_WRIST     = 9
RIGHT_WRIST    = 10
LEFT_HIP       = 11
RIGHT_HIP      = 12
LEFT_KNEE      = 13
RIGHT_KNEE     = 14
LEFT_ANKLE     = 15
RIGHT_ANKLE    = 16

KP_CONF_THRESH = 0.25


# ---------------------------------------------------------------------------
# Keypoint helpers
# ---------------------------------------------------------------------------

def kpt(
    frame_kpts: np.ndarray | None,
    idx: int,
    conf_thresh: float = KP_CONF_THRESH,
) -> np.ndarray | None:
    """Return (x, y) for keypoint idx if confidence >= thresh, else None."""
    if frame_kpts is None or frame_kpts.ndim < 2 or idx >= frame_kpts.shape[0]:
        return None
    x, y, c = float(frame_kpts[idx, 0]), float(frame_kpts[idx, 1]), float(frame_kpts[idx, 2])
    return np.array([x, y], dtype=np.float64) if c >= conf_thresh else None


def midpoint(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    return (a + b) * 0.5 if a is not None and b is not None else None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex b between rays b→a and b→c."""
    ba = a - b
    bc = c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def torso_length_px(
    frame_kpts: np.ndarray | None,
    conf_thresh: float = KP_CONF_THRESH,
) -> float | None:
    """Mid-shoulder to mid-hip distance in pixels (body scale normalizer)."""
    ls = kpt(frame_kpts, LEFT_SHOULDER,  conf_thresh)
    rs = kpt(frame_kpts, RIGHT_SHOULDER, conf_thresh)
    lh = kpt(frame_kpts, LEFT_HIP,  conf_thresh)
    rh = kpt(frame_kpts, RIGHT_HIP, conf_thresh)
    ms = midpoint(ls, rs)
    mh = midpoint(lh, rh)
    if ms is None or mh is None:
        return None
    d = float(np.linalg.norm(ms - mh))
    return d if d > 2.0 else None


def median_torso_px(
    kpts_window: list[np.ndarray | None],
    conf_thresh: float = KP_CONF_THRESH,
) -> float | None:
    lengths = [torso_length_px(k, conf_thresh) for k in kpts_window]
    clean = [t for t in lengths if t is not None]
    return float(np.median(clean)) if clean else None


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate(
    values: list[float | None],
    at_poc: float | None = None,
) -> dict:
    """Compute mean/std/min/max/p10/p90 over non-None finite values."""
    clean = [v for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return {k: None for k in ("mean", "std", "min", "max", "p10", "p90", "at_poc")}
    arr = np.array(clean, dtype=np.float64)
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "p10":    float(np.percentile(arr, 10)),
        "p90":    float(np.percentile(arr, 90)),
        "at_poc": float(at_poc) if at_poc is not None and np.isfinite(float(at_poc)) else None,
    }


def frac_true(flags: list[bool | None]) -> float | None:
    clean = [v for v in flags if v is not None]
    return float(sum(clean) / len(clean)) if clean else None


def available_kp_frac(
    kpts_window: list[np.ndarray | None],
    conf_thresh: float = KP_CONF_THRESH,
) -> float:
    n = len(kpts_window)
    if n == 0:
        return 0.0
    valid = sum(
        1 for k in kpts_window
        if k is not None and np.any(k[:, 2] >= conf_thresh)
    )
    return float(valid / n)


# ---------------------------------------------------------------------------
# Pipeline helpers (shared tackle-detection pass)
# ---------------------------------------------------------------------------

def _reset_yolo_tracker(model) -> None:
    """
    Reset the YOLO tracker state between videos.
    When a single model instance is reused across multiple videos with
    persist=True, stale track state from the previous video contaminates
    the next one — first frame of video N+1 never gets clean IDs.
    """
    try:
        if hasattr(model, "predictor") and model.predictor is not None:
            model.predictor = None
    except Exception:
        pass


def run_tackle_detection_pass(
    frames: list[np.ndarray],
    model,
    *,
    conf: float = 0.15,
    ball_conf: float = 0.15,
    device: str | None = None,
) -> tuple[dict[int, float], dict[int, int], list[dict]]:
    """
    Single-pass YOLO tracking over buffered frames.

    Resets the tracker before starting so that reusing one model instance
    across many videos stays clean.

    Returns (motion_sum, presence, frame_records) — same format used by
    tackle_bbox_pipeline._select_players_of_interest.
    motion_sum: cumulative displacement per track (may be 0 for tracks
                seen only once; use presence for "was ever seen" check).
    presence:   frame-count per track ID (reliable even on first frame).
    """
    from collections import defaultdict
    import numpy as np
    from tackle_bbox_pipeline import (
        _detect_people_and_ball_candidates,
        _box_center,
    )

    _reset_yolo_tracker(model)

    motion_sum: dict[int, float] = defaultdict(float)
    last_center: dict[int, np.ndarray] = {}
    presence: dict[int, int] = defaultdict(int)
    frame_records: list[dict] = []

    for frame_bgr in frames:
        person_ids, person_xyxy, person_confs, ball_xyxy, ball_confs = (
            _detect_people_and_ball_candidates(
                model,
                frame_bgr,
                person_conf=conf,
                ball_conf=ball_conf,
                device=device,
                persist_people=True,
                # Ball detection uses model.predict() on the SAME model instance,
                # which resets the ByteTrack predictor state and kills multi-person
                # tracking. Skip ball here; carrier/tackler still works from motion.
                include_ball=False,
            )
        )
        rec = {
            "person_ids":   person_ids,
            "person_xyxy":  person_xyxy,
            "person_confs": person_confs,
            "ball_xyxy":    ball_xyxy,
            "ball_confs":   ball_confs,
        }
        for tid, box in zip(person_ids, person_xyxy, strict=True):
            tid_i = int(tid)
            presence[tid_i] += 1
            c = _box_center(box)
            if tid_i in last_center:
                motion_sum[tid_i] += float(np.linalg.norm(c - last_center[tid_i]))
            last_center[tid_i] = c
        frame_records.append(rec)

    return dict(motion_sum), dict(presence), frame_records


def select_tackle_pair_and_roles(
    frames: list[np.ndarray],
    frame_records: list[dict],
    motion_sum: dict[int, float],
    presence: dict[int, int],
    frame_diag: float,
    *,
    conf: float = 0.15,
) -> tuple[int, int, int | None]:
    """
    Returns (tackler_tid, carrier_tid, poc_frame_idx).
    Wraps _select_players_of_interest + _choose_ballcarrier_from_possession.

    When motion_sum is thin (tracking IDs reset mid-clip), falls back to
    picking the two most-present tracks so the extractor still produces output.
    """
    import numpy as np
    from tackle_bbox_pipeline import (
        _select_players_of_interest,
        _choose_ballcarrier_from_possession,
    )

    # If motion_sum is too thin, pad it from presence so the scorer has data.
    effective_motion = dict(motion_sum)
    if len(effective_motion) < 2:
        for tid, cnt in sorted(presence.items(), key=lambda x: -x[1]):
            if tid not in effective_motion:
                effective_motion[tid] = 0.0
            if len(effective_motion) >= max(2, len(motion_sum)):
                break

    tid_a, tid_b, poc_idx, _ = _select_players_of_interest(
        frame_records,
        frame_diag,
        effective_motion,
        presence,
        person_conf_threshold=conf,
    )
    carrier_tid, tackler_tid, _ = _choose_ballcarrier_from_possession(
        frames, frame_records, tid_a, tid_b, frame_diag
    )
    return tackler_tid, carrier_tid, poc_idx


def extract_tackler_keypoints_window(
    frames: list[np.ndarray],
    frame_records: list[dict],
    tackler_tid: int,
    poc_idx: int,
    window_frames: int,
    pose_model,
    *,
    pose_conf: float = 0.18,
    device: str | None = None,
    stride: int = 1,
) -> list[np.ndarray | None]:
    """
    Run pose model on window [poc_idx, poc_idx + window_frames) frames,
    match keypoints to the tackler's bounding box, return per-frame kpts (17×3) or None.
    """
    from tackle_bbox_pipeline import (
        _best_pose_keypoints_for_box,
        POSE_MATCH_MIN_IOU,
    )

    end_idx = min(len(frames), poc_idx + window_frames)
    window_frame_indices = list(range(poc_idx, end_idx, stride))
    result: list[np.ndarray | None] = []

    for fi in window_frame_indices:
        frame_bgr = frames[fi]
        rec = frame_records[fi]

        # Find tackler's bounding box in this frame
        tackler_box: np.ndarray | None = None
        pids = rec["person_ids"]
        pxy  = rec["person_xyxy"]
        for i, tid in enumerate(pids):
            if int(tid) == tackler_tid:
                tackler_box = pxy[i].astype(np.float64)
                break

        if tackler_box is None:
            result.append(None)
            continue

        pkw: dict = {"conf": pose_conf, "verbose": False}
        if device:
            pkw["device"] = device
        pres = pose_model(frame_bgr, **pkw)[0]

        pose_boxes = (
            pres.boxes.xyxy.cpu().numpy() if pres.boxes is not None and len(pres.boxes) else np.empty((0, 4))
        )
        pose_kpts = (
            pres.keypoints.data.cpu().numpy()
            if pres.keypoints is not None and len(pres.keypoints)
            else np.empty((0, 17, 3))
        )

        kp = _best_pose_keypoints_for_box(pose_boxes, pose_kpts, tackler_box, POSE_MATCH_MIN_IOU)
        result.append(kp)

    return result


def video_id_from_path(path: str | Path) -> str:
    return Path(path).stem
