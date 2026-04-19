"""
Single-file pipeline: read a tackle clip, detect/track people with YOLO, infer
which figure is most likely the tackler (approach motion toward another player),
and write a video with a bounding box overlaid on that player.

Requires: pip install ultralytics  (also installs PyTorch)

Heuristic mode (default): COCO person + motion heuristic.
Trained mode: fine-tuned single-class `tackler` weights (see train_tackler_detector.py).

Examples:
  python tackle_bbox_pipeline.py path/to/clip.mp4 -o out_with_box.mp4
  python tackle_bbox_pipeline.py --all
  python tackle_bbox_pipeline.py clip.mp4 --mode trained --weights runs/detect/tackler/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: install with `pip install ultralytics` "
        "(from the project venv after `pip install -e .` if you add it to pyproject)."
    ) from e


CODEC_CANDIDATES = ["avc1", "mp4v"]
PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".wmv"}
DEFAULT_ALL_INPUT_DIR = PROJECT_ROOT / "raws"
DEFAULT_ALL_OUTPUT_DIR = PROJECT_ROOT / "finals" / "two_players"
PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32  # COCO sports ball
DEFAULT_WEIGHTS = "yolo11n.pt"

# Easy hot-swap for the pose model used by --mode top_motion.
# Example alternatives:
#TOP_MOTION_POSE_MODEL = "yolo11x-pose.pt"
TOP_MOTION_POSE_MODEL = "yolo26x-pose.pt"
#TOP_MOTION_POSE_MODEL = "yolo11n-pose.pt"
#TOP_MOTION_POSE_MODEL = "yolo11n.pt"

ANNOTATE_POSE_MODEL_NAME = True

DEFAULT_POSE_WEIGHTS = TOP_MOTION_POSE_MODEL
POSE_MATCH_MIN_IOU = 0.15
# Nose + shoulders, elbows, knees, ankles (main limbs); eyes/ears drawn when confident.
POSE_HIGHLIGHT_JOINTS: set[int] = {0, 5, 6, 7, 8, 13, 14, 15, 16}
POSE_FACE_JOINTS: set[int] = {1, 2, 3, 4}
COCO_SKELETON_POSE: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    for codec in CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for {path} with codecs {CODEC_CANDIDATES}")


def _box_center(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return float(inter / union) if union > 0 else 0.0


def _pair_score(
    box_i: np.ndarray, box_j: np.ndarray, frame_diag: float
) -> float:
    """Higher = more likely a interacting tackle pair (overlap + proximity)."""
    c_i = _box_center(box_i)
    c_j = _box_center(box_j)
    dist = float(np.linalg.norm(c_i - c_j)) / (frame_diag + 1e-6)
    overlap = _iou(box_i, box_j)
    proximity = max(0.0, 0.35 - dist)
    return overlap + 0.6 * proximity


def pick_interaction_pair(
    xyxy: np.ndarray,
    frame_diag: float,
    max_people: int = 4,
) -> tuple[int, int] | None:
    """Return indices of the two detections most likely in a tackle interaction."""
    n = len(xyxy)
    if n < 2:
        return None
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    order = np.argsort(-areas)
    top = order[: min(n, max_people)]
    best: tuple[int, int] | None = None
    best_s = -1.0
    diag = float(frame_diag) + 1e-6
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            i, j = int(top[a]), int(top[b])
            s = _pair_score(xyxy[i], xyxy[j], diag)
            if s > best_s:
                best_s = s
                best = (i, j)
    if best is None or best_s < 0.02:
        return None
    return best


def approach_score(
    track_id: int,
    self_center: np.ndarray,
    other_center: np.ndarray,
    history: dict[int, deque[tuple[float, float]]],
) -> float:
    """How much recent motion points from self toward the other player."""
    h = history.get(track_id)
    if h is None or len(h) < 2:
        return 0.0
    v = np.array(h[-1], dtype=np.float64) - np.array(h[-2], dtype=np.float64)
    toward = other_center - self_center
    n = float(np.linalg.norm(toward)) + 1e-6
    return float(np.dot(v, toward / n))


def choose_tackler_track(
    xyxy: np.ndarray,
    track_ids: np.ndarray,
    history: dict[int, deque[tuple[float, float]]],
    frame_diag: float,
) -> int | None:
    """
    Among tracked people, pick the track ID most likely to be the tackler.
    Uses the best interacting pair and compares approach motion toward the other.
    """
    n = len(xyxy)
    if n == 0:
        return None
    if n == 1:
        return int(track_ids[0])
    pair = pick_interaction_pair(xyxy, frame_diag)
    if pair is None:
        return None

    i, j = pair
    tid_i, tid_j = int(track_ids[i]), int(track_ids[j])
    c_i = _box_center(xyxy[i])
    c_j = _box_center(xyxy[j])
    si = approach_score(tid_i, c_i, c_j, history)
    sj = approach_score(tid_j, c_j, c_i, history)
    if abs(si - sj) < 0.5:
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        return tid_i if areas[i] < areas[j] else tid_j
    return tid_i if si > sj else tid_j


def draw_tackler_box(
    frame_bgr: np.ndarray,
    xyxy: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    label: str = "Tackler",
) -> None:
    x1, y1, x2, y2 = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        frame_bgr,
        label,
        (x1, max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def draw_player_box(
    frame_bgr: np.ndarray,
    xyxy: np.ndarray,
    *,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        frame_bgr,
        label,
        (x1, max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def _safe_argmax(values: np.ndarray) -> int | None:
    if len(values) == 0:
        return None
    return int(np.argmax(values))


def _track_activity_score(track_id: int, history: dict[int, deque[tuple[float, float]]]) -> float:
    """
    Activity score from recent center displacement magnitude.
    Higher means the player is moving more over the recent track window.
    """
    h = history.get(track_id)
    if h is None or len(h) < 2:
        return 0.0
    pts = np.array(h, dtype=np.float64)
    deltas = np.diff(pts, axis=0)
    step_dist = np.linalg.norm(deltas, axis=1)
    return float(np.mean(step_dist))


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def pick_active_interaction_pair(
    xyxy: np.ndarray,
    track_ids: np.ndarray,
    history: dict[int, deque[tuple[float, float]]],
    confs: np.ndarray,
    frame_diag: float,
) -> list[int]:
    """
    Pick a two-player pair that is both active and spatially interacting.
    This reduces sideline hijacking compared with pure activity ranking.
    Returns detection indices (not track IDs).
    """
    n = len(xyxy)
    if n == 0:
        return []
    if n == 1:
        return [0]

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    norm_areas = areas / (np.max(areas) + 1e-6)
    activity = np.array(
        [_track_activity_score(int(tid), history) for tid in track_ids],
        dtype=np.float64,
    )

    best_pair: tuple[int, int] | None = None
    best_score = -1e9
    diag = frame_diag + 1e-6
    for i in range(n):
        for j in range(i + 1, n):
            ci = _box_center(xyxy[i])
            cj = _box_center(xyxy[j])
            dist_norm = float(np.linalg.norm(ci - cj)) / diag
            proximity = max(0.0, 0.35 - dist_norm)
            clarity = confs[i] * norm_areas[i] + confs[j] * norm_areas[j]
            score = (activity[i] + activity[j]) + 1.4 * proximity + 0.2 * clarity
            if score > best_score:
                best_score = score
                best_pair = (i, j)

    if best_pair is None:
        if len(xyxy) == 0:
            return []
        if len(xyxy) == 1:
            return [0]
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        scores = confs * (areas / (np.max(areas) + 1e-6))
        order = np.argsort(-scores)
        return [int(i) for i in order[:2]]
    return [int(best_pair[0]), int(best_pair[1])]


def pick_carrier_idx_from_ball(
    person_xyxy: np.ndarray,
    person_confs: np.ndarray,
    ball_xyxy: np.ndarray | None,
    frame_diag: float,
) -> int | None:
    """
    Choose likely ball carrier as the person closest to the detected football.
    If no ball is detected, fallback to most confident prominent person.
    """
    if len(person_xyxy) == 0:
        return None
    if ball_xyxy is None:
        areas = (person_xyxy[:, 2] - person_xyxy[:, 0]) * (person_xyxy[:, 3] - person_xyxy[:, 1])
        scores = person_confs * (areas / (np.max(areas) + 1e-6))
        return _safe_argmax(scores)

    b_center = _box_center(ball_xyxy)
    p_centers = np.array([_box_center(b) for b in person_xyxy], dtype=np.float64)
    dists = np.linalg.norm(p_centers - b_center[None, :], axis=1) / (frame_diag + 1e-6)
    # Confidence bonus helps if two players are similarly close.
    scores = -dists + 0.2 * person_confs
    return _safe_argmax(scores)


def pick_tackler_idx_against_carrier(
    person_xyxy: np.ndarray,
    person_track_ids: np.ndarray,
    carrier_idx: int,
    history: dict[int, deque[tuple[float, float]]],
    frame_diag: float,
    person_confs: np.ndarray,
) -> int | None:
    """
    Choose likely tackler as non-carrier player with strong approach + proximity to carrier.
    """
    n = len(person_xyxy)
    if n <= 1 or carrier_idx < 0 or carrier_idx >= n:
        return None

    c_box = person_xyxy[carrier_idx]
    c_center = _box_center(c_box)
    c_tid = int(person_track_ids[carrier_idx])

    best_idx: int | None = None
    best_score = -1e9
    for i in range(n):
        if i == carrier_idx:
            continue
        t_tid = int(person_track_ids[i])
        t_center = _box_center(person_xyxy[i])

        # Positive when candidate moves toward carrier.
        toward = c_center - t_center
        toward_n = toward / (np.linalg.norm(toward) + 1e-6)
        h = history.get(t_tid)
        approach = 0.0
        if h is not None and len(h) >= 2:
            v = np.array(h[-1], dtype=np.float64) - np.array(h[-2], dtype=np.float64)
            approach = float(np.dot(v, toward_n))

        proximity = max(0.0, 0.35 - (_distance(t_center, c_center) / (frame_diag + 1e-6)))
        overlap = _iou(person_xyxy[i], c_box)
        # mild penalty if "candidate" is actually moving away
        score = 1.1 * approach + 1.6 * proximity + 1.2 * overlap + 0.15 * float(person_confs[i])
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def _upper_body_xyxy(xyxy: np.ndarray) -> np.ndarray:
    """Upper ~55% of the person box (torso / hands region proxy)."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    h = max(1.0, y2 - y1)
    y_split = y1 + 0.45 * h
    return np.array([x1, y1, x2, y_split], dtype=np.float64)


def _ball_hands_association_score(person_xyxy: np.ndarray, ball_xyxy: np.ndarray) -> float:
    """How strongly a ball detection associates with a person's upper body (hands proxy)."""
    upper = _upper_body_xyxy(person_xyxy)
    iou_upper = _iou(upper, ball_xyxy)
    iou_full = _iou(person_xyxy, ball_xyxy)
    bc = _box_center(ball_xyxy)
    x1, y1, x2, y2 = [float(v) for v in person_xyxy]
    inside = x1 <= bc[0] <= x2 and y1 <= bc[1] <= y2
    return float(max(iou_upper, iou_full * 0.85) + (0.15 if inside else 0.0))


def _football_appearance_score(
    frame_bgr: np.ndarray,
    ball_xyxy: np.ndarray,
    person_xyxy: np.ndarray,
    frame_diag: float,
    det_conf: float,
) -> float:
    """
    Score detections by football-like cues to reject cone false positives.
    Prefers small, oval, brown-ish objects near players.
    """
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in ball_xyxy]
    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(0, min(w_img, x2))
    y2 = max(0, min(h_img, y2))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return -1.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    mean_h = float(np.mean(h))
    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))

    # Hard color gate: football should be mostly brown, not vivid orange/neon.
    brown_mask = (
        (h >= 7.0) & (h <= 20.0) &
        (s >= 45.0) & (s <= 185.0) &
        (v >= 22.0) & (v <= 150.0)
    )
    orange_mask = (
        (h >= 8.0) & (h <= 24.0) &
        (s >= 130.0) & (v >= 135.0)
    )
    brown_ratio = float(np.mean(brown_mask))
    orange_ratio = float(np.mean(orange_mask))
    if brown_ratio < 0.30 or orange_ratio > 0.10:
        return -1.0

    # Brown footballs are darker and less saturated than bright orange cones.
    hue_score = float(np.exp(-((mean_h - 14.0) ** 2) / (2.0 * (8.0 ** 2))))
    sat_score = float(np.exp(-((mean_s - 105.0) ** 2) / (2.0 * (50.0 ** 2))))
    val_score = float(np.exp(-((mean_v - 78.0) ** 2) / (2.0 * (30.0 ** 2))))
    color_score = 0.40 * hue_score + 0.20 * sat_score + 0.20 * val_score + 0.20 * brown_ratio

    aspect = bw / float(bh)
    shape_score = max(
        float(np.exp(-((aspect - 1.65) ** 2) / (2.0 * (0.45 ** 2)))),
        float(np.exp(-((aspect - 0.62) ** 2) / (2.0 * (0.18 ** 2)))),
    )
    area_frac = (bw * bh) / float(max(1, w_img * h_img))
    size_score = float(np.exp(-((area_frac - 0.0008) ** 2) / (2.0 * (0.0007 ** 2))))

    prox_score = 0.0
    hands_score = 0.0
    b_center = _box_center(ball_xyxy)
    if len(person_xyxy) > 0:
        min_norm_d = 1.0
        best_hands_assoc = 0.0
        for p in person_xyxy:
            p_center = _box_center(p)
            d = float(np.linalg.norm(b_center - p_center)) / (frame_diag + 1e-6)
            if d < min_norm_d:
                min_norm_d = d
            best_hands_assoc = max(
                best_hands_assoc,
                _ball_hands_association_score(p.astype(np.float64), ball_xyxy.astype(np.float64)),
            )
        prox_score = float(np.exp(-((min_norm_d - 0.10) ** 2) / (2.0 * (0.18 ** 2))))
        # 0.15+ typically means ball is plausibly on/near torso-upper-body (hands proxy).
        hands_score = float(np.clip(best_hands_assoc / 0.35, 0.0, 1.0))

        # Hard gate for hand-carry prior: reject detections far from any player's carry zone.
        if best_hands_assoc < 0.07 and min_norm_d > 0.23:
            return -1.0

    score = (
        0.14 * float(det_conf)
        + 0.34 * color_score
        + 0.16 * shape_score
        + 0.08 * size_score
        + 0.10 * prox_score
        + 0.30 * hands_score
    )

    # Strongly down-rank cone-like bright orange candidates.
    cone_like = 8.0 <= mean_h <= 24.0 and mean_s >= 115.0 and mean_v >= 120.0
    if cone_like:
        score -= 0.22
    return score


def _pick_best_football_box(
    frame_bgr: np.ndarray,
    ball_xyxy: np.ndarray,
    ball_confs: np.ndarray,
    person_xyxy: np.ndarray,
    frame_diag: float,
    min_score: float = 0.42,
) -> np.ndarray | None:
    if len(ball_xyxy) == 0:
        return None
    best_idx: int | None = None
    best_score = min_score
    for i, box in enumerate(ball_xyxy):
        s = _football_appearance_score(
            frame_bgr,
            box.astype(np.float64),
            person_xyxy.astype(np.float64),
            frame_diag,
            float(ball_confs[i]),
        )
        if s > best_score:
            best_score = s
            best_idx = i
    if best_idx is None:
        return None
    return ball_xyxy[best_idx]


def _best_pose_keypoints_for_box(
    pose_boxes: np.ndarray,
    pose_kpts: np.ndarray,
    target_xyxy: np.ndarray,
    min_iou: float = POSE_MATCH_MIN_IOU,
) -> np.ndarray | None:
    if len(pose_boxes) == 0 or len(pose_kpts) == 0:
        return None
    t = target_xyxy.astype(np.float64)
    best_kp: np.ndarray | None = None
    best_score = min_iou
    for i, pb in enumerate(pose_boxes):
        s = _iou(t, pb.astype(np.float64))
        if s > best_score:
            best_score = s
            best_kp = pose_kpts[i]
    return best_kp


def _draw_pose_skeleton_on_player(
    frame_bgr: np.ndarray,
    kpts: np.ndarray,
    *,
    line_color: tuple[int, int, int],
    joint_color: tuple[int, int, int],
    conf_thresh: float = 0.25,
) -> None:
    kp_px = [(int(x), int(y), float(c)) for x, y, c in kpts]
    for i1, i2 in COCO_SKELETON_POSE:
        x1, y1, c1 = kp_px[i1]
        x2, y2, c2 = kp_px[i2]
        if c1 < conf_thresh or c2 < conf_thresh:
            continue
        cv2.line(frame_bgr, (x1, y1), (x2, y2), line_color, 2, lineType=cv2.LINE_AA)
    for j, (x, y, c) in enumerate(kp_px):
        if c < conf_thresh:
            continue
        if j in POSE_FACE_JOINTS:
            r = 3 if c >= 0.4 else 2
            cv2.circle(frame_bgr, (x, y), r, (200, 200, 255), -1, lineType=cv2.LINE_AA)
        elif j in POSE_HIGHLIGHT_JOINTS:
            cv2.circle(frame_bgr, (x, y), 6, joint_color, -1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(frame_bgr, (x, y), 3, joint_color, -1, lineType=cv2.LINE_AA)


def _head_rotation_text(kpts: np.ndarray, *, conf_face: float = 0.32, conf_body: float = 0.28) -> str | None:
    """
    Prefer eye-line roll when both eyes visible; else ear+nose hint; else nose + mid-shoulder facing.
    """
    le, re, no = kpts[1], kpts[2], kpts[0]
    ls, rs = kpts[5], kpts[6]
    if le[2] >= conf_face and re[2] >= conf_face:
        ang = float(np.degrees(np.arctan2(le[1] - re[1], le[0] - re[0])))
        return f"head roll {ang:.0f}deg (eyes)"
    if no[2] >= conf_face and (le[2] >= conf_face or re[2] >= conf_face):
        return "head: eyes+nose"
    if no[2] >= conf_face and (kpts[3][2] >= conf_face or kpts[4][2] >= conf_face):
        return "head: ear+nose"
    if no[2] >= conf_body and ls[2] >= conf_body and rs[2] >= conf_body:
        mid_sh = (ls[:2] + rs[:2]) / 2.0
        vec = no[:2] - mid_sh
        if float(np.linalg.norm(vec)) > 1e-3:
            ang = float(np.degrees(np.arctan2(vec[0], vec[1])))
            return f"head facing {ang:.0f}deg (nose-shldr)"
    return None


def run_top_motion_carrier_tackler_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    weights: str,
    conf: float,
    device: str | None,
    max_frames: int = 4500,
    pose_weights: str | None = DEFAULT_POSE_WEIGHTS,
    pose_conf: float = 0.18,
) -> None:
    """
    Two-stage pipeline on a single tracking pass (frames buffered):
    1) Accumulate per-track motion over the clip; keep the two tracks with highest motion.
    2) Among those two, label Ball Carrier vs Tackler from football detections near upper body.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_diag = float(np.hypot(width, height))

    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(frame_bgr)
    cap.release()

    if not frames:
        raise SystemExit(f"No frames read from {input_path}")

    model = YOLO(weights)
    track_kwargs: dict = {
        "persist": True,
        "classes": [PERSON_CLASS_ID, BALL_CLASS_ID],
        "conf": conf,
        "verbose": False,
    }
    if device:
        track_kwargs["device"] = device

    motion_sum: dict[int, float] = defaultdict(float)
    last_center: dict[int, np.ndarray] = {}
    presence: dict[int, int] = defaultdict(int)
    frame_records: list[dict] = []

    for frame_bgr in frames:
        results = model.track(frame_bgr, **track_kwargs)[0]
        boxes = results.boxes
        rec: dict = {
            "person_ids": np.array([], dtype=np.int64),
            "person_xyxy": np.empty((0, 4), dtype=np.float64),
            "person_confs": np.array([], dtype=np.float64),
            "balls": [],
        }
        if boxes is not None and len(boxes) > 0:
            xyxy_all = boxes.xyxy.cpu().numpy()
            confs_all = boxes.conf.cpu().numpy()
            cls_all = boxes.cls.cpu().numpy().astype(np.int64)
            ids_t = boxes.id
            if ids_t is not None:
                ids_all = ids_t.cpu().numpy().astype(np.int64)
                person_mask = cls_all == PERSON_CLASS_ID
                ball_mask = cls_all == BALL_CLASS_ID
                person_xyxy = xyxy_all[person_mask]
                person_confs = confs_all[person_mask]
                person_ids = ids_all[person_mask]
                ball_xyxy = xyxy_all[ball_mask]
                rec["person_ids"] = person_ids
                rec["person_xyxy"] = person_xyxy
                rec["person_confs"] = person_confs
                best_ball_box = _pick_best_football_box(
                    frame_bgr,
                    ball_xyxy,
                    confs_all[ball_mask],
                    person_xyxy,
                    frame_diag,
                )
                rec["balls"] = [best_ball_box.copy()] if best_ball_box is not None else []

                for tid, box in zip(person_ids, person_xyxy, strict=True):
                    tid_i = int(tid)
                    presence[tid_i] += 1
                    c = _box_center(box)
                    if tid_i in last_center:
                        motion_sum[tid_i] += float(np.linalg.norm(c - last_center[tid_i]))
                    last_center[tid_i] = c

        frame_records.append(rec)

    ordered_motion = sorted(motion_sum.items(), key=lambda kv: -kv[1])
    top_ids: list[int] = [int(t) for t, _ in ordered_motion[:2]]

    if len(top_ids) < 2:
        extra = sorted(
            [(t, c) for t, c in presence.items() if t not in top_ids],
            key=lambda kv: -kv[1],
        )
        for t, _ in extra:
            top_ids.append(int(t))
            if len(top_ids) >= 2:
                break

    if len(top_ids) < 2:
        raise SystemExit("Could not find two person tracks in video.")
    tid_a, tid_b = top_ids[0], top_ids[1]
    if tid_a == tid_b:
        for t, _ in sorted(presence.items(), key=lambda kv: -kv[1]):
            if int(t) != tid_a:
                tid_b = int(t)
                break

    carrier_scores: dict[int, float] = defaultdict(float)
    for rec in frame_records:
        pids = rec["person_ids"]
        if len(pids) == 0:
            continue
        pxy = rec["person_xyxy"]
        tid_to_i = {int(t): i for i, t in enumerate(pids)}
        for ball in rec["balls"]:
            for tid in (tid_a, tid_b):
                if tid not in tid_to_i:
                    continue
                carrier_scores[tid] += _ball_hands_association_score(
                    pxy[tid_to_i[tid]], ball
                )

    score_a = float(carrier_scores.get(tid_a, 0.0))
    score_b = float(carrier_scores.get(tid_b, 0.0))
    if score_a == 0.0 and score_b == 0.0:
        carrier_tid, tackler_tid = tid_a, tid_b
    elif score_a >= score_b:
        carrier_tid, tackler_tid = tid_a, tid_b
    else:
        carrier_tid, tackler_tid = tid_b, tid_a

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_writer(output_path, fps, width, height)
    last_box: dict[int, np.ndarray | None] = {carrier_tid: None, tackler_tid: None}
    miss_ct: dict[int, int] = {carrier_tid: 0, tackler_tid: 0}
    hold_max = 10

    pose_model: YOLO | None = None
    if pose_weights:
        try:
            pose_model = YOLO(pose_weights)
        except FileNotFoundError as e:
            raise SystemExit(
                "Pose model file not found: "
                f"{pose_weights}. Put the weights file at that path, set "
                "TOP_MOTION_POSE_MODEL to an available model, or run with --no-pose."
            ) from e

    try:
        for frame_bgr, rec in zip(frames, frame_records, strict=True):
            pids = rec["person_ids"]
            pxy = rec["person_xyxy"]
            tid_to_i = {int(t): i for i, t in enumerate(pids)} if len(pids) else {}

            pose_boxes = np.empty((0, 4))
            pose_kpts = np.empty((0, 17, 3))
            if pose_model is not None:
                pkw: dict = {"conf": pose_conf, "verbose": False}
                if device:
                    pkw["device"] = device
                pres = pose_model(frame_bgr, **pkw)[0]
                if pres.boxes is not None and len(pres.boxes):
                    pose_boxes = pres.boxes.xyxy.cpu().numpy()
                if pres.keypoints is not None and len(pres.keypoints):
                    pose_kpts = pres.keypoints.data.cpu().numpy()

            for ball in rec["balls"]:
                draw_player_box(frame_bgr, ball, color=(0, 255, 255), label="Football")

            for role_tid, label, color, sk_color in (
                (carrier_tid, "Ball Carrier", (255, 0, 0), (0, 220, 120)),
                (tackler_tid, "Tackler", (0, 0, 255), (100, 200, 255)),
            ):
                box_for_pose: np.ndarray | None = None
                if role_tid in tid_to_i:
                    box = pxy[tid_to_i[role_tid]]
                    draw_player_box(frame_bgr, box, color=color, label=label)
                    last_box[role_tid] = box.copy()
                    miss_ct[role_tid] = 0
                    box_for_pose = box
                elif last_box.get(role_tid) is not None and miss_ct[role_tid] < hold_max:
                    miss_ct[role_tid] += 1
                    hb = last_box[role_tid]
                    draw_player_box(
                        frame_bgr,
                        hb,
                        color=color,
                        label=f"{label} (hold)",
                    )
                    box_for_pose = hb
                else:
                    miss_ct[role_tid] += 1

                if (
                    pose_model is not None
                    and box_for_pose is not None
                    and len(pose_boxes) > 0
                    and len(pose_kpts) > 0
                ):
                    kp = _best_pose_keypoints_for_box(
                        pose_boxes, pose_kpts, box_for_pose
                    )
                    if kp is not None:
                        _draw_pose_skeleton_on_player(
                            frame_bgr,
                            kp,
                            line_color=sk_color,
                            joint_color=sk_color,
                            conf_thresh=pose_conf,
                        )
                        ht = _head_rotation_text(kp)
                        if ht:
                            bx1 = int(box_for_pose[0])
                            by1 = int(box_for_pose[1])
                            cv2.putText(
                                frame_bgr,
                                ht[:56],
                                (bx1, min(height - 8, by1 + 78)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                sk_color,
                                1,
                                lineType=cv2.LINE_AA,
                            )

            if pose_model is not None and ANNOTATE_POSE_MODEL_NAME:
                pose_label = f"pose model: {Path(str(pose_weights)).name}"
                cv2.putText(
                    frame_bgr,
                    pose_label[:80],
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    frame_bgr,
                    pose_label[:80],
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (20, 20, 20),
                    1,
                    lineType=cv2.LINE_AA,
                )

            writer.write(frame_bgr)
    finally:
        writer.release()

    print(
        f"Top motion IDs: {tid_a}, {tid_b} | carrier={carrier_tid} tackler={tackler_tid} "
        f"| wrote {len(frames)} frames to {output_path}"
    )


def run_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    weights: str,
    conf: float,
    device: str | None,
) -> None:
    model = YOLO(weights)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_writer(output_path, fps, width, height)
    frame_diag = float(np.hypot(width, height))

    history: dict[int, deque[tuple[float, float]]] = {}
    locked_carrier_tid: int | None = None
    locked_tackler_tid: int | None = None
    last_carrier_box: np.ndarray | None = None
    last_tackler_box: np.ndarray | None = None
    carrier_miss = 0
    tackler_miss = 0
    lock_hold_frames = 8
    max_hold_frames = 8
    frame_i = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            kwargs: dict = {
                "persist": True,
                "classes": [PERSON_CLASS_ID, BALL_CLASS_ID],
                "conf": conf,
                "verbose": False,
            }
            if device:
                kwargs["device"] = device

            results = model.track(frame_bgr, **kwargs)[0]
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                writer.write(frame_bgr)
                frame_i += 1
                continue

            xyxy_all = boxes.xyxy.cpu().numpy()
            confs_all = boxes.conf.cpu().numpy()
            cls_all = boxes.cls.cpu().numpy().astype(np.int64)
            ids_t = boxes.id
            if ids_t is None:
                writer.write(frame_bgr)
                frame_i += 1
                continue

            ids_all = ids_t.cpu().numpy().astype(np.int64)
            person_mask = cls_all == PERSON_CLASS_ID
            ball_mask = cls_all == BALL_CLASS_ID
            person_xyxy = xyxy_all[person_mask]
            person_confs = confs_all[person_mask]
            person_ids = ids_all[person_mask]
            ball_xyxy = xyxy_all[ball_mask]
            ball_confs = confs_all[ball_mask]

            if len(person_xyxy) == 0:
                writer.write(frame_bgr)
                frame_i += 1
                continue

            for tid, box in zip(person_ids, person_xyxy, strict=True):
                tid = int(tid)
                if tid not in history:
                    history[tid] = deque(maxlen=10)
                c = _box_center(box)
                history[tid].append((float(c[0]), float(c[1])))

            best_ball_box = _pick_best_football_box(
                frame_bgr,
                ball_xyxy,
                ball_confs,
                person_xyxy,
                frame_diag,
            )

            tid_to_idx = {int(t): i for i, t in enumerate(person_ids)}

            if locked_carrier_tid is None:
                c_idx = pick_carrier_idx_from_ball(person_xyxy, person_confs, best_ball_box, frame_diag)
                if c_idx is not None:
                    locked_carrier_tid = int(person_ids[c_idx])

            if locked_carrier_tid in tid_to_idx:
                carrier_idx = tid_to_idx[locked_carrier_tid]
                carrier_box = person_xyxy[carrier_idx]
                last_carrier_box = carrier_box.copy()
                carrier_miss = 0
            else:
                carrier_idx = None
                carrier_miss += 1
                if carrier_miss > lock_hold_frames:
                    locked_carrier_tid = None
                    last_carrier_box = None

            if locked_carrier_tid is not None and locked_carrier_tid in tid_to_idx:
                carrier_idx_now = tid_to_idx[locked_carrier_tid]
                tackler_idx = pick_tackler_idx_against_carrier(
                    person_xyxy,
                    person_ids,
                    carrier_idx_now,
                    history,
                    frame_diag,
                    person_confs,
                )
                if tackler_idx is not None:
                    locked_tackler_tid = int(person_ids[tackler_idx])

            if locked_tackler_tid in tid_to_idx:
                t_idx = tid_to_idx[locked_tackler_tid]
                t_box = person_xyxy[t_idx]
                last_tackler_box = t_box.copy()
                tackler_miss = 0
            else:
                tackler_miss += 1
                if tackler_miss > lock_hold_frames:
                    locked_tackler_tid = None
                    last_tackler_box = None

            if best_ball_box is not None:
                draw_player_box(frame_bgr, best_ball_box, color=(0, 255, 255), label="Football")

            if locked_carrier_tid is not None and locked_carrier_tid in tid_to_idx:
                draw_player_box(
                    frame_bgr,
                    person_xyxy[tid_to_idx[locked_carrier_tid]],
                    color=(255, 0, 0),
                    label="Ball Carrier",
                )
            elif last_carrier_box is not None and carrier_miss <= max_hold_frames:
                draw_player_box(frame_bgr, last_carrier_box, color=(255, 0, 0), label="Ball Carrier (hold)")

            if locked_tackler_tid is not None and locked_tackler_tid in tid_to_idx:
                draw_player_box(
                    frame_bgr,
                    person_xyxy[tid_to_idx[locked_tackler_tid]],
                    color=(0, 0, 255),
                    label="Tackler",
                )
            elif last_tackler_box is not None and tackler_miss <= max_hold_frames:
                draw_player_box(frame_bgr, last_tackler_box, color=(0, 0, 255), label="Tackler (hold)")

            writer.write(frame_bgr)
            frame_i += 1
    finally:
        cap.release()
        writer.release()

    print(f"Wrote {frame_i} frames to {output_path}")


def run_trained_tackler_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    weights: str,
    conf: float,
    device: str | None,
) -> None:
    """Draw the highest-confidence tackler box from a single-class fine-tuned model."""
    model = YOLO(weights)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_writer(output_path, fps, width, height)
    frame_i = 0
    history: dict[int, deque[tuple[float, float]]] = {}
    frame_diag = float(np.hypot(width, height))
    locked_carrier_tid: int | None = None
    locked_tackler_tid: int | None = None
    last_carrier_box: np.ndarray | None = None
    last_tackler_box: np.ndarray | None = None
    carrier_miss = 0
    tackler_miss = 0
    max_hold_frames = 8

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            kwargs: dict = {
                "persist": True,
                "classes": [PERSON_CLASS_ID, BALL_CLASS_ID],
                "conf": conf,
                "verbose": False,
            }
            if device:
                kwargs["device"] = device

            results = model.track(frame_bgr, **kwargs)[0]
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                writer.write(frame_bgr)
                frame_i += 1
                continue

            xyxy_all = boxes.xyxy.cpu().numpy()
            confs_all = boxes.conf.cpu().numpy()
            cls_all = boxes.cls.cpu().numpy().astype(np.int64)
            ids_t = boxes.id
            if ids_t is None:
                writer.write(frame_bgr)
                frame_i += 1
                continue

            ids_all = ids_t.cpu().numpy().astype(np.int64)
            person_mask = cls_all == PERSON_CLASS_ID
            ball_mask = cls_all == BALL_CLASS_ID
            person_xyxy = xyxy_all[person_mask]
            person_confs = confs_all[person_mask]
            person_ids = ids_all[person_mask]
            ball_xyxy = xyxy_all[ball_mask]
            ball_confs = confs_all[ball_mask]

            if len(person_xyxy) == 0:
                writer.write(frame_bgr)
                frame_i += 1
                continue

            for tid, box in zip(person_ids, person_xyxy, strict=True):
                tid = int(tid)
                if tid not in history:
                    history[tid] = deque(maxlen=10)
                c = _box_center(box)
                history[tid].append((float(c[0]), float(c[1])))

            best_ball_box = _pick_best_football_box(
                frame_bgr,
                ball_xyxy,
                ball_confs,
                person_xyxy,
                frame_diag,
            )
            tid_to_idx = {int(t): i for i, t in enumerate(person_ids)}

            if locked_carrier_tid is None:
                c_idx = pick_carrier_idx_from_ball(person_xyxy, person_confs, best_ball_box, frame_diag)
                if c_idx is not None:
                    locked_carrier_tid = int(person_ids[c_idx])

            if locked_carrier_tid in tid_to_idx:
                carrier_idx = tid_to_idx[locked_carrier_tid]
                carrier_box = person_xyxy[carrier_idx]
                last_carrier_box = carrier_box.copy()
                carrier_miss = 0
            else:
                carrier_miss += 1
                if carrier_miss > max_hold_frames:
                    locked_carrier_tid = None
                    last_carrier_box = None

            if locked_carrier_tid is not None and locked_carrier_tid in tid_to_idx:
                carrier_idx_now = tid_to_idx[locked_carrier_tid]
                tackler_idx = pick_tackler_idx_against_carrier(
                    person_xyxy,
                    person_ids,
                    carrier_idx_now,
                    history,
                    frame_diag,
                    person_confs,
                )
                if tackler_idx is not None:
                    locked_tackler_tid = int(person_ids[tackler_idx])

            if locked_tackler_tid in tid_to_idx:
                t_idx = tid_to_idx[locked_tackler_tid]
                t_box = person_xyxy[t_idx]
                last_tackler_box = t_box.copy()
                tackler_miss = 0
            else:
                tackler_miss += 1
                if tackler_miss > max_hold_frames:
                    locked_tackler_tid = None
                    last_tackler_box = None

            if best_ball_box is not None:
                draw_player_box(frame_bgr, best_ball_box, color=(0, 255, 255), label="Football")

            if locked_carrier_tid is not None and locked_carrier_tid in tid_to_idx:
                draw_player_box(
                    frame_bgr,
                    person_xyxy[tid_to_idx[locked_carrier_tid]],
                    color=(255, 0, 0),
                    label="Ball Carrier",
                )
            elif last_carrier_box is not None and carrier_miss <= max_hold_frames:
                draw_player_box(frame_bgr, last_carrier_box, color=(255, 0, 0), label="Ball Carrier (hold)")

            if locked_tackler_tid is not None and locked_tackler_tid in tid_to_idx:
                draw_player_box(
                    frame_bgr,
                    person_xyxy[tid_to_idx[locked_tackler_tid]],
                    color=(0, 0, 255),
                    label="Tackler",
                )
            elif last_tackler_box is not None and tackler_miss <= max_hold_frames:
                draw_player_box(frame_bgr, last_tackler_box, color=(0, 0, 255), label="Tackler (hold)")

            writer.write(frame_bgr)
            frame_i += 1
    finally:
        cap.release()
        writer.release()

    print(f"Wrote {frame_i} frames to {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay a tackler bounding box on football tackle video "
        "(COCO person + heuristic, or fine-tuned tackler detector)."
    )
    p.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input video path (.mp4, .mov, ...). Optional when --all is used.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Process all videos in the input directory (default: raws/).",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_ALL_INPUT_DIR,
        help="Directory to scan when --all is used (default: raws/).",
    )
    p.add_argument(
        "--mode",
        choices=("heuristic", "trained", "top_motion"),
        default=None,
        help="heuristic: ball+carrier+tackler (streaming); trained: single-class tackler; "
        "top_motion: two highest-motion tracks + ball-in-hands carrier (default with --all)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output video path for one input, or output directory with --all "
            "(default with --all: finals/two_players/)."
        ),
    )
    p.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="COCO detector checkpoint (default yolo11n.pt); trained mode: your runs/.../best.pt",
    )
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument(
        "--device",
        default=None,
        help="torch device, e.g. mps, cuda:0, cpu (default: auto)",
    )
    p.add_argument(
        "--pose-weights",
        default=DEFAULT_POSE_WEIGHTS,
        help="Pose model for top_motion (skeleton); empty string disables unless --no-pose",
    )
    p.add_argument(
        "--pose-conf",
        type=float,
        default=0.18,
        help="Keypoint confidence threshold for drawing pose in top_motion",
    )
    p.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable pose overlay for top_motion",
    )
    return p.parse_args(argv)


def _filename_tag(value: str) -> str:
    tag = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in value.strip()
    ).strip("._-")
    return tag or "unknown"


def _effective_pose_weights(args: argparse.Namespace) -> str | None:
    if args.no_pose:
        return None
    pws = (args.pose_weights or "").strip()
    return pws if pws else None


def _default_output_for(
    input_path: Path,
    *,
    mode: str,
    output_dir: Path | None = None,
    pose_weights: str | None = None,
) -> Path:
    if mode == "top_motion":
        pose_tag = (
            _filename_tag(Path(str(pose_weights)).stem)
            if pose_weights
            else "no_pose"
        )
        suffix = f"_top_motion_{pose_tag}.mp4"
    else:
        suffix = "_tackler_box.mp4"
    name = f"{input_path.stem}{suffix}"
    return (output_dir / name) if output_dir is not None else input_path.with_name(name)


def _run_one_video(args: argparse.Namespace, inp: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "trained":
        run_trained_tackler_pipeline(
            inp,
            out,
            weights=args.weights,
            conf=args.conf,
            device=args.device,
        )
    elif args.mode == "top_motion":
        pw = _effective_pose_weights(args)
        run_top_motion_carrier_tackler_pipeline(
            inp,
            out,
            weights=args.weights,
            conf=args.conf,
            device=args.device,
            pose_weights=pw,
            pose_conf=args.pose_conf,
        )
    else:
        run_pipeline(
            inp,
            out,
            weights=args.weights,
            conf=args.conf,
            device=args.device,
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.mode = args.mode or ("top_motion" if args.all else "heuristic")
    if args.all:
        input_dir = args.input_dir.expanduser().resolve()
        if not input_dir.is_dir():
            sys.exit(f"Input directory not found: {input_dir}")

        videos = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
        )
        if not videos:
            supported = ", ".join(sorted(VIDEO_SUFFIXES))
            sys.exit(f"No videos found in {input_dir} (supported: {supported})")

        output_dir = (
            args.output.expanduser().resolve()
            if args.output is not None
            else DEFAULT_ALL_OUTPUT_DIR
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Found {len(videos)} video(s) in {input_dir}")
        print(f"Output directory: {output_dir}")
        failures = 0
        pose_weights = (
            _effective_pose_weights(args) if args.mode == "top_motion" else None
        )
        for idx, inp in enumerate(videos, start=1):
            out = _default_output_for(
                inp,
                mode=args.mode,
                output_dir=output_dir,
                pose_weights=pose_weights,
            )
            print(f"\n[{idx}/{len(videos)}] {inp.name} -> {out.name}")
            try:
                _run_one_video(args, inp, out)
            except Exception as e:
                failures += 1
                print(f"  [ERROR] {inp.name}: {e}", file=sys.stderr)
        if failures:
            sys.exit(f"Completed with {failures} failed video(s). See errors above.")
        return

    if args.input is None:
        sys.exit("Input video is required unless --all is used.")

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        sys.exit(f"Input not found: {inp}")
    out = (
        _default_output_for(
            inp,
            mode=args.mode,
            pose_weights=(
                _effective_pose_weights(args) if args.mode == "top_motion" else None
            ),
        )
        if args.output is None
        else args.output.expanduser().resolve()
    )
    _run_one_video(args, inp, out)


if __name__ == "__main__":
    main()
