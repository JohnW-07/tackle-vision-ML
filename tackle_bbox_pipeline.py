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
#DEFAULT_WEIGHTS = "yolo11n.pt"
DEFAULT_WEIGHTS = "yolo26x.pt"
DEFAULT_BALL_CONF = 0.15

# Easy hot-swap for the pose model used by --mode top_motion.
# Example alternatives:
#TOP_MOTION_POSE_MODEL = "yolo11x-pose.pt"
#TOP_MOTION_POSE_MODEL = "yolo26x-pose.pt"
TOP_MOTION_POSE_MODEL = "yolo26n-pose.pt"
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

# ---------------------------------------------------------------------------
# Probabilistic ball tracker — tuning constants
# ---------------------------------------------------------------------------
BALL_TRACK_MAX_MISS = 10          # frames before track declared lost
BALL_TRACK_MIN_MATCH_SCORE = 0.24 # score threshold to update an active track
BALL_TRACK_REACQUIRE_SCORE = 0.30 # higher threshold to re-init from lost state
BALL_TRACK_HANDOFF_WINDOW = 8     # frames of widened tolerance after handoff detected
BALL_TRACK_VEL_ALPHA = 0.35       # EMA weight for velocity update
BALL_TRACK_SIZE_ALPHA = 0.25      # EMA weight for box size update
BALL_TRACK_TEMPORAL_WEIGHT = 0.42 # weight of prediction-consistency term
BALL_TRACK_CHEST_WEIGHT = 0.28    # weight of soft player-chest prior term
BALL_TRACK_APPEARANCE_WEIGHT = 0.24  # weight of appearance (color/shape) term
BALL_TRACK_OWNER_WEIGHT = 0.12    # bonus for matching previously known owner


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


def _estimate_ball_box_from_carrier(person_xyxy: np.ndarray) -> np.ndarray:
    """
    Fallback football box anchored to the carrier's torso/hips.
    This keeps the label visible when the detector drops the tiny football.
    """
    x1, y1, x2, y2 = [float(v) for v in person_xyxy]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    bw = max(10.0, min(0.30 * w, 0.18 * h))
    bh = max(8.0, bw / 1.55)
    cx = x1 + 0.58 * w
    cy = y1 + 0.56 * h
    return np.array(
        [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0],
        dtype=np.float64,
    )


def _detect_ball_from_carrier_roi(
    frame_bgr: np.ndarray,
    person_xyxy: np.ndarray,
) -> np.ndarray | None:
    """
    Detect a brown football-like blob inside the carrier box when YOLO misses it.
    Works as a fallback for low-res youth-football footage.
    """
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in person_xyxy]
    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(x1 + 1, min(w_img, x2))
    y2 = max(y1 + 1, min(h_img, y2))
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    warm_mask = cv2.inRange(hsv, np.array([5, 35, 20], dtype=np.uint8), np.array([24, 230, 220], dtype=np.uint8))
    kernel = np.ones((3, 3), dtype=np.uint8)
    warm_mask = cv2.morphologyEx(warm_mask, cv2.MORPH_OPEN, kernel)
    warm_mask = cv2.morphologyEx(warm_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(warm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    roi_h, roi_w = roi.shape[:2]
    expected = np.array([0.58 * roi_w, 0.56 * roi_h], dtype=np.float64)
    best_box: np.ndarray | None = None
    best_score = -1.0

    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = float(bw * bh)
        if area < 0.0025 * roi_w * roi_h or area > 0.11 * roi_w * roi_h:
            continue
        aspect = bw / float(max(1, bh))
        if not (0.55 <= aspect <= 2.4):
            continue
        center = np.array([bx + bw / 2.0, by + bh / 2.0], dtype=np.float64)
        dist = float(np.linalg.norm(center - expected)) / (np.hypot(roi_w, roi_h) + 1e-6)
        fill = float(cv2.contourArea(cnt)) / area
        score = 0.45 * fill + 0.30 * min(aspect, 1.8) / 1.8 + 0.25 * max(0.0, 1.0 - 2.2 * dist)
        if score > best_score:
            best_score = score
            best_box = np.array([x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh], dtype=np.float64)

    if best_score < 0.36:
        return None
    return best_box


def _player_ball_possession_score(
    frame_bgr: np.ndarray,
    player_xyxy: np.ndarray,
    ball_xyxy: np.ndarray,
    ball_confs: np.ndarray,
    frame_diag: float,
) -> tuple[float, np.ndarray | None, str | None]:
    """
    Score how likely this player is holding the football on this frame.
    Uses either a YOLO ball detection near the hands/torso or a carrier-ROI blob.
    """
    player_box = player_xyxy.astype(np.float64)
    player_arr = player_box.reshape(1, 4)
    best_score = 0.0
    best_box: np.ndarray | None = None
    best_source: str | None = None

    for i, ball in enumerate(ball_xyxy):
        ball_box = ball.astype(np.float64)
        assoc = _ball_hands_association_score(player_box, ball_box)
        if assoc < 0.035:
            continue
        app = _football_appearance_score(
            frame_bgr,
            ball_box,
            player_arr,
            frame_diag,
            float(ball_confs[i]),
        )
        app_score = float(np.clip(app, 0.0, 1.0)) if app >= 0.0 else 0.0
        assoc_score = float(np.clip(assoc / 0.35, 0.0, 1.0))
        det_score = 0.66 * assoc_score + 0.22 * app_score + 0.12 * float(ball_confs[i])
        if det_score > best_score:
            best_score = det_score
            best_box = ball_box
            best_source = "det"

    roi_box = _detect_ball_from_carrier_roi(frame_bgr, player_box)
    if roi_box is not None:
        roi_assoc = _ball_hands_association_score(player_box, roi_box)
        roi_score = 0.52 + 0.48 * float(np.clip(roi_assoc / 0.35, 0.0, 1.0))
        if roi_score > best_score:
            best_score = roi_score
            best_box = roi_box
            best_source = "roi"

    return best_score, best_box, best_source


def _choose_ballcarrier_from_possession(
    frames: list[np.ndarray],
    frame_records: list[dict],
    tid_a: int,
    tid_b: int,
    frame_diag: float,
) -> tuple[int, int, dict[int, dict[str, float]]]:
    """
    Decide carrier/tackler over the whole clip by comparing per-frame possession
    evidence for the two highest-motion tracks.
    """
    stats: dict[int, dict[str, float]] = {
        tid_a: {"score_sum": 0.0, "wins": 0.0, "strong_wins": 0.0, "evidence_frames": 0.0},
        tid_b: {"score_sum": 0.0, "wins": 0.0, "strong_wins": 0.0, "evidence_frames": 0.0},
    }

    for frame_bgr, rec in zip(frames, frame_records, strict=True):
        pids = rec["person_ids"]
        if len(pids) == 0:
            continue
        pxy = rec["person_xyxy"]
        tid_to_i = {int(t): i for i, t in enumerate(pids)}
        frame_scores: dict[int, float] = {tid_a: 0.0, tid_b: 0.0}

        for tid in (tid_a, tid_b):
            if tid not in tid_to_i:
                continue
            score, _, _ = _player_ball_possession_score(
                frame_bgr,
                pxy[tid_to_i[tid]],
                rec["ball_xyxy"],
                rec["ball_confs"],
                frame_diag,
            )
            frame_scores[tid] = score
            stats[tid]["score_sum"] += score
            if score >= 0.45:
                stats[tid]["evidence_frames"] += 1.0

        sa = frame_scores[tid_a]
        sb = frame_scores[tid_b]
        best = max(sa, sb)
        margin = abs(sa - sb)
        if best < 0.45 or margin < 0.08:
            continue
        winner = tid_a if sa > sb else tid_b
        stats[winner]["wins"] += 1.0
        if margin >= 0.18:
            stats[winner]["strong_wins"] += 1.0

    key_a = (
        stats[tid_a]["wins"],
        stats[tid_a]["strong_wins"],
        stats[tid_a]["score_sum"],
        stats[tid_a]["evidence_frames"],
    )
    key_b = (
        stats[tid_b]["wins"],
        stats[tid_b]["strong_wins"],
        stats[tid_b]["score_sum"],
        stats[tid_b]["evidence_frames"],
    )
    if key_a >= key_b:
        return tid_a, tid_b, stats
    return tid_b, tid_a, stats


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
    """Upper ~70% of the person box (torso / hands region proxy)."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    h = max(1.0, y2 - y1)
    y_split = y1 + 0.70 * h
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

    # Soft color descriptors: footballs in low-res footage often look warmer/brighter
    # than expected, so keep this permissive and let player-context do more work.
    brown_mask = (
        (h >= 5.0) & (h <= 22.0) &
        (s >= 35.0) & (s <= 210.0) &
        (v >= 18.0) & (v <= 205.0)
    )
    orange_mask = (
        (h >= 8.0) & (h <= 24.0) &
        (s >= 130.0) & (v >= 135.0)
    )
    brown_ratio = float(np.mean(brown_mask))
    orange_ratio = float(np.mean(orange_mask))

    # Brown footballs are often brighter in compression-heavy clips, so use a wider model.
    hue_score = float(np.exp(-((mean_h - 14.0) ** 2) / (2.0 * (10.0 ** 2))))
    sat_score = float(np.exp(-((mean_s - 118.0) ** 2) / (2.0 * (62.0 ** 2))))
    val_score = float(np.exp(-((mean_v - 108.0) ** 2) / (2.0 * (48.0 ** 2))))
    color_score = 0.34 * hue_score + 0.18 * sat_score + 0.18 * val_score + 0.30 * brown_ratio

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

    score = (
        0.14 * float(det_conf)
        + 0.34 * color_score
        + 0.16 * shape_score
        + 0.08 * size_score
        + 0.10 * prox_score
        + 0.30 * hands_score
    )

    # Bright training cones are usually much more orange than footballs and often lack
    # strong carry-zone association to a player.
    cone_like = 8.0 <= mean_h <= 24.0 and mean_s >= 115.0 and mean_v >= 120.0
    if cone_like:
        cone_penalty = 0.12 + 0.28 * orange_ratio
        score -= cone_penalty * max(0.20, 1.0 - hands_score)

    # Hard reject only when color looks strongly cone-like and there is weak player context.
    if orange_ratio > 0.66 and brown_ratio < 0.12 and hands_score < 0.18 and prox_score < 0.28:
        return -1.0
    return score


def _pick_best_football_box(
    frame_bgr: np.ndarray,
    ball_xyxy: np.ndarray,
    ball_confs: np.ndarray,
    person_xyxy: np.ndarray,
    frame_diag: float,
    min_score: float = 0.26,
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


# ---------------------------------------------------------------------------
# Probabilistic ball tracker
# ---------------------------------------------------------------------------

class BallTrackState:
    """Lightweight constant-velocity Kalman-style football tracker."""

    __slots__ = (
        "box", "center", "velocity", "size",
        "miss_count", "age", "visible_count", "confidence",
        "state", "owner_track_id", "last_update_frame", "_handoff_frames_left",
    )

    def __init__(self) -> None:
        self.box: np.ndarray | None = None
        self.center: np.ndarray | None = None
        self.velocity = np.zeros(2, dtype=np.float64)
        self.size = np.zeros(2, dtype=np.float64)   # (w, h)
        self.miss_count: int = 0
        self.age: int = 0
        self.visible_count: int = 0
        self.confidence: float = 0.0
        self.state: str = "lost"   # "visible" | "predicted" | "handoff" | "lost"
        self.owner_track_id: int | None = None
        self.last_update_frame: int = -1
        self._handoff_frames_left: int = 0

    def predict(self) -> np.ndarray | None:
        if self.center is None:
            return None
        pc = self.center + self.velocity
        w, h = float(self.size[0]), float(self.size[1])
        return np.array(
            [pc[0] - w / 2, pc[1] - h / 2, pc[0] + w / 2, pc[1] + h / 2],
            dtype=np.float64,
        )

    def update(self, measured_box: np.ndarray, score: float, owner_track_id: int | None) -> None:
        mb = measured_box.astype(np.float64)
        mc = np.array([(mb[0] + mb[2]) / 2.0, (mb[1] + mb[3]) / 2.0], dtype=np.float64)
        ms = np.array([mb[2] - mb[0], mb[3] - mb[1]], dtype=np.float64)
        if self.center is not None:
            self.velocity = (
                BALL_TRACK_VEL_ALPHA * (mc - self.center)
                + (1.0 - BALL_TRACK_VEL_ALPHA) * self.velocity
            )
            self.size = BALL_TRACK_SIZE_ALPHA * ms + (1.0 - BALL_TRACK_SIZE_ALPHA) * self.size
        else:
            self.velocity = np.zeros(2, dtype=np.float64)
            self.size = ms.copy()
        self.center = mc
        self.box = mb.copy()
        self.confidence = score
        self.owner_track_id = owner_track_id
        self.miss_count = 0
        self.age += 1
        self.visible_count += 1

    def mark_missed(self) -> None:
        self.miss_count += 1
        self.age += 1
        if self.center is not None:
            self.center = self.center + self.velocity
            w, h = float(self.size[0]), float(self.size[1])
            self.box = np.array(
                [
                    self.center[0] - w / 2,
                    self.center[1] - h / 2,
                    self.center[0] + w / 2,
                    self.center[1] + h / 2,
                ],
                dtype=np.float64,
            )

    def reset(self) -> None:
        self.box = None
        self.center = None
        self.velocity[:] = 0.0
        self.size[:] = 0.0
        self.miss_count = 0
        self.age = 0
        self.visible_count = 0
        self.confidence = 0.0
        self.state = "lost"
        self.owner_track_id = None
        self.last_update_frame = -1
        self._handoff_frames_left = 0


def _chest_roi_xyxy(person_xyxy: np.ndarray) -> np.ndarray:
    """Carry-zone region: middle torso through hips where youth players usually hold the ball."""
    x1, y1, x2, y2 = [float(v) for v in person_xyxy]
    w, h = x2 - x1, y2 - y1
    return np.array(
        [x1 + 0.21 * w, y1 + 0.16 * h, x1 + 0.79 * w, y1 + 0.70 * h],
        dtype=np.float64,
    )


def _ball_chest_prior_score(
    person_xyxy: np.ndarray, ball_xyxy: np.ndarray, frame_diag: float
) -> float:
    """Soft score: higher when ball overlaps or is near the player's chest ROI.
    Not a hard gate — outside the chest zone still returns > 0."""
    chest = _chest_roi_xyxy(person_xyxy)
    bc = _box_center(ball_xyxy)
    cc = _box_center(chest)
    overlap = _iou(chest, ball_xyxy.astype(np.float64))
    dist_norm = float(np.linalg.norm(bc - cc)) / (frame_diag + 1e-6)
    inside = chest[0] <= bc[0] <= chest[2] and chest[1] <= bc[1] <= chest[3]
    return float(
        overlap * 0.40
        + (0.25 if inside else 0.0)
        + float(np.exp(-dist_norm * 12.0)) * 0.35
    )


def _score_ball_candidate(
    frame_bgr: np.ndarray,
    candidate_xyxy: np.ndarray,
    candidate_conf: float,
    predicted_box: np.ndarray | None,
    person_xyxy: np.ndarray,
    person_track_ids: np.ndarray,
    frame_diag: float,
    prev_owner_track_id: int | None,
    tracker_state: str,
) -> tuple[float, int | None]:
    """Score one YOLO ball candidate; returns (total_score, best_owner_track_id)."""
    cand = candidate_xyxy.astype(np.float64)
    c_center = _box_center(cand)

    # Temporal consistency with the Kalman prediction
    temporal_score = 0.0
    if predicted_box is not None:
        pred = predicted_box.astype(np.float64)
        pred_c = _box_center(pred)
        dist_norm = float(np.linalg.norm(c_center - pred_c)) / (frame_diag + 1e-6)
        # Wider distance tolerance during handoff
        sigma = 0.14 if tracker_state == "handoff" else 0.07
        dist_score = float(np.exp(-(dist_norm ** 2) / (2.0 * sigma ** 2)))
        iou_score = _iou(pred, cand)
        pw = max(1.0, float(pred[2] - pred[0]))
        ph = max(1.0, float(pred[3] - pred[1]))
        cw = max(1.0, float(cand[2] - cand[0]))
        ch = max(1.0, float(cand[3] - cand[1]))
        size_sim = min(pw / cw, cw / pw) * min(ph / ch, ch / ph)
        temporal_score = 0.55 * dist_score + 0.30 * iou_score + 0.15 * size_sim

    # Appearance (existing color/shape scorer; returns -1 on hard color reject)
    p_arr = person_xyxy.astype(np.float64) if len(person_xyxy) > 0 else np.empty((0, 4), np.float64)
    raw_app = _football_appearance_score(frame_bgr, cand, p_arr, frame_diag, candidate_conf)
    appearance_score = float(np.clip(raw_app, 0.0, 1.0)) if raw_app >= 0.0 else 0.0
    hard_rejected = raw_app < 0.0

    # Soft player context: chest prior + proximity, owner continuity bonus
    best_owner_id: int | None = None
    best_owner_score = -1.0
    best_chest = 0.0
    n_persons = len(person_xyxy)

    for idx in range(n_persons):
        p_box = person_xyxy[idx].astype(np.float64)
        tid = int(person_track_ids[idx]) if idx < len(person_track_ids) else None
        chest = _ball_chest_prior_score(p_box, cand, frame_diag)
        prox = float(
            np.exp(
                -float(np.linalg.norm(c_center - _box_center(p_box))) / (frame_diag + 1e-6) * 8.0
            )
        )
        owner_bonus = BALL_TRACK_OWNER_WEIGHT if (tid is not None and tid == prev_owner_track_id) else 0.0
        p_score = chest + 0.25 * prox + owner_bonus
        if chest > best_chest:
            best_chest = chest
        if p_score > best_owner_score:
            best_owner_score = p_score
            best_owner_id = tid

    # During handoff: average chest across the 2 nearest players for more tolerance
    handoff_bonus = 0.0
    if tracker_state == "handoff" and n_persons >= 2:
        dists = sorted(
            (float(np.linalg.norm(c_center - _box_center(person_xyxy[i].astype(np.float64)))), i)
            for i in range(n_persons)
        )
        top2_chest = sum(
            _ball_chest_prior_score(person_xyxy[i].astype(np.float64), cand, frame_diag)
            for _, i in dists[:2]
        ) / 2.0
        handoff_bonus = max(0.0, top2_chest - best_chest) * 0.5

    player_score = best_chest + handoff_bonus

    total = (
        BALL_TRACK_TEMPORAL_WEIGHT * temporal_score
        + BALL_TRACK_APPEARANCE_WEIGHT * appearance_score
        + BALL_TRACK_CHEST_WEIGHT * player_score
        + 0.06 * candidate_conf
    )
    if hard_rejected:
        total *= 0.40  # soft penalty rather than full discard
    return float(total), best_owner_id


def _detect_handoff_condition(
    ball_track: BallTrackState,
    person_xyxy: np.ndarray,
    person_track_ids: np.ndarray,
    frame_diag: float,
) -> bool:
    """True when the known owner and another player are both near the ball and near each other."""
    if ball_track.owner_track_id is None or ball_track.center is None:
        return False
    if len(person_xyxy) < 2:
        return False

    ball_c = ball_track.center
    close_idxs: list[int] = []
    owner_close = False

    for idx, p_box in enumerate(person_xyxy):
        pc = _box_center(p_box.astype(np.float64))
        if float(np.linalg.norm(ball_c - pc)) / (frame_diag + 1e-6) < 0.20:
            close_idxs.append(idx)
            if int(person_track_ids[idx]) == ball_track.owner_track_id:
                owner_close = True

    if not owner_close or len(close_idxs) < 2:
        return False

    for a in range(len(close_idxs)):
        for b in range(a + 1, len(close_idxs)):
            ca = _box_center(person_xyxy[close_idxs[a]].astype(np.float64))
            cb = _box_center(person_xyxy[close_idxs[b]].astype(np.float64))
            if float(np.linalg.norm(ca - cb)) / (frame_diag + 1e-6) < 0.25:
                return True
    return False


_BALL_STATE_LABEL: dict[str, str] = {
    "visible": "Football",
    "predicted": "Football (pred)",
    "handoff": "Football (handoff)",
    "lost": "Football",
}


def update_ball_track(
    frame_bgr: np.ndarray,
    ball_track: BallTrackState,
    ball_xyxy: np.ndarray,
    ball_confs: np.ndarray,
    person_xyxy: np.ndarray,
    person_track_ids: np.ndarray,
    frame_diag: float,
    frame_i: int,
) -> np.ndarray | None:
    """
    One-frame update of the probabilistic ball tracker.
    Returns the current drawable ball box, or None when the track is lost.
    """
    predicted_box = ball_track.predict()

    # Handoff detection only while track is active
    if ball_track.state != "lost":
        if _detect_handoff_condition(ball_track, person_xyxy, person_track_ids, frame_diag):
            if ball_track._handoff_frames_left <= 0:
                ball_track._handoff_frames_left = BALL_TRACK_HANDOFF_WINDOW
        if ball_track._handoff_frames_left > 0:
            ball_track.state = "handoff"
            ball_track._handoff_frames_left -= 1

    # Score every YOLO ball candidate
    best_score = -1.0
    best_box: np.ndarray | None = None
    best_owner: int | None = None

    for i in range(len(ball_xyxy)):
        score, owner_id = _score_ball_candidate(
            frame_bgr,
            ball_xyxy[i],
            float(ball_confs[i]),
            predicted_box,
            person_xyxy,
            person_track_ids,
            frame_diag,
            ball_track.owner_track_id,
            ball_track.state,
        )
        if score > best_score:
            best_score = score
            best_box = ball_xyxy[i]
            best_owner = owner_id

    threshold = BALL_TRACK_REACQUIRE_SCORE if ball_track.state == "lost" else BALL_TRACK_MIN_MATCH_SCORE

    if best_box is not None and best_score >= threshold:
        ball_track.update(best_box, best_score, best_owner)
        if ball_track._handoff_frames_left <= 0:
            ball_track.state = "visible"
        ball_track.last_update_frame = frame_i
        return ball_track.box

    if ball_track.state == "lost":
        return None
    ball_track.mark_missed()
    if ball_track.miss_count > BALL_TRACK_MAX_MISS:
        ball_track.reset()
        return None
    if ball_track.state != "handoff":
        ball_track.state = "predicted"
    return ball_track.box


def _detect_people_and_ball_candidates(
    model: YOLO,
    frame_bgr: np.ndarray,
    *,
    person_conf: float,
    ball_conf: float,
    device: str | None,
    persist_people: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    people_kwargs: dict = {
        "persist": persist_people,
        "classes": [PERSON_CLASS_ID],
        "conf": person_conf,
        "verbose": False,
    }
    if device:
        people_kwargs["device"] = device
    people_res = model.track(frame_bgr, **people_kwargs)[0]

    person_ids = np.array([], dtype=np.int64)
    person_xyxy = np.empty((0, 4), dtype=np.float64)
    person_confs = np.array([], dtype=np.float64)
    if people_res.boxes is not None and len(people_res.boxes) > 0 and people_res.boxes.id is not None:
        person_xyxy = people_res.boxes.xyxy.cpu().numpy()
        person_confs = people_res.boxes.conf.cpu().numpy()
        person_ids = people_res.boxes.id.cpu().numpy().astype(np.int64)

    ball_kwargs: dict = {
        "classes": [BALL_CLASS_ID],
        "conf": ball_conf,
        "verbose": False,
    }
    if device:
        ball_kwargs["device"] = device
    ball_res = model.predict(frame_bgr, **ball_kwargs)[0]

    ball_xyxy = np.empty((0, 4), dtype=np.float64)
    ball_confs = np.array([], dtype=np.float64)
    if ball_res.boxes is not None and len(ball_res.boxes) > 0:
        ball_xyxy = ball_res.boxes.xyxy.cpu().numpy()
        ball_confs = ball_res.boxes.conf.cpu().numpy()

    return person_ids, person_xyxy, person_confs, ball_xyxy, ball_confs


def _find_climax_pair(
    frame_records: list[dict],
    frame_diag: float,
    smooth_window: int = 5,
) -> tuple[int, int, int] | None:
    """
    Returns (climax_frame_idx, tid_a, tid_b) for the frame with peak player interaction,
    or None if no valid interacting pair is found.
    """
    n = len(frame_records)
    if n < 2:
        return None

    raw_scores: list[float] = []
    raw_pairs: list[tuple[int, int] | None] = []

    for rec in frame_records:
        pids = rec["person_ids"]
        pxy = rec["person_xyxy"]
        if len(pids) < 2:
            raw_scores.append(0.0)
            raw_pairs.append(None)
            continue
        best_s = 0.0
        best_pair: tuple[int, int] | None = None
        for a in range(len(pids)):
            for b in range(a + 1, len(pids)):
                s = _pair_score(pxy[a], pxy[b], frame_diag)
                if s > best_s:
                    best_s = s
                    best_pair = (int(pids[a]), int(pids[b]))
        raw_scores.append(best_s)
        raw_pairs.append(best_pair)

    raw_arr = np.array(raw_scores, dtype=np.float64)
    smoothed = np.convolve(raw_arr, np.ones(smooth_window) / smooth_window, mode="same")

    climax_frame = int(np.argmax(smoothed))
    if smoothed[climax_frame] <= 0.0:
        return None

    half = smooth_window // 2
    lo = max(0, climax_frame - half)
    hi = min(n - 1, climax_frame + half)
    best_raw_score = -1.0
    best_raw_frame = climax_frame
    for fi in range(lo, hi + 1):
        if raw_scores[fi] > best_raw_score and raw_pairs[fi] is not None:
            best_raw_score = raw_scores[fi]
            best_raw_frame = fi

    pair = raw_pairs[best_raw_frame]
    if pair is None:
        return None
    return (best_raw_frame, pair[0], pair[1])


def _build_track_interpolated_boxes(
    track_id: int,
    frame_records: list[dict],
) -> dict[int, np.ndarray]:
    """
    Returns {frame_idx: xyxy_box} for every frame.
    Gaps are filled by linear interpolation; edge frames hold the boundary observation.
    """
    observed: dict[int, np.ndarray] = {}
    for fi, rec in enumerate(frame_records):
        pids = rec["person_ids"]
        pxy = rec["person_xyxy"]
        for i, tid in enumerate(pids):
            if int(tid) == track_id:
                observed[fi] = pxy[i].astype(np.float64)
                break

    if not observed:
        return {}

    result: dict[int, np.ndarray] = {}
    n = len(frame_records)
    sorted_frames = sorted(observed)
    first_f = sorted_frames[0]
    last_f = sorted_frames[-1]

    for fi in range(0, first_f):
        result[fi] = observed[first_f].copy()

    for k in range(len(sorted_frames)):
        f_k = sorted_frames[k]
        result[f_k] = observed[f_k].copy()
        if k + 1 < len(sorted_frames):
            f_next = sorted_frames[k + 1]
            for fi in range(f_k + 1, f_next):
                alpha = (fi - f_k) / (f_next - f_k)
                result[fi] = (1.0 - alpha) * observed[f_k] + alpha * observed[f_next]

    for fi in range(last_f + 1, n):
        result[fi] = observed[last_f].copy()

    return result


def _relink_fragmented_track(
    primary_boxes: dict[int, np.ndarray],
    track_id: int,
    frame_records: list[dict],
    max_gap: int = 15,
    iou_thresh: float = 0.10,
) -> dict[int, np.ndarray]:
    """
    Finds gaps > max_gap frames where no raw observation exists for track_id and tries
    to absorb a nearby new track that appears near the gap boundary.
    """
    if not primary_boxes:
        return primary_boxes

    observed_frames: set[int] = set()
    for fi, rec in enumerate(frame_records):
        for tid in rec["person_ids"]:
            if int(tid) == track_id:
                observed_frames.add(fi)
                break

    if not observed_frames:
        return primary_boxes

    sorted_obs = sorted(observed_frames)
    n = len(frame_records)

    for k in range(len(sorted_obs) - 1):
        f_end = sorted_obs[k]
        f_start = sorted_obs[k + 1]
        if f_start - f_end - 1 <= max_gap:
            continue

        boundary_box = primary_boxes.get(f_end)
        if boundary_box is None:
            continue

        window_end = min(f_end + max_gap, n - 1)
        track_first_appearances: dict[int, int] = {}
        for fi in range(f_end + 1, window_end + 1):
            for tid in frame_records[fi]["person_ids"]:
                cand_tid = int(tid)
                if cand_tid == track_id or cand_tid in track_first_appearances:
                    continue
                prev_present = any(
                    int(t) == cand_tid
                    for fr in range(max(0, fi - max_gap), fi)
                    for t in frame_records[fr]["person_ids"]
                )
                if not prev_present:
                    track_first_appearances[cand_tid] = fi

        best_cand: int | None = None
        best_iou = iou_thresh
        for cand_tid, cand_first_fi in track_first_appearances.items():
            pids = frame_records[cand_first_fi]["person_ids"]
            pxy = frame_records[cand_first_fi]["person_xyxy"]
            for i, tid in enumerate(pids):
                if int(tid) == cand_tid:
                    score = _iou(boundary_box, pxy[i].astype(np.float64))
                    if score > best_iou:
                        best_iou = score
                        best_cand = cand_tid
                    break

        if best_cand is None:
            continue

        cand_first = track_first_appearances[best_cand]
        for fi in range(cand_first, f_start):
            pids = frame_records[fi]["person_ids"]
            pxy = frame_records[fi]["person_xyxy"]
            for i, tid in enumerate(pids):
                if int(tid) == best_cand:
                    primary_boxes[fi] = pxy[i].astype(np.float64)
                    break

    return primary_boxes


def run_climax_anchored_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    weights: str,
    conf: float,
    ball_conf: float,
    device: str | None,
    max_frames: int = 4500,
    pose_weights: str | None = DEFAULT_POSE_WEIGHTS,
    pose_conf: float = 0.18,
    climax_smooth_window: int = 5,
) -> None:
    """
    Climax-anchored tackle detection: finds the moment of peak player interaction,
    locks those two tracks, then renders them with interpolation through the full clip.
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
    motion_sum: dict[int, float] = defaultdict(float)
    last_center: dict[int, np.ndarray] = {}
    presence: dict[int, int] = defaultdict(int)
    frame_records: list[dict] = []

    for frame_bgr in frames:
        rec: dict = {
            "person_ids": np.array([], dtype=np.int64),
            "person_xyxy": np.empty((0, 4), dtype=np.float64),
            "person_confs": np.array([], dtype=np.float64),
            "ball_xyxy": np.empty((0, 4), dtype=np.float64),
            "ball_confs": np.array([], dtype=np.float64),
        }
        person_ids, person_xyxy, person_confs, ball_xyxy, ball_confs = _detect_people_and_ball_candidates(
            model,
            frame_bgr,
            person_conf=conf,
            ball_conf=ball_conf,
            device=device,
            persist_people=True,
        )
        rec["person_ids"] = person_ids
        rec["person_xyxy"] = person_xyxy
        rec["person_confs"] = person_confs
        rec["ball_xyxy"] = ball_xyxy
        rec["ball_confs"] = ball_confs

        for tid, box in zip(person_ids, person_xyxy, strict=True):
            tid_i = int(tid)
            presence[tid_i] += 1
            c = _box_center(box)
            if tid_i in last_center:
                motion_sum[tid_i] += float(np.linalg.norm(c - last_center[tid_i]))
            last_center[tid_i] = c

        frame_records.append(rec)

    climax_result = _find_climax_pair(frame_records, frame_diag, climax_smooth_window)
    climax_frame_idx: int | None = None

    if climax_result is not None:
        climax_frame_idx, tid_a, tid_b = climax_result
        print(f"Climax frame {climax_frame_idx}: tid_a={tid_a} tid_b={tid_b}")
    else:
        print("Climax pair not found — falling back to top-motion tracks")
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

    carrier_tid, tackler_tid, possession_stats = _choose_ballcarrier_from_possession(
        frames,
        frame_records,
        tid_a,
        tid_b,
        frame_diag,
    )

    carrier_boxes = _build_track_interpolated_boxes(carrier_tid, frame_records)
    tackler_boxes = _build_track_interpolated_boxes(tackler_tid, frame_records)
    carrier_boxes = _relink_fragmented_track(carrier_boxes, carrier_tid, frame_records)
    tackler_boxes = _relink_fragmented_track(tackler_boxes, tackler_tid, frame_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_writer(output_path, fps, width, height)

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
        for frame_idx, (frame_bgr, rec) in enumerate(zip(frames, frame_records, strict=True)):
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

            carrier_box = carrier_boxes.get(frame_idx)
            tackler_box = tackler_boxes.get(frame_idx)

            ball_box: np.ndarray | None = None
            ball_label = "Football"
            if carrier_box is not None:
                _, ball_box, ball_source = _player_ball_possession_score(
                    frame_bgr,
                    carrier_box,
                    rec["ball_xyxy"],
                    rec["ball_confs"],
                    frame_diag,
                )
                if ball_box is None:
                    ball_box = _estimate_ball_box_from_carrier(carrier_box)
                    ball_label = "Football (est)"
                elif ball_source == "roi":
                    ball_label = "Football"

            if ball_box is not None:
                draw_player_box(
                    frame_bgr,
                    ball_box,
                    color=(0, 255, 255),
                    label=ball_label,
                )

            for role_box, label, color, sk_color in (
                (carrier_box, "Ball Carrier", (255, 0, 0), (0, 220, 120)),
                (tackler_box, "Tackler", (0, 0, 255), (100, 200, 255)),
            ):
                if role_box is None:
                    continue
                draw_player_box(frame_bgr, role_box, color=color, label=label)

                if (
                    pose_model is not None
                    and len(pose_boxes) > 0
                    and len(pose_kpts) > 0
                ):
                    kp = _best_pose_keypoints_for_box(pose_boxes, pose_kpts, role_box)
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
                            bx1 = int(role_box[0])
                            by1 = int(role_box[1])
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

    carrier_stats = possession_stats[carrier_tid]
    tackler_stats = possession_stats[tackler_tid]
    climax_info = f"climax_frame={climax_frame_idx}" if climax_frame_idx is not None else "climax=fallback"
    print(
        f"{climax_info} | IDs: {tid_a}, {tid_b} | carrier={carrier_tid} tackler={tackler_tid} "
        f"| wins {carrier_tid}:{carrier_stats['wins']:.0f} vs {tackler_tid}:{tackler_stats['wins']:.0f} "
        f"| evidence {carrier_tid}:{carrier_stats['evidence_frames']:.0f} vs {tackler_tid}:{tackler_stats['evidence_frames']:.0f} "
        f"| wrote {len(frames)} frames to {output_path}"
    )


def run_top_motion_carrier_tackler_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    weights: str,
    conf: float,
    ball_conf: float,
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
    motion_sum: dict[int, float] = defaultdict(float)
    last_center: dict[int, np.ndarray] = {}
    presence: dict[int, int] = defaultdict(int)
    frame_records: list[dict] = []

    for frame_bgr in frames:
        rec: dict = {
            "person_ids": np.array([], dtype=np.int64),
            "person_xyxy": np.empty((0, 4), dtype=np.float64),
            "person_confs": np.array([], dtype=np.float64),
            "ball_xyxy": np.empty((0, 4), dtype=np.float64),
            "ball_confs": np.array([], dtype=np.float64),
        }
        person_ids, person_xyxy, person_confs, ball_xyxy, ball_confs = _detect_people_and_ball_candidates(
            model,
            frame_bgr,
            person_conf=conf,
            ball_conf=ball_conf,
            device=device,
            persist_people=True,
        )
        rec["person_ids"] = person_ids
        rec["person_xyxy"] = person_xyxy
        rec["person_confs"] = person_confs
        rec["ball_xyxy"] = ball_xyxy
        rec["ball_confs"] = ball_confs

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

    carrier_tid, tackler_tid, possession_stats = _choose_ballcarrier_from_possession(
        frames,
        frame_records,
        tid_a,
        tid_b,
        frame_diag,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_writer(output_path, fps, width, height)
    last_box: dict[int, np.ndarray | None] = {carrier_tid: None, tackler_tid: None}
    miss_ct: dict[int, int] = {carrier_tid: 0, tackler_tid: 0}

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
        for frame_idx, (frame_bgr, rec) in enumerate(zip(frames, frame_records, strict=True)):
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

            ball_box: np.ndarray | None = None
            ball_label = "Football"
            if carrier_tid in tid_to_i:
                carrier_box = pxy[tid_to_i[carrier_tid]]
                _, ball_box, ball_source = _player_ball_possession_score(
                    frame_bgr,
                    carrier_box,
                    rec["ball_xyxy"],
                    rec["ball_confs"],
                    frame_diag,
                )
                if ball_box is None:
                    ball_box = _estimate_ball_box_from_carrier(carrier_box)
                    ball_label = "Football (est)"
                elif ball_source == "roi":
                    ball_label = "Football"
            elif last_box.get(carrier_tid) is not None:
                carrier_box = last_box[carrier_tid]
                _, ball_box, ball_source = _player_ball_possession_score(
                    frame_bgr,
                    carrier_box,
                    rec["ball_xyxy"],
                    rec["ball_confs"],
                    frame_diag,
                )
                if ball_box is None:
                    ball_box = _estimate_ball_box_from_carrier(carrier_box)
                    ball_label = "Football (est)"
                elif ball_source == "roi":
                    ball_label = "Football"

            if ball_box is not None:
                draw_player_box(
                    frame_bgr,
                    ball_box,
                    color=(0, 255, 255),
                    label=ball_label,
                )

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
                elif last_box.get(role_tid) is not None:
                    # Always show last known position — no miss-count limit
                    miss_ct[role_tid] += 1
                    hb = last_box[role_tid]
                    draw_player_box(frame_bgr, hb, color=color, label=label)
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

    carrier_stats = possession_stats[carrier_tid]
    tackler_stats = possession_stats[tackler_tid]
    print(
        f"Top motion IDs: {tid_a}, {tid_b} | carrier={carrier_tid} tackler={tackler_tid} "
        f"| wins {carrier_tid}:{carrier_stats['wins']:.0f} vs {tackler_tid}:{tackler_stats['wins']:.0f} "
        f"| evidence {carrier_tid}:{carrier_stats['evidence_frames']:.0f} vs {tackler_tid}:{tackler_stats['evidence_frames']:.0f} "
        f"| wrote {len(frames)} frames to {output_path}"
    )


def run_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    weights: str,
    conf: float,
    ball_conf: float,
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
    ball_track = BallTrackState()
    frame_i = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            person_ids, person_xyxy, person_confs, ball_xyxy, ball_confs = _detect_people_and_ball_candidates(
                model,
                frame_bgr,
                person_conf=conf,
                ball_conf=ball_conf,
                device=device,
                persist_people=True,
            )

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

            ball_box = update_ball_track(
                frame_bgr,
                ball_track,
                ball_xyxy,
                ball_confs,
                person_xyxy,
                person_ids,
                frame_diag,
                frame_i,
            )

            tid_to_idx = {int(t): i for i, t in enumerate(person_ids)}

            def _best_carrier_idx() -> int:
                c = pick_carrier_idx_from_ball(person_xyxy, person_confs, ball_track.box, frame_diag)
                if c is not None:
                    return c
                areas = (person_xyxy[:, 2] - person_xyxy[:, 0]) * (person_xyxy[:, 3] - person_xyxy[:, 1])
                return int(np.argmax(person_confs * (areas / (np.max(areas) + 1e-6))))

            # --- Carrier: always assign, re-acquire on prolonged miss ---
            if locked_carrier_tid is None:
                locked_carrier_tid = int(person_ids[_best_carrier_idx()])

            if locked_carrier_tid in tid_to_idx:
                carrier_box = person_xyxy[tid_to_idx[locked_carrier_tid]]
                last_carrier_box = carrier_box.copy()
                carrier_miss = 0
            else:
                carrier_miss += 1
                if carrier_miss > lock_hold_frames:
                    # Re-pick rather than drop to None
                    new_c = _best_carrier_idx()
                    locked_carrier_tid = int(person_ids[new_c])
                    last_carrier_box = person_xyxy[new_c].copy()
                    carrier_miss = 0

            # --- Tackler: always assign, re-acquire on prolonged miss ---
            if locked_carrier_tid in tid_to_idx:
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

            if locked_tackler_tid is None:
                # Bootstrap: pick first person that isn't the carrier
                for i, tid in enumerate(person_ids):
                    if int(tid) != locked_carrier_tid:
                        locked_tackler_tid = int(tid)
                        break
                if locked_tackler_tid is None and len(person_ids) > 0:
                    locked_tackler_tid = int(person_ids[0])

            if locked_tackler_tid in tid_to_idx:
                t_box = person_xyxy[tid_to_idx[locked_tackler_tid]]
                last_tackler_box = t_box.copy()
                tackler_miss = 0
            else:
                tackler_miss += 1
                if tackler_miss > lock_hold_frames:
                    # Re-pick best non-carrier person
                    for i, tid in enumerate(person_ids):
                        if int(tid) != locked_carrier_tid:
                            locked_tackler_tid = int(tid)
                            last_tackler_box = person_xyxy[i].copy()
                            tackler_miss = 0
                            break

            if ball_box is not None:
                draw_player_box(
                    frame_bgr,
                    ball_box,
                    color=(0, 255, 255),
                    label=_BALL_STATE_LABEL.get(ball_track.state, "Football"),
                )
            elif locked_carrier_tid is not None and locked_carrier_tid in tid_to_idx:
                carrier_box = person_xyxy[tid_to_idx[locked_carrier_tid]]
                fallback_ball = _detect_ball_from_carrier_roi(frame_bgr, carrier_box)
                draw_player_box(
                    frame_bgr,
                    fallback_ball if fallback_ball is not None else _estimate_ball_box_from_carrier(carrier_box),
                    color=(0, 255, 255),
                    label="Football" if fallback_ball is not None else "Football (est)",
                )
            elif last_carrier_box is not None and carrier_miss <= max_hold_frames:
                fallback_ball = _detect_ball_from_carrier_roi(frame_bgr, last_carrier_box)
                draw_player_box(
                    frame_bgr,
                    fallback_ball if fallback_ball is not None else _estimate_ball_box_from_carrier(last_carrier_box),
                    color=(0, 255, 255),
                    label="Football" if fallback_ball is not None else "Football (est)",
                )

            # Draw carrier — always show last known box, no miss limit
            if locked_carrier_tid is not None and locked_carrier_tid in tid_to_idx:
                draw_player_box(
                    frame_bgr,
                    person_xyxy[tid_to_idx[locked_carrier_tid]],
                    color=(255, 0, 0),
                    label="Ball Carrier",
                )
            elif last_carrier_box is not None:
                draw_player_box(frame_bgr, last_carrier_box, color=(255, 0, 0), label="Ball Carrier")

            # Draw tackler — always show last known box, no miss limit
            if locked_tackler_tid is not None and locked_tackler_tid in tid_to_idx:
                draw_player_box(
                    frame_bgr,
                    person_xyxy[tid_to_idx[locked_tackler_tid]],
                    color=(0, 0, 255),
                    label="Tackler",
                )
            elif last_tackler_box is not None:
                draw_player_box(frame_bgr, last_tackler_box, color=(0, 0, 255), label="Tackler")

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
    ball_conf: float,
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
    last_ball_box: np.ndarray | None = None
    carrier_miss = 0
    tackler_miss = 0
    ball_miss = 0
    max_hold_frames = 8

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            person_ids, person_xyxy, person_confs, ball_xyxy, ball_confs = _detect_people_and_ball_candidates(
                model,
                frame_bgr,
                person_conf=conf,
                ball_conf=ball_conf,
                device=device,
                persist_people=True,
            )

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
                last_ball_box = best_ball_box.copy()
                ball_miss = 0
                draw_player_box(frame_bgr, best_ball_box, color=(0, 255, 255), label="Football")
            elif last_ball_box is not None and ball_miss <= max_hold_frames:
                ball_miss += 1
                draw_player_box(frame_bgr, last_ball_box, color=(0, 255, 255), label="Football")
            elif locked_carrier_tid is not None and locked_carrier_tid in tid_to_idx:
                carrier_box = person_xyxy[tid_to_idx[locked_carrier_tid]]
                fallback_ball = _detect_ball_from_carrier_roi(frame_bgr, carrier_box)
                draw_player_box(
                    frame_bgr,
                    fallback_ball if fallback_ball is not None else _estimate_ball_box_from_carrier(carrier_box),
                    color=(0, 255, 255),
                    label="Football" if fallback_ball is not None else "Football (est)",
                )
            else:
                ball_miss += 1

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
        choices=("heuristic", "trained", "top_motion", "climax"),
        default=None,
        help="heuristic: ball+carrier+tackler (streaming); trained: single-class tackler; "
        "top_motion: two highest-motion tracks + ball-in-hands carrier (default with --all); "
        "climax: climax-anchored selection (peak interaction frame) with interpolated tracks",
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
        "--ball-conf",
        type=float,
        default=DEFAULT_BALL_CONF,
        help="Football detection confidence threshold; lower than --conf to recover small blurry balls",
    )
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
    elif mode == "climax":
        pose_tag = (
            _filename_tag(Path(str(pose_weights)).stem)
            if pose_weights
            else "no_pose"
        )
        suffix = f"_climax_{pose_tag}.mp4"
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
            ball_conf=args.ball_conf,
            device=args.device,
        )
    elif args.mode == "top_motion":
        pw = _effective_pose_weights(args)
        run_top_motion_carrier_tackler_pipeline(
            inp,
            out,
            weights=args.weights,
            conf=args.conf,
            ball_conf=args.ball_conf,
            device=args.device,
            pose_weights=pw,
            pose_conf=args.pose_conf,
        )
    elif args.mode == "climax":
        pw = _effective_pose_weights(args)
        run_climax_anchored_pipeline(
            inp,
            out,
            weights=args.weights,
            conf=args.conf,
            ball_conf=args.ball_conf,
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
            ball_conf=args.ball_conf,
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
            _effective_pose_weights(args) if args.mode in ("top_motion", "climax") else None
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
                _effective_pose_weights(args) if args.mode in ("top_motion", "climax") else None
            ),
        )
        if args.output is None
        else args.output.expanduser().resolve()
    )
    _run_one_video(args, inp, out)


if __name__ == "__main__":
    main()
