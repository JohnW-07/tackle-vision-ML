"""
Tackle pair identification from per-frame player detections + optional track IDs.

Implements the weighted scoring pipeline described in the project spec:
proximity, overlap, opposing motion, closing distance, overlap growth,
temporal persistence, crowd penalty, size similarity; plus temporal smoothing.

Frame record format (matches tackle_bbox_pipeline):
  {
    "person_ids": np.ndarray shape (N,) int64,
    "person_xyxy": np.ndarray shape (N, 4) float,
    "person_confs": np.ndarray shape (N,) float,
  }
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Reduced heuristic set (4 signals only):
# 1) rapid closing, 2) whole-video motion, 3) opposing motion, 4) dynamic persistence
# with crowd penalty applied as a suppression term.
W_RAPID_CLOSING = 4.2
W_MOTION_PRIOR = 5.0
W_MOTION_OPPOSITION = 2.2
W_DYNAMIC_PERSISTENCE = 2.0
W_CROWD_PENALTY = 2.4

# Normalization / thresholds (fraction of frame diagonal unless noted)
R_DENSITY_NORM = 0.28  # radius around pair midpoint to count crowd
DENSITY_THRESHOLD = 5.0  # cap for penalty normalization
T_PERSIST = 10  # persistence_frames cap for score component
MIN_VEL_STRONG_NORM = 0.0060  # minimum dynamic speed for tackle-like activity
SAME_DIRECTION_SCALE = 0.35  # handoff: reduce contribution when dot(v_i,v_j) > 0

MIN_CLOSING_FOR_DYNAMIC = 0.0025

# Temporal smoothing (Step 7)
SMOOTH_SWITCH_RATIO = 1.18  # new winner must beat locked score by this ratio
SMOOTH_SWITCH_MARGIN = 0.08  # absolute score margin on normalized scale
LOCK_DECAY = 0.92  # when forcing previous pair, decay locked strength slightly
MIN_DYNAMIC_FRAMES_FOR_VALID_PAIR = 6
MIN_CLOSING_SUM_FOR_VALID_PAIR = 0.045  # summed normalized positive closing


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
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


def _pair_key(ta: int, tb: int) -> tuple[int, int]:
    return (ta, tb) if ta < tb else (tb, ta)


@dataclass
class _FrameScratch:
    prev_centers: dict[int, np.ndarray] = field(default_factory=dict)
    prev_dist: dict[tuple[int, int], float] = field(default_factory=dict)
    prev_iou: dict[tuple[int, int], float] = field(default_factory=dict)
    persistence: dict[tuple[int, int], int] = field(default_factory=dict)


def _filter_players(
    person_ids: np.ndarray,
    person_xyxy: np.ndarray,
    person_confs: np.ndarray,
    conf_threshold: float,
) -> list[int]:
    """Indices into arrays for kept detections."""
    if len(person_xyxy) == 0:
        return []
    keep: list[int] = []
    for i in range(len(person_xyxy)):
        if float(person_confs[i]) >= conf_threshold:
            keep.append(i)
    return keep


def _local_density(
    mid: np.ndarray,
    centers: list[np.ndarray],
    exclude_idx: set[int],
    radius: float,
) -> int:
    cnt = 0
    for ei, c in enumerate(centers):
        if ei in exclude_idx:
            continue
        if float(np.linalg.norm(c - mid)) <= radius:
            cnt += 1
    return cnt


def score_pair_for_frame(
    *,
    tid_i: int,
    tid_j: int,
    box_i: np.ndarray,
    box_j: np.ndarray,
    conf_i: float,
    conf_j: float,
    frame_diag: float,
    centers_frame: list[np.ndarray],
    ia: int,
    ib: int,
    scratch: _FrameScratch,
    vel_i: np.ndarray | None,
    vel_j: np.ndarray | None,
    frame_max_closing: float,
) -> tuple[float, dict[str, float]]:
    """Returns (total_score, debug_components)."""
    diag = float(frame_diag) + 1e-6
    c_i = _box_center_xyxy(box_i)
    c_j = _box_center_xyxy(box_j)
    d = float(np.linalg.norm(c_i - c_j))
    iou = _iou_xyxy(box_i, box_j)

    mid = (c_i + c_j) * 0.5
    R = R_DENSITY_NORM * diag
    exclude = {ia, ib}
    density = _local_density(mid, centers_frame, exclude, R)
    crowd_penalty = min(1.0, density / max(DENSITY_THRESHOLD, 1e-6))

    pk = _pair_key(tid_i, tid_j)
    prev_d = scratch.prev_dist.get(pk)
    delta_d_norm = 0.0
    if prev_d is not None:
        delta_d_norm = (prev_d - d) / diag
    closing_raw = max(0.0, delta_d_norm)
    rapid_closing_score = closing_raw / (frame_max_closing + 1e-6)
    rapid_closing_score = float(np.clip(rapid_closing_score, 0.0, 1.0))

    scratch.prev_dist[pk] = d
    scratch.prev_iou[pk] = iou

    # Velocity-based motion opposition (Step 3.6, 8.1)
    motion_opposition_score = 0.0
    speed_i = speed_j = 0.0
    if vel_i is not None and vel_j is not None:
        speed_i = float(np.linalg.norm(vel_i)) / diag
        speed_j = float(np.linalg.norm(vel_j)) / diag
        denom = speed_i * speed_j + 1e-9
        cos_align = float(np.dot(vel_i, vel_j)) / denom
        motion_opposition_score = max(0.0, (-cos_align) * 0.5 + 0.5)
        if np.dot(vel_i, vel_j) > 0:
            motion_opposition_score *= SAME_DIRECTION_SCALE

    max_speed = max(speed_i, speed_j)

    # Dynamic persistence favors pairs that repeatedly close quickly.
    dynamic_contact = max_speed >= MIN_VEL_STRONG_NORM or closing_raw >= MIN_CLOSING_FOR_DYNAMIC
    if dynamic_contact:
        scratch.persistence[pk] = scratch.persistence.get(pk, 0) + 1
    else:
        scratch.persistence[pk] = 0
    persistence_frames = scratch.persistence[pk]
    dynamic_persistence_score = min(1.0, persistence_frames / float(T_PERSIST))

    crowd_term = W_CROWD_PENALTY * crowd_penalty

    total = (
        W_RAPID_CLOSING * rapid_closing_score
        + W_MOTION_OPPOSITION * motion_opposition_score
        + W_DYNAMIC_PERSISTENCE * dynamic_persistence_score
        - crowd_term
    )

    # Hard reject stationary/slowly changing pairs even if close.
    if not dynamic_contact and iou < 0.08:
        total = -1.0

    debug = {
        "closing": rapid_closing_score,
        "motion_opp": motion_opposition_score,
        "persist": dynamic_persistence_score,
        "crowd": crowd_penalty,
        "iou": iou,
        "conf_avg": 0.5 * (conf_i + conf_j),
    }
    total *= 0.90 + 0.10 * debug["conf_avg"]

    return float(total), debug


def _box_center_xyxy(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)


def best_pair_one_frame(
    rec: dict[str, Any],
    *,
    frame_diag: float,
    conf_threshold: float,
    scratch: _FrameScratch,
) -> tuple[tuple[int, int] | None, float, tuple[np.ndarray, np.ndarray] | None]:
    """Best tackle pair on a single frame."""
    pids = rec["person_ids"]
    pxy = rec["person_xyxy"]
    pconf = rec["person_confs"]
    if len(pids) < 2:
        scratch.prev_centers.clear()
        return None, 0.0, None

    keep = _filter_players(pids, pxy, pconf, conf_threshold)
    if len(keep) < 2:
        scratch.prev_centers.clear()
        for ki in keep:
            scratch.prev_centers[int(pids[ki])] = _box_center_xyxy(
                pxy[ki].astype(np.float64)
            ).copy()
        return None, 0.0, None

    centers = [_box_center_xyxy(pxy[i].astype(np.float64)) for i in keep]
    best_key: tuple[int, int] | None = None
    best_score = -1e18
    best_boxes: tuple[np.ndarray, np.ndarray] | None = None

    # Compute closing rates for all visible pairs first, then normalize "rapid" closing
    # relative to other pairs in this frame.
    frame_closing_vals: list[float] = []
    for ai in range(len(keep)):
        for bi in range(ai + 1, len(keep)):
            ia, ib = keep[ai], keep[bi]
            tid_a, tid_b = int(pids[ia]), int(pids[ib])
            box_a = pxy[ia].astype(np.float64)
            box_b = pxy[ib].astype(np.float64)
            c_a = _box_center_xyxy(box_a)
            c_b = _box_center_xyxy(box_b)
            d_now = float(np.linalg.norm(c_a - c_b))
            pk = _pair_key(tid_a, tid_b)
            d_prev = scratch.prev_dist.get(pk)
            if d_prev is None:
                frame_closing_vals.append(0.0)
            else:
                frame_closing_vals.append(max(0.0, (d_prev - d_now) / (frame_diag + 1e-6)))
    frame_max_closing = max(frame_closing_vals) if frame_closing_vals else 0.0

    for ai in range(len(keep)):
        for bi in range(ai + 1, len(keep)):
            ia, ib = keep[ai], keep[bi]
            tid_a, tid_b = int(pids[ia]), int(pids[ib])
            box_a = pxy[ia].astype(np.float64)
            box_b = pxy[ib].astype(np.float64)

            vel_a = vel_b = None
            if tid_a in scratch.prev_centers:
                vel_a = centers[ai] - scratch.prev_centers[tid_a]
            if tid_b in scratch.prev_centers:
                vel_b = centers[bi] - scratch.prev_centers[tid_b]

            score, _ = score_pair_for_frame(
                tid_i=tid_a,
                tid_j=tid_b,
                box_i=box_a,
                box_j=box_b,
                conf_i=float(pconf[ia]),
                conf_j=float(pconf[ib]),
                frame_diag=frame_diag,
                centers_frame=centers,
                ia=ai,
                ib=bi,
                scratch=scratch,
                vel_i=vel_a,
                vel_j=vel_b,
                frame_max_closing=frame_max_closing,
            )

            pk = _pair_key(tid_a, tid_b)
            if score > best_score:
                best_score = score
                best_key = pk
                best_boxes = (box_a.copy(), box_b.copy())

    # Update prev_centers for next frame (after all pairs — use current centers)
    scratch.prev_centers.clear()
    for idx, ki in enumerate(keep):
        scratch.prev_centers[int(pids[ki])] = centers[idx].copy()

    if best_key is None:
        return None, 0.0, None
    return best_key, best_score, best_boxes


def _fallback_pair_from_motion_presence(
    motion_sum: dict[int, float],
    presence: dict[int, int],
) -> tuple[int, int]:
    ordered_motion = sorted(motion_sum.items(), key=lambda kv: -kv[1])
    top_ids: list[int] = [int(t) for t, _ in ordered_motion[:2]]
    if len(top_ids) < 2:
        extra = sorted(
            [(t, c) for t, c in presence.items() if int(t) not in top_ids],
            key=lambda kv: -kv[1],
        )
        for t, _ in extra:
            top_ids.append(int(t))
            if len(top_ids) >= 2:
                break
    if len(top_ids) < 2:
        raise ValueError("could not find two person tracks")
    ta, tb = top_ids[0], top_ids[1]
    if ta == tb:
        for t, _ in sorted(presence.items(), key=lambda kv: -kv[1]):
            if int(t) != ta:
                tb = int(t)
                break
    return ta, tb


def _pair_dynamic_evidence(
    frame_records: list[dict[str, Any]],
    pair: tuple[int, int],
    frame_diag: float,
) -> tuple[int, float]:
    """
    Returns (dynamic_frames, closing_sum_norm) for a track pair across the clip.
    Used to hard-reject stationary close-by pairs.
    """
    a, b = int(pair[0]), int(pair[1])
    prev_dist: float | None = None
    dynamic_frames = 0
    closing_sum = 0.0
    diag = float(frame_diag) + 1e-6
    prev_centers: dict[int, np.ndarray] = {}

    for rec in frame_records:
        pids = rec["person_ids"]
        pxy = rec["person_xyxy"]
        if len(pids) == 0:
            continue
        tid_to_i = {int(t): i for i, t in enumerate(pids)}
        if a not in tid_to_i or b not in tid_to_i:
            continue
        ca = _box_center_xyxy(pxy[tid_to_i[a]].astype(np.float64))
        cb = _box_center_xyxy(pxy[tid_to_i[b]].astype(np.float64))
        d = float(np.linalg.norm(ca - cb))

        closing = 0.0
        if prev_dist is not None:
            closing = max(0.0, (prev_dist - d) / diag)
        closing_sum += closing
        prev_dist = d

        va = vb = 0.0
        if a in prev_centers:
            va = float(np.linalg.norm(ca - prev_centers[a])) / diag
        if b in prev_centers:
            vb = float(np.linalg.norm(cb - prev_centers[b])) / diag

        if max(va, vb) >= MIN_VEL_STRONG_NORM or closing >= 0.0025:
            dynamic_frames += 1

        prev_centers[a] = ca
        prev_centers[b] = cb

    return dynamic_frames, float(closing_sum)


def _pair_motion_prior(
    pair: tuple[int, int],
    motion_sum: dict[int, float] | None,
    max_motion: float,
) -> float:
    if not motion_sum:
        return 0.0
    a, b = int(pair[0]), int(pair[1])
    mean_motion = 0.5 * (motion_sum.get(a, 0.0) + motion_sum.get(b, 0.0))
    return float(np.clip(mean_motion / (max_motion + 1e-6), 0.0, 1.0))


def select_tackle_pair_from_sequence(
    frame_records: list[dict[str, Any]],
    *,
    frame_diag: float,
    person_conf_threshold: float = 0.5,
    motion_sum: dict[int, float] | None = None,
    presence: dict[int, int] | None = None,
) -> tuple[int, int, int | None, float]:
    """
    Run per-frame tackle scoring + temporal hysteresis, aggregate to one pair for the clip.

    Returns:
        (track_id_a, track_id_b, best_frame_index, confidence in [0,1])
    """
    if not frame_records:
        raise ValueError("empty frame_records")

    scratch = _FrameScratch()
    locked_pair: tuple[int, int] | None = None
    locked_strength = 0.0
    pair_accum: dict[tuple[int, int], float] = defaultdict(float)
    pair_best_frame: dict[tuple[int, int], tuple[float, int]] = {}

    for fi, rec in enumerate(frame_records):
        raw_key, raw_score, _ = best_pair_one_frame(
            rec,
            frame_diag=frame_diag,
            conf_threshold=person_conf_threshold,
            scratch=scratch,
        )
        if raw_key is None:
            continue

        chosen = raw_key
        if locked_pair is None:
            locked_pair = raw_key
            locked_strength = raw_score
        else:
            if raw_key == locked_pair:
                locked_strength = 0.65 * locked_strength + 0.35 * raw_score
            else:
                if raw_score >= locked_strength * SMOOTH_SWITCH_RATIO + SMOOTH_SWITCH_MARGIN:
                    locked_pair = raw_key
                    locked_strength = raw_score
                else:
                    locked_strength *= LOCK_DECAY

        pair_accum[chosen] += max(0.0, raw_score)
        prev = pair_best_frame.get(chosen)
        if prev is None or raw_score > prev[0]:
            pair_best_frame[chosen] = (raw_score, fi)

    if not pair_accum:
        if motion_sum is not None and presence is not None:
            ta, tb = _fallback_pair_from_motion_presence(motion_sum, presence)
            return ta, tb, None, 0.0
        raise ValueError("no tackle pair scores and no fallback motion/presence")

    max_motion = max(motion_sum.values()) if motion_sum else 1.0
    rescored: list[tuple[float, tuple[int, int], float]] = []
    for pair, base in pair_accum.items():
        dyn_frames, closing_sum = _pair_dynamic_evidence(frame_records, pair, frame_diag)
        # Hard rejection for stationary-pair false positives.
        if (
            dyn_frames < MIN_DYNAMIC_FRAMES_FOR_VALID_PAIR
            or closing_sum < MIN_CLOSING_SUM_FOR_VALID_PAIR
        ):
            continue
        motion_prior = _pair_motion_prior(pair, motion_sum, max_motion)
        final_score = base + W_MOTION_PRIOR * motion_prior
        rescored.append((final_score, pair, base))
    if not rescored:
        for pair, base in pair_accum.items():
            motion_prior = _pair_motion_prior(pair, motion_sum, max_motion)
            rescored.append((base + W_MOTION_PRIOR * motion_prior, pair, base))
    rescored.sort(key=lambda x: -x[0])
    ranked = [(pair, final) for final, pair, _ in rescored]
    best_pair = ranked[0][0]
    best_total = ranked[0][1]
    second_total = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = float(best_total / (best_total + second_total + 1e-6))

    _, peak_f = pair_best_frame.get(best_pair, (0.0, -1))
    peak_frame = int(peak_f) if peak_f >= 0 else None
    return best_pair[0], best_pair[1], peak_frame, confidence


def tackle_pair_for_frame_public(
    frame_records: list[dict[str, Any]],
    frame_index: int,
    *,
    frame_diag: float,
    person_conf_threshold: float = 0.5,
    scratch: _FrameScratch | None = None,
) -> tuple[tuple[int, int] | None, float, tuple[np.ndarray, np.ndarray] | None]:
    """
    API for callers who want the spec output on one frame (requires sequential scratch state).

    If scratch is None, creates fresh state (accurate only for first frame unless reused).
    """
    if frame_index < 0 or frame_index >= len(frame_records):
        raise IndexError("frame_index out of range")
    st = scratch if scratch is not None else _FrameScratch()
    return best_pair_one_frame(
        frame_records[frame_index],
        frame_diag=frame_diag,
        conf_threshold=person_conf_threshold,
        scratch=st,
    )
