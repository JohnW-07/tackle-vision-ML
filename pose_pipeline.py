"""
pose_pipeline.py – Overlay YOLO11 pose skeletons on annotated football frames
and generate per-clip injury-risk diagnostic JSON files.

Reads frame images from raws/images/ (Roboflow naming convention), groups them
by video clip, looks up matching annotations in raws/labels/, runs pose
estimation on the target class, writes one annotated video per clip to
finals/pose/, and writes a companion JSON with kinematics and injury-risk
diagnostics.

Usage:
  python pose_pipeline.py                   # process all clips in raws/
  python pose_pipeline.py --class-id 1     # pose a different class
  python pose_pipeline.py --source-fps 60  # correct the source frame rate

Requirements:
  pip install ultralytics   (yolo11x-pose.pt auto-downloads on first run)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("Missing dependency: pip install ultralytics") from e


# ── Edit these constants to change pipeline behaviour ─────────────────────────
# Roboflow label convention: 0=BallCarrier, 1=FirstTouch, 3=Tackler
# Each entry: label, skeleton_color (BGR), keypoint_color (BGR), bbox_color (BGR)
ROLE_CONFIG: dict[int, dict] = {
    0: {
        "label":          "BallCarrier",
        "skeleton_color": (255, 255, 0),   # cyan
        "keypoint_color": (200, 220, 0),
        "bbox_color":     (200, 200, 0),
    },
    3: {
        "label":          "Tackler",
        "skeleton_color": (0, 0, 255),     # red
        "keypoint_color": (50, 50, 255),
        "bbox_color":     (0, 0, 200),
    },
}

POSE_WEIGHTS       = "yolo11x-pose.pt"  # Best pose model < 5 GB (auto-downloads)
POSE_CONF          = 0.25              # Pose detection confidence threshold
ANNOTATION_MIN_IOU = 0.2              # Min IoU between annotation box and pose detection
OUTPUT_FPS         = 10.0             # FPS for stitched output video (frames are sparse)

# Kinematics — edit SOURCE_FPS to match the original video frame rate.
# Frames in raws/ are assumed to be consecutive frames (stride = 1).
# All Gs values scale with SOURCE_FPS², so this matters.
SOURCE_FPS         = 30.0             # Original video frame rate (frames/second)

# Body-proportion constants for pixel→metre scale estimation.
# The pipeline tries shoulder width first, then hip width as a fallback.
SHOULDER_WIDTH_M   = 0.45             # Average adult shoulder width (metres)
HIP_WIDTH_M        = 0.35             # Average adult hip width (metres)

# Injury-risk thresholds (edit to adjust flag sensitivity)
HEAD_ACCEL_RISK_G   = 4.0    # Head acceleration flag (g)
JOINT_ACCEL_RISK_G  = 10.0   # Any-joint acceleration flag (g)
JERK_RISK_G_PER_S   = 30.0   # Jerk magnitude flag (g/s)
NECK_FLEX_RISK_DEG  = 45.0   # Neck flexion from vertical flag (°)
SPINE_LEAN_RISK_DEG = 45.0   # Trunk lean from vertical flag (°)
KNEE_FLEX_RISK_DEG  = 150.0  # Extreme knee flexion flag (°)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).resolve().parent
RAWS_FRAMES_DIR = PROJECT_ROOT / "raws" / "images"
RAWS_LABELS_DIR = PROJECT_ROOT / "raws" / "labels"
FINALS_DIR      = PROJECT_ROOT / "finals" / "pose"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# COCO 17-keypoint names (index = keypoint ID)
JOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Joints included in kinematics and diagnostics.
# Eyes, ears, and wrists are excluded — too small to track reliably at
# typical broadcast resolution.
ACTIVE_JOINT_IDS: set[int] = {
    0,        # nose (head centre)
    5, 6,     # shoulders
    7, 8,     # elbows
    11, 12,   # hips
    13, 14,   # knees
    15, 16,   # ankles
}

# Joint angle definitions: (vertex_idx, arm1_idx, arm2_idx)
# Wrist-dependent elbow angles are omitted (wrists excluded above).
JOINT_ANGLE_DEFS: dict[str, tuple[int, int, int]] = {
    "left_knee":      (13, 11, 15),  # hip – knee – ankle
    "right_knee":     (14, 12, 16),
    "left_hip":       (11,  5, 13),  # shoulder – hip – knee
    "right_hip":      (12,  6, 14),
    "left_shoulder":  (5,   7, 11),  # elbow – shoulder – hip
    "right_shoulder": (6,   8, 12),
}

# COCO skeleton pairs for drawing
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
CODEC_CANDIDATES = ["avc1", "mp4v"]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _angle_deg(
    a: np.ndarray, vertex: np.ndarray, b: np.ndarray
) -> float | None:
    """Angle in degrees at `vertex` formed by rays vertex→a and vertex→b."""
    va = a - vertex
    vb = b - vertex
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-6 or nb < 1e-6:
        return None
    cos_a = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _angle_from_vertical_deg(tip: np.ndarray, base: np.ndarray) -> float | None:
    """
    Angle in degrees between the vector (base → tip) and the upward vertical.
    In image coords y increases downward, so upward = (0, -1).
    """
    vec = tip - base
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return None
    upward = np.array([0.0, -1.0])
    cos_a = float(np.clip(np.dot(vec / n, upward), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


# ── Scale estimation ──────────────────────────────────────────────────────────

def _estimate_pixels_per_metre(
    kpts_sequence: list[np.ndarray],   # each (17, 3)
    conf_thresh: float = 0.4,
) -> tuple[float | None, str]:
    """
    Estimate image scale using visible shoulder or hip width.
    Returns (pixels_per_metre, method_used).
    """
    shoulder_widths: list[float] = []
    hip_widths: list[float] = []

    for kpts in kpts_sequence:
        ls, rs = kpts[5], kpts[6]   # left/right shoulder
        lh, rh = kpts[11], kpts[12] # left/right hip

        if ls[2] >= conf_thresh and rs[2] >= conf_thresh:
            shoulder_widths.append(float(np.linalg.norm(ls[:2] - rs[:2])))

        if lh[2] >= conf_thresh and rh[2] >= conf_thresh:
            hip_widths.append(float(np.linalg.norm(lh[:2] - rh[:2])))

    if shoulder_widths:
        ppm = float(np.median(shoulder_widths)) / SHOULDER_WIDTH_M
        return ppm, "shoulder_width"
    if hip_widths:
        ppm = float(np.median(hip_widths)) / HIP_WIDTH_M
        return ppm, "hip_width"
    return None, "unavailable"


# ── Kinematics ────────────────────────────────────────────────────────────────

def _finite_diff(
    values: np.ndarray,   # shape (N,) or (N, 2)
    dt: float,
    order: int,
) -> np.ndarray:
    """
    Compute central finite differences of given order along axis 0.
    Endpoints use forward/backward differences.
    """
    n = len(values)
    result = np.zeros_like(values, dtype=np.float64)
    if n < 2:
        return result
    for i in range(n):
        if i == 0:
            result[i] = (values[1] - values[0]) / dt
        elif i == n - 1:
            result[i] = (values[-1] - values[-2]) / dt
        else:
            result[i] = (values[i + 1] - values[i - 1]) / (2 * dt)
    if order == 1:
        return result
    return _finite_diff(result, dt, order - 1)


def _compute_kinematics(
    frame_indices: list[int],
    kpts_sequence: list[np.ndarray],   # each (17, 3)
    pixels_per_metre: float | None,
    source_fps: float,
    conf_thresh: float = 0.3,
) -> dict:
    """
    Compute per-joint velocity, acceleration, and jerk.
    Returns a dict keyed by joint name.
    """
    n = len(kpts_sequence)
    if n < 2:
        return {}

    # Build per-joint position arrays, masking low-confidence frames as NaN
    pos = np.full((n, 17, 2), np.nan)
    conf_arr = np.zeros((n, 17))
    for t, kpts in enumerate(kpts_sequence):
        for j in range(17):
            conf_arr[t, j] = float(kpts[j, 2])
            if kpts[j, 2] >= conf_thresh:
                pos[t, j] = kpts[j, :2]

    # Time step per frame pair (variable if frame indices have gaps)
    # For a single representative dt we use median spacing
    if len(frame_indices) > 1:
        gaps = np.diff(frame_indices)
        dt = float(np.median(gaps)) / source_fps
    else:
        dt = 1.0 / source_fps

    scale = (1.0 / pixels_per_metre) if pixels_per_metre else None
    G = 9.81

    joint_stats: dict = {}
    for j, jname in enumerate(JOINT_NAMES):
        if j not in ACTIVE_JOINT_IDS:
            continue
        p = pos[:, j, :]   # (N, 2) — may contain NaN
        valid = ~np.isnan(p[:, 0])
        n_valid = int(valid.sum())
        mean_conf = float(conf_arr[:, j].mean())

        if n_valid < 2:
            joint_stats[jname] = {
                "frames_visible": n_valid,
                "mean_confidence": round(mean_conf, 3),
                "note": "insufficient visible frames for kinematics",
            }
            continue

        # Interpolate over NaN gaps for differentiation only
        t_all = np.arange(n)
        t_valid = t_all[valid]
        p_interp = np.column_stack([
            np.interp(t_all, t_valid, p[valid, 0]),
            np.interp(t_all, t_valid, p[valid, 1]),
        ])

        vel = _finite_diff(p_interp, dt, 1)    # px/s
        acc = _finite_diff(p_interp, dt, 2)    # px/s²
        jrk = _finite_diff(p_interp, dt, 3)    # px/s³

        speed_px  = np.linalg.norm(vel, axis=1)
        accel_mag = np.linalg.norm(acc, axis=1)
        jerk_mag  = np.linalg.norm(jrk, axis=1)

        entry: dict = {
            "frames_visible": n_valid,
            "mean_confidence": round(mean_conf, 3),
        }

        if scale:
            speed_ms   = speed_px  * scale
            accel_ms2  = accel_mag * scale
            jerk_ms3   = jerk_mag  * scale
            accel_g    = accel_ms2 / G
            jerk_g_s   = jerk_ms3  / G

            entry.update({
                "max_speed_m_s":         round(float(np.nanmax(speed_ms)),  3),
                "max_acceleration_g":    round(float(np.nanmax(accel_g)),   3),
                "mean_acceleration_g":   round(float(np.nanmean(accel_g)),  3),
                "max_jerk_g_per_s":      round(float(np.nanmax(jerk_g_s)),  3),
                "peak_accel_frame":      int(t_all[np.argmax(accel_g)]),
            })
        else:
            # No scale: report in pixel units for relative comparison
            entry.update({
                "max_speed_px_per_s":    round(float(np.nanmax(speed_px)),  2),
                "max_acceleration_px_s2":round(float(np.nanmax(accel_mag)), 2),
                "max_jerk_px_s3":        round(float(np.nanmax(jerk_mag)),  2),
                "note": "no scale estimate — pixel units only",
            })

        joint_stats[jname] = entry

    return joint_stats


# ── Joint angles ──────────────────────────────────────────────────────────────

def _compute_joint_angles(
    kpts_sequence: list[np.ndarray],
    conf_thresh: float = 0.3,
) -> dict:
    """
    Compute per-frame joint angles and body-orientation angles.
    Returns aggregated stats (mean, min, max, range) per angle.
    """
    angle_frames: dict[str, list[float]] = defaultdict(list)
    neck_angles: list[float] = []
    spine_angles: list[float] = []

    for kpts in kpts_sequence:
        # Named joint angles
        for angle_name, (v_idx, a1_idx, a2_idx) in JOINT_ANGLE_DEFS.items():
            kv, ka1, ka2 = kpts[v_idx], kpts[a1_idx], kpts[a2_idx]
            if kv[2] < conf_thresh or ka1[2] < conf_thresh or ka2[2] < conf_thresh:
                continue
            a = _angle_deg(ka1[:2], kv[:2], ka2[:2])
            if a is not None:
                angle_frames[angle_name].append(a)

        # Neck angle: mid-shoulder → nose vs. vertical
        ls, rs = kpts[5], kpts[6]
        if ls[2] >= conf_thresh and rs[2] >= conf_thresh and kpts[0][2] >= conf_thresh:
            mid_shoulder = (ls[:2] + rs[:2]) / 2.0
            a = _angle_from_vertical_deg(kpts[0][:2], mid_shoulder)
            if a is not None:
                neck_angles.append(a)

        # Spine angle: mid-hip → mid-shoulder vs. vertical
        lh, rh = kpts[11], kpts[12]
        if (ls[2] >= conf_thresh and rs[2] >= conf_thresh
                and lh[2] >= conf_thresh and rh[2] >= conf_thresh):
            mid_hip = (lh[:2] + rh[:2]) / 2.0
            mid_sh  = (ls[:2] + rs[:2]) / 2.0
            a = _angle_from_vertical_deg(mid_sh, mid_hip)
            if a is not None:
                spine_angles.append(a)

    def _stats(vals: list[float]) -> dict | None:
        if not vals:
            return None
        return {
            "mean_deg":  round(float(np.mean(vals)),  1),
            "min_deg":   round(float(np.min(vals)),   1),
            "max_deg":   round(float(np.max(vals)),   1),
            "range_deg": round(float(np.ptp(vals)),   1),
            "n_frames":  len(vals),
        }

    result: dict = {}
    for name, vals in angle_frames.items():
        s = _stats(vals)
        if s:
            result[name] = s

    body_orient: dict = {}
    s = _stats(neck_angles)
    if s:
        body_orient["neck_angle_from_vertical"] = s
    s = _stats(spine_angles)
    if s:
        body_orient["spine_angle_from_vertical"] = s

    return {"joint_angles": result, "body_orientation": body_orient}


# ── Risk flags ────────────────────────────────────────────────────────────────

def _build_risk_flags(
    kinematics: dict,
    angle_data: dict,
    has_scale: bool,
) -> list[dict]:
    flags: list[dict] = []

    if has_scale:
        # Head acceleration
        nose_k = kinematics.get("nose", {})
        head_g = nose_k.get("max_acceleration_g")
        if head_g is not None and head_g >= HEAD_ACCEL_RISK_G:
            flags.append({
                "joint": "nose",
                "metric": "head_acceleration_g",
                "value": round(head_g, 3),
                "threshold": HEAD_ACCEL_RISK_G,
                "severity": "high" if head_g >= HEAD_ACCEL_RISK_G * 1.5 else "moderate",
                "note": "Elevated head acceleration — concussion screening indicator",
            })

        # Any joint exceeding general threshold
        for jname, jdata in kinematics.items():
            ag = jdata.get("max_acceleration_g")
            if ag is not None and ag >= JOINT_ACCEL_RISK_G and jname != "nose":
                flags.append({
                    "joint": jname,
                    "metric": "acceleration_g",
                    "value": round(ag, 3),
                    "threshold": JOINT_ACCEL_RISK_G,
                    "severity": "high" if ag >= JOINT_ACCEL_RISK_G * 1.5 else "moderate",
                    "note": "Extreme joint acceleration",
                })

        # Jerk flags
        for jname, jdata in kinematics.items():
            jrk = jdata.get("max_jerk_g_per_s")
            if jrk is not None and jrk >= JERK_RISK_G_PER_S:
                flags.append({
                    "joint": jname,
                    "metric": "jerk_g_per_s",
                    "value": round(jrk, 3),
                    "threshold": JERK_RISK_G_PER_S,
                    "severity": "moderate",
                    "note": "High rate-of-force change — soft tissue loading indicator",
                })

    # Joint angle flags (no scale needed)
    joint_angles = angle_data.get("joint_angles", {})
    for side in ("left_knee", "right_knee"):
        s = joint_angles.get(side, {})
        mx = s.get("max_deg")
        if mx is not None and mx >= KNEE_FLEX_RISK_DEG:
            flags.append({
                "joint": side,
                "metric": "flexion_deg",
                "value": round(mx, 1),
                "threshold": KNEE_FLEX_RISK_DEG,
                "severity": "moderate",
                "note": "Extreme knee flexion — ligament loading indicator",
            })

    body_orient = angle_data.get("body_orientation", {})
    neck = body_orient.get("neck_angle_from_vertical", {})
    neck_max = neck.get("max_deg")
    if neck_max is not None and neck_max >= NECK_FLEX_RISK_DEG:
        flags.append({
            "joint": "neck",
            "metric": "angle_from_vertical_deg",
            "value": round(neck_max, 1),
            "threshold": NECK_FLEX_RISK_DEG,
            "severity": "high" if neck_max >= NECK_FLEX_RISK_DEG * 1.5 else "moderate",
            "note": "High neck flexion — cervical spine loading indicator",
        })

    spine = body_orient.get("spine_angle_from_vertical", {})
    spine_max = spine.get("max_deg")
    if spine_max is not None and spine_max >= SPINE_LEAN_RISK_DEG:
        flags.append({
            "joint": "spine",
            "metric": "trunk_lean_deg",
            "value": round(spine_max, 1),
            "threshold": SPINE_LEAN_RISK_DEG,
            "severity": "moderate",
            "note": "High trunk lean — lumbar loading indicator",
        })

    return flags


# ── Diagnostics JSON ──────────────────────────────────────────────────────────

def build_diagnostics(
    video_stem: str,
    frame_indices: list[int],
    kpts_sequence: list[np.ndarray],
    frames_processed: int,
    frames_with_pose: int,
    class_id: int,
    source_fps: float,
) -> dict:
    pixels_per_metre, scale_method = _estimate_pixels_per_metre(kpts_sequence)
    has_scale = pixels_per_metre is not None

    kinematics = _compute_kinematics(
        frame_indices, kpts_sequence, pixels_per_metre, source_fps
    )
    angle_data = _compute_joint_angles(kpts_sequence)
    risk_flags = _build_risk_flags(kinematics, angle_data, has_scale)

    # Build ranked peak-acceleration table (for quick review)
    peak_accels: dict[str, float] = {}
    for jname, jdata in kinematics.items():
        ag = jdata.get("max_acceleration_g")
        if ag is not None:
            peak_accels[jname] = round(ag, 3)
    ranked = dict(
        sorted(peak_accels.items(), key=lambda kv: kv[1], reverse=True)
    )

    overall_risk = "low"
    if any(f["severity"] == "high" for f in risk_flags):
        overall_risk = "high"
    elif risk_flags:
        overall_risk = "moderate"

    return {
        "clip": video_stem,
        "metadata": {
            "frames_processed": frames_processed,
            "frames_with_pose": frames_with_pose,
            "pose_class_id": class_id,
            "source_fps_assumed": source_fps,
            "pixels_per_metre_estimated": (
                round(pixels_per_metre, 2) if pixels_per_metre else None
            ),
            "scale_method": scale_method,
            "shoulder_width_assumption_m": SHOULDER_WIDTH_M,
            "hip_width_assumption_m": HIP_WIDTH_M,
            "caveats": (
                "Kinematics are derived from 2D projected pixel positions. "
                "Out-of-plane motion is not captured. Scale is estimated from "
                "assumed body proportions. All values are screening indicators "
                "only and should not be used as definitive clinical measurements."
            ),
        },
        "joint_kinematics": kinematics,
        "joint_angles": angle_data.get("joint_angles", {}),
        "body_orientation": angle_data.get("body_orientation", {}),
        "injury_risk_summary": {
            "overall_risk_level": overall_risk,
            "risk_flags": risk_flags,
            "peak_accelerations_g_ranked": ranked if has_scale else {},
        },
    }


# ── Annotation loading ────────────────────────────────────────────────────────

def _parse_frame_stem(stem: str) -> tuple[str, int] | None:
    m = re.match(r"^(.+)_mp4-(\d+)_jpg", stem)
    if m:
        return m.group(1), int(m.group(2))
    return None


def _discover_clips(
    frames_dir: Path,
) -> dict[str, list[tuple[int, Path]]]:
    clips: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in frames_dir.iterdir():
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parsed = _parse_frame_stem(p.stem)
        if parsed is None:
            print(f"[skip] unrecognised filename: {p.name}", file=sys.stderr)
            continue
        vstem, fidx = parsed
        clips[vstem].append((fidx, p))
    return {vs: sorted(frames) for vs, frames in clips.items()}


def _build_label_index(labels_dir: Path, video_stem: str) -> dict[int, Path]:
    index: dict[int, Path] = {}
    if not labels_dir.is_dir():
        return index
    for p in labels_dir.iterdir():
        if p.suffix != ".txt":
            continue
        parsed = _parse_frame_stem(p.stem)
        if parsed is None:
            continue
        vstem, fidx = parsed
        if vstem == video_stem:
            index[fidx] = p
    return index


def _polygon_to_xyxy(
    coords: list[float], width: int, height: int
) -> tuple[int, int, int, int]:
    xs = [coords[i] * width  for i in range(0, len(coords), 2)]
    ys = [coords[i] * height for i in range(1, len(coords), 2)]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _load_target_boxes(
    label_path: Path, class_id: int, width: int, height: int
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts or int(parts[0]) != class_id:
            continue
        coords = [float(v) for v in parts[1:]]
        if len(coords) < 4:
            continue
        boxes.append(_polygon_to_xyxy(coords, width, height))
    return boxes


# ── Pose matching and drawing ─────────────────────────────────────────────────

def _iou(a: tuple[int, int, int, int], b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def _best_pose_match(
    pose_boxes: np.ndarray,
    pose_kpts: np.ndarray,
    target_box: tuple[int, int, int, int],
    min_iou: float,
) -> np.ndarray | None:
    best_kp: np.ndarray | None = None
    best_score = min_iou
    for i, box in enumerate(pose_boxes):
        score = _iou(target_box, box)
        if score > best_score:
            best_score = score
            best_kp = pose_kpts[i]
    return best_kp


def _draw_skeleton(
    frame: np.ndarray,
    kpts: np.ndarray,
    *,
    skeleton_color: tuple[int, int, int] = (0, 255, 0),
    keypoint_color: tuple[int, int, int] = (0, 200, 255),
    conf_thresh: float = 0.3,
) -> None:
    kp_px = [(int(x), int(y), float(c)) for x, y, c in kpts]
    for i1, i2 in COCO_SKELETON:
        x1, y1, c1 = kp_px[i1]
        x2, y2, c2 = kp_px[i2]
        if c1 < conf_thresh or c2 < conf_thresh:
            continue
        cv2.line(frame, (x1, y1), (x2, y2), skeleton_color, 2, lineType=cv2.LINE_AA)
    for x, y, c in kp_px:
        if c < conf_thresh:
            continue
        cv2.circle(frame, (x, y), 4, keypoint_color, -1, lineType=cv2.LINE_AA)


# ── Video writer ──────────────────────────────────────────────────────────────

def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    for codec in CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for {path}")


# ── Per-clip processing ───────────────────────────────────────────────────────

def _draw_role_box(
    frame: np.ndarray,
    tbox: tuple[int, int, int, int],
    label: str,
    bbox_color: tuple[int, int, int],
) -> None:
    """Draw a labeled bounding box for a role (BallCarrier or Tackler)."""
    x1, y1, x2, y2 = tbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
    text_y = max(20, y1 - 8)
    cv2.putText(
        frame, label, (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, bbox_color, 2, lineType=cv2.LINE_AA,
    )


def _process_clip(
    video_stem: str,
    frames: list[tuple[int, Path]],
    label_index: dict[int, Path],
    output_path: Path,
    model: YOLO,
    *,
    output_fps: float,
    source_fps: float,
    conf: float,
    device: str | None,
    role_config: dict[int, dict],
    min_iou: float,
) -> tuple[int, dict[int, tuple[list[int], list[np.ndarray]]]]:
    """
    Process one clip. Returns:
      (total_frames, {class_id: (kpt_frame_indices, kpts_sequence)})

    Pose estimation runs once per frame (when any role has annotation boxes).
    Each role is drawn with its own color and label.
    """
    first_img = cv2.imread(str(frames[0][1]))
    if first_img is None:
        print(f"[skip] cannot read {frames[0][1]}", file=sys.stderr)
        return 0, {cid: ([], []) for cid in role_config}
    height, width = first_img.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_writer(output_path, output_fps, width, height)

    total = 0
    # Per-role keypoint accumulation
    role_kpts: dict[int, tuple[list[int], list[np.ndarray]]] = {
        cid: ([], []) for cid in role_config
    }

    for fidx, img_path in frames:
        frame_bgr = cv2.imread(str(img_path))
        if frame_bgr is None:
            print(f"[skip] cannot read {img_path}", file=sys.stderr)
            continue

        # Collect annotation boxes per role for this frame
        role_boxes: dict[int, list[tuple[int, int, int, int]]] = {}
        if fidx in label_index:
            for cid in role_config:
                boxes = _load_target_boxes(label_index[fidx], cid, width, height)
                if boxes:
                    role_boxes[cid] = boxes

        if role_boxes:
            # Run pose estimation once for the whole frame
            kwargs: dict = {"conf": conf, "verbose": False}
            if device:
                kwargs["device"] = device
            results = model(frame_bgr, **kwargs)[0]

            pose_boxes = (
                results.boxes.xyxy.cpu().numpy()
                if results.boxes is not None and len(results.boxes)
                else np.empty((0, 4))
            )
            pose_kpts = (
                results.keypoints.data.cpu().numpy()
                if results.keypoints is not None and len(results.keypoints)
                else np.empty((0, 17, 3))
            )

            for cid, tboxes in role_boxes.items():
                role = role_config[cid]
                for tbox in tboxes:
                    _draw_role_box(
                        frame_bgr, tbox,
                        role["label"], role["bbox_color"],
                    )
                    kp = _best_pose_match(pose_boxes, pose_kpts, tbox, min_iou)
                    if kp is not None:
                        _draw_skeleton(
                            frame_bgr, kp,
                            skeleton_color=role["skeleton_color"],
                            keypoint_color=role["keypoint_color"],
                        )
                        role_kpts[cid][0].append(fidx)
                        role_kpts[cid][1].append(kp)

        writer.write(frame_bgr)
        total += 1

    writer.release()
    return total, role_kpts


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pose_pipeline(
    frames_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    *,
    weights: str,
    conf: float,
    device: str | None,
    role_config: dict[int, dict],
    min_iou: float,
    output_fps: float,
    source_fps: float,
) -> None:
    clips = _discover_clips(frames_dir)
    if not clips:
        sys.exit(f"No recognisable image frames found in {frames_dir}")

    role_names = ", ".join(r["label"] for r in role_config.values())
    print(f"Found {len(clips)} clip(s): {', '.join(sorted(clips))}")
    print(f"Roles: {role_names}")
    model = YOLO(weights)

    for video_stem, frames in sorted(clips.items()):
        label_index = _build_label_index(labels_dir, video_stem)
        out_video = output_dir / f"{video_stem}.pose.mp4"
        out_json  = output_dir / f"{video_stem}.diagnostics.json"

        print(f"\n  {video_stem}  ({len(frames)} frames, {len(label_index)} annotated)")

        total, role_kpts = _process_clip(
            video_stem, frames, label_index, out_video, model,
            output_fps=output_fps,
            source_fps=source_fps,
            conf=conf,
            device=device,
            role_config=role_config,
            min_iou=min_iou,
        )

        # Count frames where at least one role had a pose match
        frames_with_pose = len(set(
            fidx
            for fidxs, _ in role_kpts.values()
            for fidx in fidxs
        ))
        print(f"  → {out_video.name}  ({frames_with_pose}/{total} frames with pose overlay)")

        # Build per-role diagnostics, combined into one JSON
        roles_diag: dict = {}
        for cid, (kpt_fidxs, kpts_seq) in role_kpts.items():
            role_label = role_config[cid]["label"]
            if kpts_seq:
                diag = build_diagnostics(
                    video_stem, kpt_fidxs, kpts_seq,
                    frames_processed=total,
                    frames_with_pose=len(kpt_fidxs),
                    class_id=cid,
                    source_fps=source_fps,
                )
                roles_diag[role_label] = {
                    "joint_kinematics": diag["joint_kinematics"],
                    "joint_angles": diag["joint_angles"],
                    "body_orientation": diag["body_orientation"],
                    "injury_risk_summary": diag["injury_risk_summary"],
                }
                risk = diag["injury_risk_summary"]["overall_risk_level"]
                n_flags = len(diag["injury_risk_summary"]["risk_flags"])
                print(f"     {role_label}: {len(kpt_fidxs)} pose frames, risk={risk}, {n_flags} flag(s)")
            else:
                print(f"     {role_label}: no pose matches", file=sys.stderr)

        if roles_diag:
            combined = {
                "clip": video_stem,
                "metadata": {
                    "frames_processed": total,
                    "frames_with_pose": frames_with_pose,
                    "roles": list(roles_diag.keys()),
                    "source_fps_assumed": source_fps,
                },
                "roles": roles_diag,
            }
            out_json.write_text(json.dumps(combined, indent=2))
            print(f"  → {out_json.name}")
        else:
            print(f"  → no pose matches for any role; JSON skipped", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Overlay pose skeletons on tackle frames and generate "
            "injury-risk diagnostic JSON using ground-truth YOLO annotations."
        )
    )
    p.add_argument("--frames-dir",  type=Path, default=RAWS_FRAMES_DIR)
    p.add_argument("--labels-dir",  type=Path, default=RAWS_LABELS_DIR)
    p.add_argument("--output-dir",  type=Path, default=FINALS_DIR)
    p.add_argument("--weights",     default=POSE_WEIGHTS)
    p.add_argument("--conf",        type=float, default=POSE_CONF)
    p.add_argument("--device",      default=None,
                   help="Torch device: cpu, cuda:0, mps (default: auto)")
    p.add_argument("--min-iou",     type=float, default=ANNOTATION_MIN_IOU)
    p.add_argument("--output-fps",  type=float, default=OUTPUT_FPS,
                   help="FPS of the stitched output video")
    p.add_argument("--source-fps",  type=float, default=SOURCE_FPS,
                   help="Original capture frame rate (used for kinematics)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_pose_pipeline(
        args.frames_dir.expanduser().resolve(),
        args.labels_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        weights=args.weights,
        conf=args.conf,
        device=args.device,
        role_config=ROLE_CONFIG,
        min_iou=args.min_iou,
        output_fps=args.output_fps,
        source_fps=args.source_fps,
    )


if __name__ == "__main__":
    main()
