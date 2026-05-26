"""
Component 1 — Head & Neck biomechanical feature extractor.

Rubric levels:
  1  Head rotated away from ball carrier (avoidance)
  2  Head down — initiated contact with neck flexed
  3  Head up — initiated contact, in front of carrier
  4  Head accurately behind/to side, aligned and stable neck/head position

Metrics computed (per tackler, over window from POC onward):
  neck_flexion_deg         angle at neck (nose→midShoulder→midHip); 0=upright, 90+=head-down
  head_roll_deg            lateral ear-to-ear tilt from horizontal; 0=level
  ear_below_shoulder_frac  fraction of frames where either ear is below shoulder level (head-down cue)
  nose_above_shoulder_frac fraction of frames where nose is above mid-shoulder height (head-up cue)
  head_centered_frac       fraction where nose is laterally between the shoulders (not rotated away)
  head_kp_valid_frac       fraction of frames with usable head keypoints

Usage:
  python scripts/extract_head_neck.py --input-dir /path/to/videos --output head_neck.json
  python scripts/extract_head_neck.py --input-dir raws/ --output out/head_neck.json --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._biomech_utils import (
    KP_CONF_THRESH,
    LEFT_EAR, RIGHT_EAR,
    LEFT_EYE, RIGHT_EYE,
    LEFT_HIP, RIGHT_HIP,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    NOSE,
    aggregate,
    available_kp_frac,
    frac_true,
    kpt,
    median_torso_px,
    midpoint,
    run_tackle_detection_pass,
    select_tackle_pair_and_roles,
    extract_tackler_keypoints_window,
    video_id_from_path,
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".wmv"}
TOOL_VERSION   = "head-neck-extractor 0.1.0"
MAX_FRAMES     = 4500


# ---------------------------------------------------------------------------
# Per-frame metric computation
# ---------------------------------------------------------------------------

def _neck_flexion_deg(frame_kpts: np.ndarray | None, conf: float) -> float | None:
    """
    Angle at mid-shoulder between the nose and mid-hip directions.
    0° = fully upright, >90° = severely flexed (head-down).
    """
    nose = kpt(frame_kpts, NOSE, conf)
    ls   = kpt(frame_kpts, LEFT_SHOULDER,  conf)
    rs   = kpt(frame_kpts, RIGHT_SHOULDER, conf)
    lh   = kpt(frame_kpts, LEFT_HIP,  conf)
    rh   = kpt(frame_kpts, RIGHT_HIP, conf)
    ms   = midpoint(ls, rs)
    mh   = midpoint(lh, rh)
    if nose is None or ms is None or mh is None:
        return None
    ba = nose - ms
    bc = mh - ms
    na, nb = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nb < 1e-6:
        return None
    cos_a = np.clip(np.dot(ba, bc) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _head_roll_deg(frame_kpts: np.ndarray | None, conf: float) -> float | None:
    """
    Lateral tilt of the head measured from the ear-to-ear line.
    0° = level, positive = tilted clockwise (right ear lower).
    Falls back to eye-to-eye line when ears are not visible.
    """
    le = kpt(frame_kpts, LEFT_EAR,  conf)
    re = kpt(frame_kpts, RIGHT_EAR, conf)
    if le is not None and re is not None:
        ang = float(np.degrees(np.arctan2(le[1] - re[1], re[0] - le[0])))
        return ang
    # fallback: eye line
    ley = kpt(frame_kpts, LEFT_EYE,  conf)
    rey = kpt(frame_kpts, RIGHT_EYE, conf)
    if ley is not None and rey is not None:
        return float(np.degrees(np.arctan2(ley[1] - rey[1], rey[0] - ley[0])))
    return None


def _ear_below_shoulder(frame_kpts: np.ndarray | None, conf: float) -> bool | None:
    """
    True when either ear is below the corresponding shoulder (image y increases downward).
    Indicates severe neck flexion / head-down posture (Level 2 signal).
    """
    le = kpt(frame_kpts, LEFT_EAR,      conf)
    re = kpt(frame_kpts, RIGHT_EAR,     conf)
    ls = kpt(frame_kpts, LEFT_SHOULDER, conf)
    rs = kpt(frame_kpts, RIGHT_SHOULDER, conf)

    left_below  = (le is not None and ls is not None and le[1] > ls[1])
    right_below = (re is not None and rs is not None and re[1] > rs[1])

    if le is None and re is None:
        return None
    return left_below or right_below


def _nose_above_shoulder(frame_kpts: np.ndarray | None, conf: float) -> bool | None:
    """
    True when nose is above (smaller y) the mid-shoulder.
    Indicates head-up posture (Level 3-4 signal).
    """
    nose = kpt(frame_kpts, NOSE, conf)
    ls   = kpt(frame_kpts, LEFT_SHOULDER,  conf)
    rs   = kpt(frame_kpts, RIGHT_SHOULDER, conf)
    ms   = midpoint(ls, rs)
    if nose is None or ms is None:
        return None
    return bool(nose[1] < ms[1])


def _head_centered(frame_kpts: np.ndarray | None, conf: float) -> bool | None:
    """
    True when nose x-coordinate falls between left- and right-shoulder x range.
    Low fraction signals head rotated away from the carrier (Level 1 signal).
    """
    nose = kpt(frame_kpts, NOSE, conf)
    ls   = kpt(frame_kpts, LEFT_SHOULDER,  conf)
    rs   = kpt(frame_kpts, RIGHT_SHOULDER, conf)
    if nose is None or ls is None or rs is None:
        return None
    x_min = min(ls[0], rs[0])
    x_max = max(ls[0], rs[0])
    # Add 20% margin on each side to avoid being too sensitive to minor turns.
    margin = 0.20 * (x_max - x_min + 1e-6)
    return bool(x_min - margin <= nose[0] <= x_max + margin)


def _head_kp_valid(frame_kpts: np.ndarray | None, conf: float) -> bool:
    """True when at least one head-region keypoint is confident."""
    if frame_kpts is None:
        return False
    head_indices = [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]
    return any(float(frame_kpts[i, 2]) >= conf for i in head_indices if i < frame_kpts.shape[0])


# ---------------------------------------------------------------------------
# Per-clip computation
# ---------------------------------------------------------------------------

def compute_head_neck_metrics(
    kpts_window: list[np.ndarray | None],
    poc_relative_idx: int = 0,
    conf: float = KP_CONF_THRESH,
) -> dict:
    """
    Given per-frame COCO-17 keypoints for the tackler over a window,
    return the head/neck component summary dict.
    poc_relative_idx: index within kpts_window that corresponds to POC.
    """
    n = len(kpts_window)
    poc_kpts = kpts_window[poc_relative_idx] if poc_relative_idx < n else None

    # Per-frame signals
    neck_flex   = [_neck_flexion_deg(k, conf)    for k in kpts_window]
    head_roll   = [_head_roll_deg(k, conf)        for k in kpts_window]
    ear_below   = [_ear_below_shoulder(k, conf)   for k in kpts_window]
    nose_above  = [_nose_above_shoulder(k, conf)  for k in kpts_window]
    head_center = [_head_centered(k, conf)         for k in kpts_window]
    head_valid  = [_head_kp_valid(k, conf)         for k in kpts_window]

    # at_poc values
    nf_at_poc  = _neck_flexion_deg(poc_kpts, conf)
    roll_at_poc = _head_roll_deg(poc_kpts, conf)

    summary = {
        "neck_flexion_deg":         aggregate(neck_flex,  nf_at_poc),
        "head_roll_deg":            aggregate(head_roll,  roll_at_poc),
        "ear_below_shoulder_frac":  frac_true(ear_below),
        "nose_above_shoulder_frac": frac_true(nose_above),
        "head_centered_frac":       frac_true(head_center),
        "head_kp_valid_frac":       frac_true(head_valid),
    }
    metrics = list(summary.keys())
    return {"summary": summary, "metrics": metrics, "level": None, "time_series": None}


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def process_video(
    video_path: Path,
    detect_model,
    pose_model,
    *,
    conf: float,
    ball_conf: float,
    device: str | None,
    window_frames: int,
    pose_conf: float,
    kp_conf: float,
    stride: int,
) -> dict | None:
    """Run the full extraction pipeline for one video. Returns clip dict or None on failure."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps    = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_diag = float(np.hypot(width, height))

    frames: list[np.ndarray] = []
    while len(frames) < MAX_FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    if not frames:
        return None

    motion_sum, presence, frame_records = run_tackle_detection_pass(
        frames, detect_model,
        conf=conf, ball_conf=ball_conf, device=device,
    )

    n_tracks = len(presence)
    n_with_motion = len(motion_sum)
    if n_tracks < 2:
        print(f"  [WARN] {video_path.name}: only {n_tracks} track(s) detected — skipping")
        return None
    if n_with_motion < 2:
        print(f"  [WARN] {video_path.name}: {n_tracks} track(s) detected but only "
              f"{n_with_motion} had motion across frames (try --conf lower, e.g. 0.15)")

    tackler_tid, _, poc_idx = select_tackle_pair_and_roles(
        frames, frame_records, motion_sum, presence, frame_diag, conf=conf,
    )
    if poc_idx is None:
        poc_idx = 0

    kpts_window = extract_tackler_keypoints_window(
        frames, frame_records, tackler_tid, poc_idx, window_frames,
        pose_model,
        pose_conf=pose_conf, device=device, stride=stride,
    )

    if not kpts_window:
        return None

    torso_px = median_torso_px(kpts_window, kp_conf)
    window_start = poc_idx
    window_end   = min(len(frames) - 1, poc_idx + window_frames - 1)
    avail_frac   = available_kp_frac(kpts_window, kp_conf)

    head_neck = compute_head_neck_metrics(kpts_window, poc_relative_idx=0, conf=kp_conf)

    return {
        "schema_version": "components.v1",
        "video_id": video_id_from_path(video_path),
        "meta": {
            "fps":                        fps,
            "frame_stride":               stride,
            "poc_index":                  poc_idx,
            "poc_time_s":                 poc_idx / fps,
            "window_frames":              len(kpts_window),
            "window_start_index":         window_start,
            "window_end_index":           window_end,
            "normalization_torso_length_px": torso_px,
            "tool_version":               TOOL_VERSION,
        },
        "keypoints": {
            "format":                   "COCO-17",
            "coords":                   "pixel",
            "confidence_threshold":     kp_conf,
            "available_keypoints_frac": avail_frac,
        },
        "components": {
            "head_neck":       head_neck,
            "upper_extremity": {"summary": {}, "metrics": [], "level": None, "time_series": None},
            "com_spine":       {"summary": {}, "metrics": [], "level": None, "time_series": None},
            "lower_extremity": {"summary": {}, "metrics": [], "level": None, "time_series": None},
        },
        "global": {"quality_flag": avail_frac >= 0.5},
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract head/neck biomechanical features from tackle clips.")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory containing tackle video files.")
    p.add_argument("--output",    type=Path, required=True,
                   help="Output JSON file (multi-clip wrapper format).")
    p.add_argument("--weights",       default="yolo11n.pt",
                   help="YOLO detection model weights (default: yolo11n.pt).")
    p.add_argument("--pose-weights",  default="yolo11n-pose.pt",
                   help="YOLO pose model weights (default: yolo11n-pose.pt).")
    p.add_argument("--conf",          type=float, default=0.15)
    p.add_argument("--ball-conf",     type=float, default=0.15)
    p.add_argument("--pose-conf",     type=float, default=0.18)
    p.add_argument("--kp-conf",       type=float, default=KP_CONF_THRESH,
                   help="Keypoint confidence threshold for metric computation.")
    p.add_argument("--device",        default=None,
                   help="Torch device: mps, cuda:0, cpu (default: auto).")
    p.add_argument("--window-frames", type=int, default=30,
                   help="Number of frames after POC to analyse (default: 30).")
    p.add_argument("--stride",        type=int, default=1,
                   help="Frame stride within the window (default: 1).")
    args = p.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Missing dependency: pip install ultralytics")

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")

    videos = sorted(p for p in input_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        sys.exit(f"No video files found in {input_dir}")

    print(f"Loading detection model: {args.weights}")
    detect_model = YOLO(args.weights)
    print(f"Loading pose model:      {args.pose_weights}")
    pose_model   = YOLO(args.pose_weights)

    clips: dict[str, dict] = {}
    for idx, vp in enumerate(videos, 1):
        vid = video_id_from_path(vp)
        print(f"[{idx}/{len(videos)}] {vp.name}")
        try:
            result = process_video(
                vp, detect_model, pose_model,
                conf=args.conf, ball_conf=args.ball_conf, device=args.device,
                window_frames=args.window_frames, pose_conf=args.pose_conf,
                kp_conf=args.kp_conf, stride=args.stride,
            )
            if result is None:
                print(f"  [SKIP] {vp.name}: no result")
                continue
            clips[vid] = result
            hn = result["components"]["head_neck"]
            nf = hn["summary"].get("neck_flexion_deg")
            val = f'{nf["mean"]:.1f}°' if isinstance(nf, dict) and nf.get("mean") is not None else "n/a"
            print(f"  neck_flexion mean={val}  kp_avail={result['keypoints']['available_keypoints_frac']:.2f}")
        except Exception as e:
            print(f"  [ERROR] {vp.name}: {e}", file=sys.stderr)
            traceback.print_exc()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": "components.v1",
        "tool_version":   TOOL_VERSION,
        "exported_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clip_count":     len(clips),
        "clips":          clips,
    }
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {len(clips)} clip(s) → {args.output}")


if __name__ == "__main__":
    main()
