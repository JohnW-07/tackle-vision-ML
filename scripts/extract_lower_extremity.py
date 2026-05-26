"""
Component 4 — Lower Extremity biomechanical feature extractor.

Rubric levels:
  1  No leg drive forward at POC through the entire tackle (upright brace and hold)
  2  Early drive/dive at carrier and/or ineffective leg drive — does not drive carrier
     in direction of applied force at POC and through tackle
  3  Well-timed lower extremity triple extension (hips, knees, ankles) and leg drive
     at POC and through entire tackle; drives carrier in direction of applied force

Metrics computed (per tackler, over window from POC onward):
  knee_flex_left_deg          angle at left knee (left_hip → left_knee → left_ankle)
  knee_flex_right_deg         angle at right knee
  hip_flex_left_deg           angle at left hip (mid_spine → left_hip → left_knee)
  hip_flex_right_deg          angle at right hip
  knee_extension_vel_deg_s    frame-to-frame rate of mean knee-angle change (positive = extending)
  triple_extension_score      (mean_knee + mean_hip) / 360 — higher = more extended
  triple_extension_frac       fraction of frames where both knees ≥ 150° (near full extension)
  stance_width_norm           ankle-to-ankle distance / torso length
  leg_drive_frac              fraction of frames with active extension drive
                              (mean knee angle increasing AND hip below mid-shoulder)
  lower_kp_valid_frac         fraction of frames with usable lower-body keypoints

Usage:
  python scripts/extract_lower_extremity.py --input-dir /path/to/videos --output lower_ext.json
  python scripts/extract_lower_extremity.py --input-dir raws/ --output out/lower_ext.json --device mps
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
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    aggregate,
    available_kp_frac,
    frac_true,
    kpt,
    median_torso_px,
    midpoint,
    angle_3pt,
    run_tackle_detection_pass,
    select_tackle_pair_and_roles,
    extract_tackler_keypoints_window,
    video_id_from_path,
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".wmv"}
TOOL_VERSION   = "lower-extremity-extractor 0.1.0"
MAX_FRAMES     = 4500

# Threshold for "approaching full extension" — knee angle ≥ this = near extension
KNEE_EXTENSION_THRESH_DEG = 150.0


# ---------------------------------------------------------------------------
# Per-frame metric computation
# ---------------------------------------------------------------------------

def _knee_flex_deg(frame_kpts: np.ndarray | None, side: str, conf: float) -> float | None:
    """Angle at knee joint (hip → knee → ankle). 180° = fully extended, ~90° = bent."""
    if side == "left":
        hip_i, knee_i, ankle_i = LEFT_HIP,  LEFT_KNEE,  LEFT_ANKLE
    else:
        hip_i, knee_i, ankle_i = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE

    hip   = kpt(frame_kpts, hip_i,   conf)
    knee  = kpt(frame_kpts, knee_i,  conf)
    ankle = kpt(frame_kpts, ankle_i, conf)

    if hip is None or knee is None or ankle is None:
        return None
    return angle_3pt(hip, knee, ankle)


def _hip_flex_deg(frame_kpts: np.ndarray | None, side: str, conf: float) -> float | None:
    """
    Angle at hip joint (mid_shoulder → hip → knee).
    180° = fully upright/extended (standing tall), smaller = more forward-flexed.
    Approximates triple-extension hip drive; measured from the torso reference.
    """
    ls = kpt(frame_kpts, LEFT_SHOULDER,  conf)
    rs = kpt(frame_kpts, RIGHT_SHOULDER, conf)
    ms = midpoint(ls, rs)

    if side == "left":
        hip_i, knee_i = LEFT_HIP,  LEFT_KNEE
    else:
        hip_i, knee_i = RIGHT_HIP, RIGHT_KNEE

    hip  = kpt(frame_kpts, hip_i,  conf)
    knee = kpt(frame_kpts, knee_i, conf)

    if ms is None or hip is None or knee is None:
        return None
    return angle_3pt(ms, hip, knee)


def _mean_knee_deg(kl: float | None, kr: float | None) -> float | None:
    vals = [v for v in (kl, kr) if v is not None]
    return float(np.mean(vals)) if vals else None


def _triple_extension_score_frame(
    kl: float | None, kr: float | None,
    hl: float | None, hr: float | None,
) -> float | None:
    """Composite score: (mean_knee + mean_hip) / 360. Higher = more extended."""
    k_vals = [v for v in (kl, kr) if v is not None]
    h_vals = [v for v in (hl, hr) if v is not None]
    if not k_vals or not h_vals:
        return None
    mean_k = float(np.mean(k_vals))
    mean_h = float(np.mean(h_vals))
    return (mean_k + mean_h) / 360.0


def _both_knees_extended(kl: float | None, kr: float | None) -> bool | None:
    """True when both knees are ≥ KNEE_EXTENSION_THRESH_DEG (near full extension)."""
    if kl is None and kr is None:
        return None
    vals = [v for v in (kl, kr) if v is not None]
    return all(v >= KNEE_EXTENSION_THRESH_DEG for v in vals)


def _stance_width_norm(
    frame_kpts: np.ndarray | None,
    torso_len: float | None,
    conf: float,
) -> float | None:
    """Ankle-to-ankle horizontal distance normalized by torso length."""
    la = kpt(frame_kpts, LEFT_ANKLE,  conf)
    ra = kpt(frame_kpts, RIGHT_ANKLE, conf)
    if la is None or ra is None or torso_len is None or torso_len < 1e-6:
        return None
    return float(np.linalg.norm(la - ra)) / torso_len


def _lower_kp_valid(frame_kpts: np.ndarray | None, conf: float) -> bool:
    """True if at least one lower-body keypoint is confident."""
    if frame_kpts is None:
        return False
    lower_indices = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
    return any(float(frame_kpts[i, 2]) >= conf for i in lower_indices if i < frame_kpts.shape[0])


# ---------------------------------------------------------------------------
# Per-clip computation
# ---------------------------------------------------------------------------

def compute_lower_extremity_metrics(
    kpts_window: list[np.ndarray | None],
    fps: float,
    stride: int,
    poc_relative_idx: int = 0,
    conf: float = KP_CONF_THRESH,
) -> dict:
    """
    Given per-frame COCO-17 keypoints for the tackler over a window,
    return the lower_extremity component summary dict.
    """
    n = len(kpts_window)
    poc_kpts = kpts_window[poc_relative_idx] if poc_relative_idx < n else None

    # Per-frame signals
    kl_series = [_knee_flex_deg(k, "left",  conf) for k in kpts_window]
    kr_series = [_knee_flex_deg(k, "right", conf) for k in kpts_window]
    hl_series = [_hip_flex_deg(k, "left",   conf) for k in kpts_window]
    hr_series = [_hip_flex_deg(k, "right",  conf) for k in kpts_window]

    mean_knee_series = [_mean_knee_deg(kl, kr) for kl, kr in zip(kl_series, kr_series)]
    te_score_series  = [_triple_extension_score_frame(kl, kr, hl, hr)
                        for kl, kr, hl, hr in zip(kl_series, kr_series, hl_series, hr_series)]
    both_ext_series  = [_both_knees_extended(kl, kr) for kl, kr in zip(kl_series, kr_series)]

    # Stance width uses per-frame torso for normalization
    torso_series = [None] * n  # computed inline below
    sw_series: list[float | None] = []
    for i, k in enumerate(kpts_window):
        from scripts._biomech_utils import torso_length_px
        tl = torso_length_px(k, conf)
        torso_series[i] = tl
        sw_series.append(_stance_width_norm(k, tl, conf))

    lower_valid_series = [_lower_kp_valid(k, conf) for k in kpts_window]

    # Knee extension velocity (deg/s) — forward difference on mean knee angle
    dt_s = float(stride) / max(fps, 1e-6)
    vel_series: list[float | None] = [None]
    for i in range(1, n):
        prev, curr = mean_knee_series[i - 1], mean_knee_series[i]
        if prev is None or curr is None:
            vel_series.append(None)
        else:
            vel_series.append((curr - prev) / dt_s)

    # Leg drive: knee extending (vel > 0) AND hip is below mid-shoulder (forward lean)
    leg_drive_flags: list[bool | None] = []
    for i, k in enumerate(kpts_window):
        ls = kpt(k, LEFT_SHOULDER,  conf)
        rs = kpt(k, RIGHT_SHOULDER, conf)
        lh = kpt(k, LEFT_HIP,  conf)
        rh = kpt(k, RIGHT_HIP, conf)
        ms = midpoint(ls, rs)
        mh = midpoint(lh, rh)
        v  = vel_series[i]
        if ms is None or mh is None or v is None:
            leg_drive_flags.append(None)
        else:
            # Hip below mid-shoulder in image (y↓) = mh[1] > ms[1]
            forward_lean = bool(mh[1] > ms[1])
            leg_drive_flags.append(bool(v > 0.0 and forward_lean))

    # at_poc values
    kl_poc = _knee_flex_deg(poc_kpts, "left",  conf)
    kr_poc = _knee_flex_deg(poc_kpts, "right", conf)
    hl_poc = _hip_flex_deg(poc_kpts,  "left",  conf)
    hr_poc = _hip_flex_deg(poc_kpts,  "right", conf)
    mk_poc = _mean_knee_deg(kl_poc, kr_poc)
    te_poc = _triple_extension_score_frame(kl_poc, kr_poc, hl_poc, hr_poc)

    summary = {
        "knee_flex_left_deg":       aggregate(kl_series,       kl_poc),
        "knee_flex_right_deg":      aggregate(kr_series,       kr_poc),
        "hip_flex_left_deg":        aggregate(hl_series,       hl_poc),
        "hip_flex_right_deg":       aggregate(hr_series,       hr_poc),
        "knee_extension_vel_deg_s": aggregate(vel_series,      None),
        "triple_extension_score":   aggregate(te_score_series, te_poc),
        "triple_extension_frac":    frac_true(both_ext_series),
        "stance_width_norm":        aggregate(sw_series,       _stance_width_norm(poc_kpts, torso_series[0], conf)),
        "leg_drive_frac":           frac_true(leg_drive_flags),
        "lower_kp_valid_frac":      frac_true(lower_valid_series),
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

    torso_px   = median_torso_px(kpts_window, kp_conf)
    window_end = min(len(frames) - 1, poc_idx + window_frames - 1)
    avail_frac = available_kp_frac(kpts_window, kp_conf)

    lower_ext = compute_lower_extremity_metrics(
        kpts_window, fps, stride,
        poc_relative_idx=0, conf=kp_conf,
    )

    return {
        "schema_version": "components.v1",
        "video_id": video_id_from_path(video_path),
        "meta": {
            "fps":                           fps,
            "frame_stride":                  stride,
            "poc_index":                     poc_idx,
            "poc_time_s":                    poc_idx / fps,
            "window_frames":                 len(kpts_window),
            "window_start_index":            poc_idx,
            "window_end_index":              window_end,
            "normalization_torso_length_px": torso_px,
            "tool_version":                  TOOL_VERSION,
        },
        "keypoints": {
            "format":                   "COCO-17",
            "coords":                   "pixel",
            "confidence_threshold":     kp_conf,
            "available_keypoints_frac": avail_frac,
        },
        "components": {
            "head_neck":       {"summary": {}, "metrics": [], "level": None, "time_series": None},
            "upper_extremity": {"summary": {}, "metrics": [], "level": None, "time_series": None},
            "com_spine":       {"summary": {}, "metrics": [], "level": None, "time_series": None},
            "lower_extremity": lower_ext,
        },
        "global": {"quality_flag": avail_frac >= 0.5},
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Extract lower extremity biomechanical features from tackle clips.")
    p.add_argument("--input-dir",     type=Path, required=True)
    p.add_argument("--output",        type=Path, required=True)
    p.add_argument("--weights",       default="yolo11n.pt")
    p.add_argument("--pose-weights",  default="yolo11n-pose.pt")
    p.add_argument("--conf",          type=float, default=0.15)
    p.add_argument("--ball-conf",     type=float, default=0.15)
    p.add_argument("--pose-conf",     type=float, default=0.18)
    p.add_argument("--kp-conf",       type=float, default=KP_CONF_THRESH)
    p.add_argument("--device",        default=None)
    p.add_argument("--window-frames", type=int, default=30)
    p.add_argument("--stride",        type=int, default=1)
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
            le = result["components"]["lower_extremity"]
            kl = le["summary"].get("knee_flex_left_deg")
            val = f'{kl["mean"]:.1f}°' if isinstance(kl, dict) and kl.get("mean") is not None else "n/a"
            print(f"  knee_flex_left mean={val}  kp_avail={result['keypoints']['available_keypoints_frac']:.2f}")
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
