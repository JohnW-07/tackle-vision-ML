"""
TackleVision ML inference server.

POST /analyze   — upload a video file, get back an AnalysisResult JSON
GET  /health    — liveness check

Start with:
  python api/server.py
  # or
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "out" / "model.pkl"

def _load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)

_model_bundle = None

def get_model():
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = _load_model()
    return _model_bundle


# ---------------------------------------------------------------------------
# COCO-17 upper extremity extraction (mirrors head_neck / lower_extremity scripts)
# ---------------------------------------------------------------------------
from scripts._biomech_utils import (
    KP_CONF_THRESH,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_HIP, RIGHT_HIP,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_WRIST, RIGHT_WRIST,
    aggregate,
    angle_3pt,
    available_kp_frac,
    extract_tackler_keypoints_window,
    frac_true,
    kpt,
    median_torso_px,
    midpoint,
    run_tackle_detection_pass,
    select_tackle_pair_and_roles,
    torso_length_px,
)
from scripts.extract_head_neck import compute_head_neck_metrics
from scripts.extract_lower_extremity import compute_lower_extremity_metrics

ARM_EXTENSION_THRESH_DEG = 160.0  # elbow angle >= this = arm extended


def _elbow_flex_deg(frame_kpts, side: str, conf: float) -> float | None:
    sh_i  = LEFT_SHOULDER  if side == "left" else RIGHT_SHOULDER
    el_i  = LEFT_ELBOW     if side == "left" else RIGHT_ELBOW
    wr_i  = LEFT_WRIST     if side == "left" else RIGHT_WRIST
    sh = kpt(frame_kpts, sh_i, conf)
    el = kpt(frame_kpts, el_i, conf)
    wr = kpt(frame_kpts, wr_i, conf)
    if sh is None or el is None or wr is None:
        return None
    return angle_3pt(sh, el, wr)


def _shoulder_flex_deg(frame_kpts, side: str, conf: float) -> float | None:
    lh = kpt(frame_kpts, LEFT_HIP,  conf)
    rh = kpt(frame_kpts, RIGHT_HIP, conf)
    mh = midpoint(lh, rh)
    sh_i = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
    el_i = LEFT_ELBOW    if side == "left" else RIGHT_ELBOW
    sh = kpt(frame_kpts, sh_i, conf)
    el = kpt(frame_kpts, el_i, conf)
    if mh is None or sh is None or el is None:
        return None
    return angle_3pt(mh, sh, el)


def _wrist_shoulder_dist_norm(frame_kpts, side: str, torso_len: float | None, conf: float) -> float | None:
    sh_i = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
    wr_i = LEFT_WRIST    if side == "left" else RIGHT_WRIST
    sh = kpt(frame_kpts, sh_i, conf)
    wr = kpt(frame_kpts, wr_i, conf)
    if sh is None or wr is None or torso_len is None or torso_len < 1e-6:
        return None
    return float(np.linalg.norm(wr - sh)) / torso_len


def _upper_kp_valid(frame_kpts, conf: float) -> bool:
    if frame_kpts is None:
        return False
    upper = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
    return any(float(frame_kpts[i, 2]) >= conf for i in upper if i < frame_kpts.shape[0])


def compute_upper_extremity_metrics(
    kpts_window: list[np.ndarray | None],
    poc_relative_idx: int = 0,
    conf: float = KP_CONF_THRESH,
) -> dict:
    n = len(kpts_window)
    poc_kpts = kpts_window[poc_relative_idx] if poc_relative_idx < n else None

    el_l = [_elbow_flex_deg(k, "left",  conf) for k in kpts_window]
    el_r = [_elbow_flex_deg(k, "right", conf) for k in kpts_window]
    sh_l = [_shoulder_flex_deg(k, "left",  conf) for k in kpts_window]
    sh_r = [_shoulder_flex_deg(k, "right", conf) for k in kpts_window]

    torso_series = [torso_length_px(k, conf) for k in kpts_window]
    ws_l = [_wrist_shoulder_dist_norm(k, "left",  t, conf) for k, t in zip(kpts_window, torso_series)]
    ws_r = [_wrist_shoulder_dist_norm(k, "right", t, conf) for k, t in zip(kpts_window, torso_series)]

    # Derived series
    el_max = [max(a, b) for a, b in zip(el_l, el_r) if a is not None and b is not None]
    el_min_series = [min(a, b) for a, b in zip(el_l, el_r) if a is not None and b is not None]
    asym_deg  = [abs(a - b) for a, b in zip(el_l, el_r) if a is not None and b is not None]
    asym_norm = [abs(a - b) / t for a, b, t in zip(ws_l, ws_r, torso_series)
                 if a is not None and b is not None and t is not None and t > 1e-6]

    lh = kpt(poc_kpts, LEFT_HIP,  conf)
    rh = kpt(poc_kpts, RIGHT_HIP, conf)
    ls = kpt(poc_kpts, LEFT_SHOULDER,  conf)
    rs = kpt(poc_kpts, RIGHT_SHOULDER, conf)
    ms_poc = midpoint(ls, rs) if ls is not None and rs is not None else None

    both_hands_above = [
        (kpt(k, LEFT_WRIST, conf) is not None and
         kpt(k, RIGHT_WRIST, conf) is not None and
         kpt(k, LEFT_SHOULDER, conf) is not None and
         kpt(k, RIGHT_SHOULDER, conf) is not None and
         kpt(k, LEFT_WRIST, conf)[1] < midpoint(kpt(k, LEFT_SHOULDER, conf), kpt(k, RIGHT_SHOULDER, conf))[1] and
         kpt(k, RIGHT_WRIST, conf)[1] < midpoint(kpt(k, LEFT_SHOULDER, conf), kpt(k, RIGHT_SHOULDER, conf))[1])
        if (kpt(k, LEFT_WRIST, conf) is not None and kpt(k, RIGHT_WRIST, conf) is not None and
            kpt(k, LEFT_SHOULDER, conf) is not None and kpt(k, RIGHT_SHOULDER, conf) is not None)
        else None
        for k in kpts_window
    ]
    any_hand_above = [
        (kpt(k, LEFT_WRIST, conf) is not None and kpt(k, LEFT_SHOULDER, conf) is not None and
         kpt(k, LEFT_WRIST, conf)[1] < kpt(k, LEFT_SHOULDER, conf)[1]) or
        (kpt(k, RIGHT_WRIST, conf) is not None and kpt(k, RIGHT_SHOULDER, conf) is not None and
         kpt(k, RIGHT_WRIST, conf)[1] < kpt(k, RIGHT_SHOULDER, conf)[1])
        if (kpt(k, LEFT_WRIST, conf) is not None or kpt(k, RIGHT_WRIST, conf) is not None)
        else None
        for k in kpts_window
    ]
    both_ext   = [bool(a >= ARM_EXTENSION_THRESH_DEG and b >= ARM_EXTENSION_THRESH_DEG)
                  if a is not None and b is not None else None for a, b in zip(el_l, el_r)]
    left_ext   = [bool(a >= ARM_EXTENSION_THRESH_DEG) if a is not None else None for a in el_l]
    right_ext  = [bool(b >= ARM_EXTENSION_THRESH_DEG) if b is not None else None for b in el_r]
    upper_valid = [_upper_kp_valid(k, conf) for k in kpts_window]

    el_l_poc = _elbow_flex_deg(poc_kpts, "left",  conf)
    el_r_poc = _elbow_flex_deg(poc_kpts, "right", conf)
    sh_l_poc = _shoulder_flex_deg(poc_kpts, "left",  conf)
    sh_r_poc = _shoulder_flex_deg(poc_kpts, "right", conf)
    tl_poc   = torso_length_px(poc_kpts, conf)
    ws_l_poc = _wrist_shoulder_dist_norm(poc_kpts, "left",  tl_poc, conf)
    ws_r_poc = _wrist_shoulder_dist_norm(poc_kpts, "right", tl_poc, conf)

    # el_max_series and el_min_series need to match kpts_window length for aggregate
    el_max_full = [max(a, b) if a is not None and b is not None else (a or b)
                   for a, b in zip(el_l, el_r)]
    el_min_full = [min(a, b) if a is not None and b is not None else (a or b)
                   for a, b in zip(el_l, el_r)]
    asym_deg_full  = [abs(a - b) if a is not None and b is not None else None
                      for a, b in zip(el_l, el_r)]
    asym_norm_full = [abs(a - b) / t if a is not None and b is not None and t is not None and t > 1e-6
                      else None for a, b, t in zip(ws_l, ws_r, torso_series)]

    el_max_poc = max(el_l_poc, el_r_poc) if el_l_poc is not None and el_r_poc is not None else (el_l_poc or el_r_poc)
    el_min_poc = min(el_l_poc, el_r_poc) if el_l_poc is not None and el_r_poc is not None else (el_l_poc or el_r_poc)

    summary = {
        "elbow_flex_left_deg":             aggregate(el_l,            el_l_poc),
        "elbow_flex_right_deg":            aggregate(el_r,            el_r_poc),
        "shoulder_flex_left_deg":          aggregate(sh_l,            sh_l_poc),
        "shoulder_flex_right_deg":         aggregate(sh_r,            sh_r_poc),
        "wrist_shoulder_dist_left_norm":   aggregate(ws_l,            ws_l_poc),
        "wrist_shoulder_dist_right_norm":  aggregate(ws_r,            ws_r_poc),
        "elbow_flex_max_deg":              aggregate(el_max_full,      el_max_poc),
        "elbow_flex_min_deg":              aggregate(el_min_full,      el_min_poc),
        "elbow_flex_asymmetry_deg":        aggregate(asym_deg_full,    abs(el_l_poc - el_r_poc) if el_l_poc is not None and el_r_poc is not None else None),
        "arm_extension_asymmetry_norm":    aggregate(asym_norm_full,   abs(ws_l_poc - ws_r_poc) if ws_l_poc is not None and ws_r_poc is not None else None),
        "both_hands_above_shoulders_frac": frac_true(both_hands_above),
        "any_hand_above_shoulder_frac":    frac_true(any_hand_above),
        "both_arms_extended_frac":         frac_true(both_ext),
        "left_arm_extended_frac":          frac_true(left_ext),
        "right_arm_extended_frac":         frac_true(right_ext),
        "upper_kp_valid_frac":             frac_true(upper_valid),
    }
    return {"summary": summary, "metrics": list(summary.keys()), "level": None, "time_series": None}


# ---------------------------------------------------------------------------
# Feature flattening (mirrors build_feature_csv.py)
# ---------------------------------------------------------------------------
AGGREGATE_STATS    = ("mean", "std", "min", "max", "p10", "p90", "at_poc")
DROP_STAT_SUFFIXES = ("_std", "_min", "_max", "_p10", "_p90")
COMPONENTS         = ("head_neck", "upper_extremity", "com_spine", "lower_extremity")


def flatten_components(components: dict) -> dict[str, float | None]:
    row: dict[str, float | None] = {}
    for comp in COMPONENTS:
        summary = components.get(comp, {}).get("summary", {})
        for metric, val in summary.items():
            if isinstance(val, dict):
                for stat in AGGREGATE_STATS:
                    row[f"{comp}.{metric}_{stat}"] = val.get(stat)
            else:
                row[f"{comp}.{metric}"] = val
    return row


def select_features(row: dict, feature_cols: list[str]) -> list[float | None]:
    """Return values in the exact column order the model expects, with None for missing."""
    return [row.get(col) for col in feature_cols]


# ---------------------------------------------------------------------------
# Score conversion helpers
# ---------------------------------------------------------------------------

def criterion_note(criterion_id: str, score: int) -> str:
    if criterion_id == 'head_position':
        if score >= 75:
            return "Head up and aligned throughout contact — safe posture"
        if score >= 55:
            return "Head position marginal — some neck flexion near contact"
        return "Head down at contact — neck flexion detected, injury risk elevated"
    if criterion_id == 'shoulder_contact':
        if score >= 75:
            return "Good shoulder positioning — lead shoulder used effectively"
        if score >= 55:
            return "Shoulder contact inconsistent — partial technique detected"
        return "Poor shoulder technique — crown or head leading into contact"
    if criterion_id == 'approach_angle':
        if score >= 75:
            return "Controlled trunk alignment and body control into contact"
        if score >= 55:
            return "Some trunk control — mild forward lean detected"
        return "Excessive forward lean — trunk not aligned at contact"
    if criterion_id == 'wrap_technique':
        if score >= 75:
            return "Full arm wrap with drive through the ball carrier"
        if score >= 55:
            return "Arm wrap inconsistent — partial or asymmetric technique"
        return "Poor arm technique — asymmetric or absent wrap"
    if criterion_id == 'follow_through':
        if score >= 75:
            return "Strong leg drive sustained through contact"
        if score >= 55:
            return "Partial leg drive — extension not fully sustained"
        return "No leg drive detected — tackle finished with upper body only"
    return ""


def model_to_100(score: float | None) -> int:
    """Convert model 1–4 scale to 0–100."""
    if score is None:
        return 50
    return round(max(0.0, min(100.0, ((score - 1.0) / 3.0) * 100.0)))


def severity_from_score(score: int) -> str:
    if score >= 75:
        return "low"
    if score >= 55:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------------

MAX_FRAMES    = 4500
WINDOW_FRAMES = 30
STRIDE        = 1
CONF          = 0.15
POSE_CONF     = 0.18


def run_inference(video_path: str, device: str | None = None) -> dict:
    """Run the full pipeline on a video file. Returns raw component scores."""
    from ultralytics import YOLO
    detect_model = YOLO("yolo11n.pt")
    pose_model   = YOLO("yolo11n-pose.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps        = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_diag = float(np.hypot(width, height))

    frames: list[np.ndarray] = []
    while len(frames) < MAX_FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    if not frames:
        raise ValueError("Video has no readable frames")

    # Detection pass
    motion_sum, presence, frame_records = run_tackle_detection_pass(
        frames, detect_model, conf=CONF, ball_conf=CONF, device=device,
    )

    if len(presence) < 2:
        # Retry at lower confidence before giving up
        motion_sum, presence, frame_records = run_tackle_detection_pass(
            frames, detect_model, conf=0.10, ball_conf=0.10, device=device,
        )
    if len(presence) < 2:
        raise ValueError(f"Only {len(presence)} player(s) detected — ensure the video shows two players")

    tackler_tid, _, poc_idx = select_tackle_pair_and_roles(
        frames, frame_records, motion_sum, presence, frame_diag, conf=CONF,
    )
    poc_idx = poc_idx or 0

    # Pose extraction — lower conf thresholds widen the window for difficult angles
    kpts_window = extract_tackler_keypoints_window(
        frames, frame_records, tackler_tid, poc_idx, WINDOW_FRAMES,
        pose_model, pose_conf=POSE_CONF, device=device, stride=STRIDE,
    )

    # Retry at lower pose confidence if the first pass yielded nothing
    if not kpts_window or available_kp_frac(kpts_window, KP_CONF_THRESH) < 0.15:
        kpts_window = extract_tackler_keypoints_window(
            frames, frame_records, tackler_tid, poc_idx, WINDOW_FRAMES,
            pose_model, pose_conf=0.10, device=device, stride=STRIDE,
        )

    # If still empty, produce an all-None window so the model can still run
    # with imputed medians from training — confidence will be 0 to flag it.
    if not kpts_window:
        kpts_window = [None] * min(WINDOW_FRAMES, len(frames) - poc_idx)

    avail_frac = available_kp_frac(kpts_window, KP_CONF_THRESH)

    # Per-component metrics
    head_neck_comp  = compute_head_neck_metrics(kpts_window, poc_relative_idx=0)
    lower_ext_comp  = compute_lower_extremity_metrics(kpts_window, fps, STRIDE, poc_relative_idx=0)
    upper_ext_comp  = compute_upper_extremity_metrics(kpts_window, poc_relative_idx=0)

    components = {
        "head_neck":       head_neck_comp,
        "upper_extremity": upper_ext_comp,
        "com_spine":       {"summary": {}, "metrics": [], "level": None, "time_series": None},
        "lower_extremity": lower_ext_comp,
    }

    return {"components": components, "avail_frac": avail_frac, "poc_idx": poc_idx, "fps": fps}


def predict(video_path: str, device: str | None = None) -> dict:
    """Full pipeline: video → AnalysisResult dict."""
    bundle = get_model()
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    target_cols  = bundle["target_cols"]

    pipeline_out = run_inference(video_path, device=device)
    components   = pipeline_out["components"]
    avail_frac   = pipeline_out["avail_frac"]

    # Flatten to feature row
    raw_row   = flatten_components(components)
    feat_vals = select_features(raw_row, feature_cols)

    import pandas as pd
    X = pd.DataFrame([feat_vals], columns=feature_cols)
    preds = model.predict(X)[0]

    scores = {col.replace("label_", ""): float(val) for col, val in zip(target_cols, preds)}

    # Map to AnalysisResult
    hn = model_to_100(scores.get("head_neck"))
    ue = model_to_100(scores.get("upper_extremity"))
    cs = model_to_100(scores.get("com_spine"))
    le = model_to_100(scores.get("lower_extremity"))
    oa = model_to_100(scores.get("overall_avg"))

    criteria = [
        {"id": "head_position",   "label": "Head position",
         "score": hn, "note": criterion_note("head_position",   hn)},
        {"id": "shoulder_contact","label": "Shoulder contact",
         "score": ue, "note": criterion_note("shoulder_contact", ue)},
        {"id": "approach_angle",  "label": "Approach angle",
         "score": cs, "note": criterion_note("approach_angle",   cs)},
        {"id": "wrap_technique",  "label": "Wrap technique",
         "score": ue, "note": criterion_note("wrap_technique",   ue)},
        {"id": "follow_through",  "label": "Follow-through",
         "score": le, "note": criterion_note("follow_through",   le)},
    ]

    flags: list[str] = []
    if hn < 60:
        flags.append("head_down_contact_risk")
    if oa < 55:
        flags.append("coach_review_recommended")
    if le < 50:
        flags.append("no_leg_drive_detected")
    if ue < 50:
        flags.append("arm_technique_risk")

    return {
        "version":      1,
        "overallScore": oa,
        "severity":     severity_from_score(oa),
        "criteria":     criteria,
        "confidence":   round(float(avail_frac), 2),
        "flags":        flags,
        "analyzedAt":   datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="TackleVision ML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        get_model()
        loaded = True
    except Exception:
        loaded = False
    return {"status": "ok", "model_loaded": loaded}


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        result = predict(tmp_path, device="mps")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    finally:
        os.unlink(tmp_path)

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
