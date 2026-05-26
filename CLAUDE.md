# tackle-vision-ML — Claude Reference

## Project purpose

Build an ML pipeline for **tackle safety scoring**:

1. Run YOLO-based video inference on tackle clips → identify Ball Carrier + Tackler, extract per-frame pose keypoints
2. Compute pose-derived biomechanical features (head/neck angle, knee angle, trunk angle, etc.)
3. Export per-video `*.components.json` and `*.kinematics.json` files (see `schema/components.v1.json`)
4. Flatten to a CSV of fixed summary scalars (mean, std, min, max, p10, p90, at_poc, frac_true)
5. Join with grader-averaged safety labels → train an ML model to predict safety scores

**4 label components + overall:** `head_neck`, `upper_extremity`, `com_spine`, `lower_extremity`  
Labels come from PDFs graded by multiple raters, averaged into `labels/labels_avg.json`.

---

## Repo structure

```
tackle_bbox_pipeline.py    # Main pipeline (2600+ lines) — primary entry point
tackle_pair_scorer.py      # Weighted tackle pair selection from per-frame detections
schema/components.v1.json  # JSON Schema for biomechanical component exports
train_tackler_detector.py  # Fine-tune YOLO on labeled tackle frames
extract_dataset_frames.py  # Export frames from raws/ for labeling (YOLO format)
augment.py                 # Temporally consistent clip augmentation (Albumentations)
upsample.py                # Upsample raws/ with SeedVR2 → upsampled/
clear.py                   # Delete generated files from augmented/ and finals/
dataset/data.yaml          # YOLO dataset config
.githooks/pre-commit       # Blocks committing video files
```

**Gitignored (local only):**
- `raws/`, `upsampled/`, `augmented/`, `finals/` — video artifacts
- `*.pt`, `*.pth` — model weights
- `runs/`, `dataset/images/`, `dataset/labels/` — YOLO training outputs
- `out/` — component JSON exports

---

## Pipeline modes

| Mode | How it works |
|------|-------------|
| `top_motion` *(default with --all)* | Buffer up to 4500 frames → track all persons → pick 2 by motion+interaction score → score possession → label Ball Carrier / Tackler → render with pose overlay |
| `climax` | Find peak interaction frame → lock that pair → interpolate boxes through clip → same possession/render logic |
| `heuristic` *(default single file)* | Streaming frame-by-frame: probabilistic ball tracker + carrier-from-ball + tackler-toward-carrier |
| `trained` | Single-class tackler detector (fine-tuned weights via `--weights`) |

**Canonical command:**
```bash
python3 tackle_bbox_pipeline.py --all --mode top_motion --device mps
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode top_motion --device mps
```

**With dedicated football model:**
```bash
python3 tackle_bbox_pipeline.py --all --mode top_motion \
  --football-weights path/to/football_best.pt \
  --football-conf 0.18 --football-roi-pad 0.35 --football-hold-frames 6 \
  --device mps
```

---

## Key architecture — `tackle_bbox_pipeline.py`

### Tackle pair selection (`_select_players_of_interest`)
Delegates to `tackle_pair_scorer.select_tackle_pair_from_sequence`, which runs:
- Per-frame weighted scoring: rapid closing, motion opposition, dynamic persistence, fast contact, action cluster, crowd penalty
- Temporal hysteresis (locked pair must be beaten by `SMOOTH_SWITCH_RATIO × 1.18 + 0.08` margin)
- Clip-level rescoring: adds motion prior (W=14.0) + presence prior (W=2.0)
- Hard rejects stationary pairs (< 6 dynamic frames or < 0.045 summed closing)
- Returns `(tid_a, tid_b, peak_frame_idx, confidence)`

### Ball carrier vs tackler (`_choose_ballcarrier_from_possession`)
Compares per-frame possession evidence for the two selected track IDs:
- YOLO ball detection near player upper body + `_ball_hands_association_score`
- `_football_appearance_score`: HSV brown/warm color + shape + size + proximity
- ROI fallback: `_detect_ball_from_carrier_roi` (morphological blob on HSV warm mask)
- Player with more `wins` (frames where their possession score wins by ≥ 0.08) → Ball Carrier; other → Tackler

### Ball tracker (`BallTrackState`)
Lightweight constant-velocity Kalman-style tracker with states: `visible / predicted / handoff / lost`.  
Per-frame scoring mixes: temporal consistency (0.42), appearance (0.24), chest-prior (0.28), det conf (0.06).

### Pose overlay
Second YOLO pose model (`yolo11n-pose.pt` default). Matched to player box by IOU ≥ 0.15.  
COCO-17 keypoints; draws skeleton + head rotation annotation.

---

## Biomechanical schema (`schema/components.v1.json`)

Each `*.components.json` export has:
```jsonc
{
  "schema_version": "components.v1",
  "video_id": "...",
  "meta": { "fps", "frame_stride", "poc_index", "poc_time_s", "window_frames", ... },
  "keypoints": { "format": "COCO-17", "coords": "pixel", "confidence_threshold": 0.25, ... },
  "components": {
    "head_neck": { "summary": { "<metric>": { "mean", "std", "min", "max", "p10", "p90", "at_poc" } }, "metrics": [...], "level": null, "time_series": null },
    "upper_extremity": { ... },
    "com_spine": { ... },
    "lower_extremity": { ... }
  },
  "exported_at": "<ISO8601>"
}
```

**Units:** angles = deg, angular velocity = deg/s, distances = normalized by torso length (0–1 fraction), fractions = 0–1, time = s or frames, missing = `null`.

**ML feature naming convention (flat CSV):** `<component>.<metric>_<stat>`, e.g.:
- `head_neck.neck_angle_deg_mean`
- `head_neck.ear_below_shoulder_frac_p90`
- `lower_extremity.knee_angle_deg_at_poc`
- `lower_extremity.knee_extension_vel_deg_s_mean`

---

## Planned but not yet implemented (as of this branch)

- `scripts/extract_biomech_table.py` — flatten `out/*.components.json` → CSV with the column names above
- Biomechanical extractor for `head_neck` (neck angle, ear-below-shoulder) and `lower_extremity` (knee angle, knee extension velocity, wrist lock, trunk angle) using tackler keypoints
- Schema validation tooling
- `labels/` folder (label ingestion was done in a prior session; check `pranav/head-neck-legdrive` history)

---

## Supporting scripts

| Script | Purpose |
|--------|---------|
| `train_tackler_detector.py` | Fine-tune YOLO on `dataset/` for single-class tackler detection |
| `extract_dataset_frames.py` | Sample frames from `raws/` → `dataset/images/{train,val}/` for labeling |
| `augment.py` | Generate temporally consistent augmented clips → `augmented/` |
| `upsample.py` | Upsample `raws/` with SeedVR2 → `upsampled/` |
| `clear.py` | Delete contents of `augmented/` and `finals/` |

---

## Dataset layout (trained mode)

```
dataset/
  data.yaml
  images/{train,val}/*.jpg    # gitignored
  labels/{train,val}/*.txt    # gitignored; YOLO format: "0 xc yc w h"
```

---

## Model weight constants (hot-swap in file header)

```python
DEFAULT_WEIGHTS = "yolo11n.pt"          # person detector
TOP_MOTION_POSE_MODEL = "yolo11n-pose.pt"  # pose model
BALL_CLASS_ID = 32                       # COCO sports ball
```

README references `yolo26x.pt` / `yolo26x-pose.pt` as preferred for quality.

---

## Git hooks & conventions

- **pre-commit** (`.githooks/pre-commit`): blocks staging any video file extension — run `git restore --staged -- <file>` to unstage
- All generated videos go in `finals/` (gitignored); commit scripts and configs only
- Branch `dev/john` is the main integration branch; `pranav/head-neck-legdrive` held biomech WIP

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics opencv-python numpy albumentations
```

Device flags: `--device mps` (Apple Silicon), `--device cuda:0` (NVIDIA), omit for auto.
