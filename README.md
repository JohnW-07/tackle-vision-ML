# tackle-vision-ML

Video tooling for tackle clips, centered on `tackle_bbox_pipeline.py`.

Additional detail: [TACKLEBBOX_PIPELINE.md](TACKLEBBOX_PIPELINE.md).

## Main Pipeline

`tackle_bbox_pipeline.py` detects and tracks players, estimates ball carrier vs tackler, optionally overlays pose skeletons, and writes an annotated video.

### Modes

| Mode | Behavior |
|------|----------|
| `top_motion` | Buffers up to 4500 frames, picks two players (motion + optional football support), scores possession over the clip, labels **Ball Carrier** / **Tackler**, draws football + pose. Default when `--all` is used. |
| `climax` | Uses peak interaction timing, interpolates tracks for smoother boxes. |
| `heuristic` | Streaming pass with ball tracker + carrier/tackler heuristics. |
| `trained` | Single-class tackler detector weights (`train_tackler_detector.py`). |

### Recommended commands

```bash
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode top_motion --device mps
python3 tackle_bbox_pipeline.py --all --input-dir raws --output finals/two_players --device mps
```

### Dedicated football model (recommended)

If COCO “sports ball” misses your football, pass a trained football-only checkpoint. The pipeline runs that model **only inside the two selected players’ boxes** (with padding), optionally re-ranks which two tracks matter using football support, then assigns carrier vs tackler.

```bash
python3 tackle_bbox_pipeline.py --all --mode top_motion \
  --input-dir raws --output finals/two_players \
  --weights yolo26x.pt \
  --football-weights path/to/best.pt \
  --football-conf 0.18 \
  --football-roi-pad 0.35 \
  --football-hold-frames 6 \
  --pose-weights yolo26x-pose.pt \
  --device mps
```

- **`--football-weights`**: omit to use the main detector’s COCO ball class only (via `--ball-conf`).
- **`--football-conf`**: confidence inside padded player ROI (default `0.18`).
- **`--football-roi-pad`**: expand each player box before football inference (default `0.35`).
- **`--football-hold-frames`**: reuse last football detection briefly when the detector flickers (default `6`).

`top_motion` still scores carrier vs tackler from upper-body / possession-style cues using those football detections; ROI blob fallback remains when the ball disappears.

### Defaults (see constants in `tackle_bbox_pipeline.py`)

- Person detector: `yolo26x.pt` (`--weights`)
- Pose: `yolo26x-pose.pt` (`--pose-weights`)
- Batch input directory (`--all`): `raws/`
- Batch output directory: `finals/two_players/`
- Output filename suffix for `top_motion`: `*_top_motion_<pose-model-stem>.mp4`

### Useful variants

```bash
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode top_motion --pose-weights yolo11n-pose.pt
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode top_motion --no-pose
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode top_motion --ball-conf 0.12
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode climax --device mps
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode heuristic
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode trained --weights runs/detect/tackler/weights/best.pt
```

## Football-only annotation (optional)

`testing/test.py` (if present beside this repo) annotates videos with a standalone football detector: set `WEIGHTS_PATH` and run against a folder of inputs. Generated videos belong under `.gitignore`; commit scripts and configs only.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install ultralytics opencv-python numpy albumentations
```

Use `--device mps` on Apple Silicon, `--device cuda:0` on CUDA, or omit `--device` for Ultralytics auto-selection.

## Supporting Scripts

- `upsample.py`: upsample videos from `raws/` into `upsampled/` with SeedVR2.
- `augment.py`: create temporally consistent augmented clips in `augmented/`.
- `extract_dataset_frames.py`: export frames into `dataset/images/{train,val}/`.
- `train_tackler_detector.py`: train a single-class tackler detector from
  `dataset/data.yaml`.
- `clear.py`: clear generated files from `augmented/`.

## Dataset Layout

For trained mode, label exported frames in YOLO format:

```text
dataset/images/train/*.jpg
dataset/images/val/*.jpg
dataset/labels/train/*.txt
dataset/labels/val/*.txt
```

Each label row is `0 xc yc w h`, normalized to 0-1, where class `0` is
`tackler`.
