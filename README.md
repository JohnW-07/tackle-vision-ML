# tackle-vision-ML

Video tooling for tackle clips, centered on `tackle_bbox_pipeline.py`.

## Main Pipeline

`tackle_bbox_pipeline.py` detects and tracks players, identifies the likely ball
carrier and tackler, and writes an annotated video.

Recommended mode:

```bash
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode top_motion --device mps
python3 tackle_bbox_pipeline.py --all --input-dir raws --output finals/two_players --device mps
```

`top_motion` buffers up to 4500 frames, runs one YOLO tracking pass, keeps the
two highest-motion person tracks, uses football detections near the upper body to
choose the ball carrier, then labels the other player as tackler. It draws the
ball carrier, tackler, football, pose skeletons, and head-orientation text.

Defaults:

- Detector: `yolo11n.pt`
- Pose overlay: `yolo26x-pose.pt`
- Batch input: `raws/`
- Batch output: `finals/two_players/`
- Output suffix: `*_top_motion_<pose-model>.mp4`

Useful variants:

```bash
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode top_motion --pose-weights yolo11n-pose.pt
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode top_motion --no-pose
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode heuristic
python3 tackle_bbox_pipeline.py raws/clip.mp4 --mode trained --weights runs/detect/tackler/weights/best.pt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install ultralytics opencv-python numpy albumentations
```

Use `--device mps` on Apple Silicon, `--device cuda:0` on CUDA, or omit
`--device` for Ultralytics auto-selection.

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
