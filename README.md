# tackle-vision-ml

Augmentation-only starter for tackle video experiments.

## What It Does

Generates temporally consistent augmented variants of `.mp4` and `.mov` clips using:

- exposure and color shifts
- coarse occlusion / blocking
- horizontal flip, small rotation, and translation

## Layout

- `src/tackle_vision_augment.py`: augmentation implementation
- `augment.py`: root-level runner
- `clear.py`: clears generated files in `augmented/`
- `raws/`: default input directory
- `augmented/`: default output directory
- `upsampled/`: optional input directory via `-upsampled`

## Usage

```bash
python3 augment.py
python3 augment.py 3 -all
python3 augment.py 5 -exposure
python3 augment.py 2 -blocking -translation
python3 augment.py 3 -all -upsampled
```

After installing the package:

```bash
tackle-vision-ml 3 -all
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

## Notes

- `assign_label()` is a placeholder for future dataset labeling logic.
- The old ML pipeline was intentionally removed so this can be rebuilt cleanly.

## Top-motion pipeline (`tackle_bbox_pipeline.py`) — quick notes

- **stack**: `yolo11n.pt` + `yolo11n-pose.pt` only (no yolo26, no separate “legacy” mode)
- **top_motion**: buffer clip → one track pass → keep2 highest-motion person tracks → ball near upper body → carrier vs tackler → draw boxes + skeleton + head text
- **other modes**: `heuristic` (streaming tackler-ish box), `trained` (your `best.pt`)
- **outputs**: default `*_top_motion.mp4` next to input; batch vids often dropped in `finals/two_players/`
- **tradeoffs**: nano = fast/light, misses more than bigger weights; long clips capped ~4500 frames in code
- **removed**: yolo26 default, yolov8n+yolo11 compare run, interpolation experiment (reverted), `_top_motion_legacy` suffix
