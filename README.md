# tackle-vision-ml

Python ML pipeline for short football tackle clips that outputs stable, normalized joint kinematic time-series for downstream safety classification.

## What it outputs

- Relative (scale-normalized) 3D-like joint trajectories for:
  - `head`, `neck`, `left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`, `left_knee`, `right_knee`
- Per-joint time series:
  - `pos`, `vel`, `acc`, `jerk`
- Summary metrics:
  - per-joint max acceleration and time-of-peak
  - torso deceleration peak metrics
- Impact localization:
  - impact frame and `+/-15` window (configurable)
- Optional annotated video:
  - 2D skeleton/keypoint overlay on sampled frames

## Architecture

1. Player detection/tracking: YOLOv8 + ByteTrack
2. Tackling-player selection: motion + proximity/size heuristic
3. 2D pose: YOLOv8-pose on selected track crops
4. 3D lifting: VideoPose3D adapter interface + temporal fallback lifting
5. Coordinate normalization: centered at hips, scaled by shoulder-center to hip-center torso length
6. Temporal smoothing: Savitzky-Golay (or moving average)
7. Kinematics: velocity/acceleration/jerk via temporal derivatives
8. Impact detection: torso acceleration/deceleration cues
9. Export: per-clip JSON and optional annotated mp4

## Setup

```bash
cd tackle-vision-ML
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[vision,dev]"
```

Notes:
- CPU is the default (`--device cpu`).
- CUDA is optional (`--device cuda`) if your environment supports it.
- VideoPose3D checkpoints/repo layout vary; this code keeps an adapter hook and a temporal lifting fallback so the end-to-end pipeline still runs on laptop setups.

## CLI usage

Single clip:

```bash
tackle-vision-ml \
  --input /absolute/path/to/clip.mp4 \
  --output-dir ./out \
  --device cpu \
  --frame-stride 2 \
  --pose3d-window-size 27 \
  --pose3d-stride 5 \
  --torso-length-aggregation median \
  --smoothing-method savgol \
  --smoothing-window 11 \
  --smoothing-polyorder 2 \
  --impact-window-radius 15 \
  --annotated-video
```

Batch example (shell loop):

```bash
mkdir -p out
for f in /path/to/clips/*.mp4; do
  tackle-vision-ml --input "$f" --output-dir ./out --device cpu --frame-stride 2
done
```

## Example JSON output

File: `out/clip_name.kinematics.json`

```json
{
  "meta": {
    "clip_name": "clip_name.mp4",
    "frame_count": 120,
    "fps": 30.0,
    "dt": 0.0333333333,
    "selected_track_id": 4,
    "models": {
      "detector_tracker": "YOLOv8+ByteTrack",
      "pose2d": "YOLOv8-pose/compatible",
      "pose3d": "temporal_fallback"
    },
    "annotated_video": "out/clip_name.annotated.mp4"
  },
  "impact": {
    "impact_frame_index": 257,
    "impact_time_seconds": 2.3,
    "window_frame_range": [242, 272]
  },
  "time_series": {
    "head": {
      "pos": [[0.1, -0.2, 0.0]],
      "vel": [[0.0, 0.0, 0.0]],
      "acc": [[0.0, 0.0, 0.0]],
      "jerk": [[0.0, 0.0, 0.0]],
      "max_acceleration": 7.2,
      "time_of_peak_acceleration_seconds": 2.2667,
      "frame_of_peak_acceleration": 255
    }
  },
  "summary": {
    "per_joint": {
      "head": {
        "max_acceleration": 7.2,
        "time_of_peak_acceleration_seconds": 2.2667
      }
    },
    "torso_deceleration": {
      "max_deceleration": 5.4,
      "time_of_max_deceleration_seconds": 2.3,
      "frame_of_max_deceleration": 257
    }
  }
}
```

## Assumptions and limitations

- Single-camera input only.
- No labeled data is required.
- Values are relative (normalized), not absolute biomechanical units.
- Impact frame is estimated from motion cues, not ground-truth contact labels.
- 3D lifting is camera-space relative and root-centered by design.

## Project layout

- `src/tackle_vision_ml/cli.py`: CLI entrypoint
- `src/tackle_vision_ml/pipeline.py`: stage orchestration
- `src/tackle_vision_ml/tracking.py`: YOLOv8 + ByteTrack + tackler selection
- `src/tackle_vision_ml/pose2d.py`: 2D keypoint extraction for selected track
- `src/tackle_vision_ml/pose3d.py`: temporal 3D lifting adapter
- `src/tackle_vision_ml/normalization.py`: centering and torso-length scaling
- `src/tackle_vision_ml/smoothing.py`: temporal denoising + missing handling
- `src/tackle_vision_ml/kinematics.py`: derivatives and torso metrics
- `src/tackle_vision_ml/impact.py`: impact frame/window detection
- `src/tackle_vision_ml/export.py`: JSON export + optional annotated video

