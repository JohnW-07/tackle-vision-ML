from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tackle_vision_ml.config import PipelineConfig
from tackle_vision_ml.pipeline import run_clip


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tackle-vision-ml",
        description="Extract normalized tackling joint kinematics features from a short video clip.",
    )
    p.add_argument("--input", required=True, type=str, help="Path to input video clip")
    p.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to write per-clip JSON (and optional annotated video)",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    p.add_argument("--frame-stride", type=int, default=2, help="Sample every Nth frame for speed")
    p.add_argument("--max-frames", type=int, default=None, help="Optional cap for smoke tests")

    # Models
    p.add_argument("--yolo-detect-model", default=PipelineConfig.yolo_detect_model)
    p.add_argument("--yolo-pose-model", default=PipelineConfig.yolo_pose_model)
    p.add_argument(
        "--pose3d-checkpoint",
        default=None,
        type=str,
        help="Optional local VideoPose3D checkpoint path (if required by your setup)",
    )

    # Temporal lifting / smoothing
    p.add_argument("--pose3d-window-size", type=int, default=PipelineConfig.pose3d_window_size)
    p.add_argument("--pose3d-stride", type=int, default=PipelineConfig.pose3d_stride)
    p.add_argument("--torso-length-aggregation", default=PipelineConfig.torso_length_aggregation)
    p.add_argument("--smoothing-method", default=PipelineConfig.smoothing_method)
    p.add_argument("--smoothing-window", type=int, default=PipelineConfig.smoothing_window)
    p.add_argument("--smoothing-polyorder", type=int, default=PipelineConfig.smoothing_polyorder)
    p.add_argument("--missing-max-gap", type=int, default=PipelineConfig.missing_max_gap)

    # Impact
    p.add_argument("--impact-window-radius", type=int, default=PipelineConfig.impact_window_radius)

    p.add_argument("--annotated-video", action="store_true", help="Write an optional skeleton overlay video")
    p.add_argument("--version", action="store_true", help="Print config version")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.version:
        print("tackle-vision-ml 0.1.0")
        return 0

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        device=args.device,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        yolo_detect_model=args.yolo_detect_model,
        yolo_pose_model=args.yolo_pose_model,
        pose3d_checkpoint=args.pose3d_checkpoint,
        pose3d_window_size=args.pose3d_window_size,
        pose3d_stride=args.pose3d_stride,
        torso_length_aggregation=args.torso_length_aggregation,
        smoothing_method=args.smoothing_method,
        smoothing_window=args.smoothing_window,
        smoothing_polyorder=args.smoothing_polyorder,
        missing_max_gap=args.missing_max_gap,
        impact_window_radius=args.impact_window_radius,
        output_annotated_video=args.annotated_video,
    )

    result_path, result_obj = run_clip(input_path, output_dir, config)

    # Print a small summary (useful for CLI/batch scripts)
    print(json.dumps({"output_json": str(result_path), "meta": result_obj.get("meta", {})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

