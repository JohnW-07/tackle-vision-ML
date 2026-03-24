from __future__ import annotations

from pathlib import Path
from typing import Any

from tackle_vision_ml.config import PipelineConfig


def run_clip(input_path: Path, output_dir: Path, config: PipelineConfig) -> tuple[Path, dict[str, Any]]:
    """
    Orchestrate the full pipeline for a single input clip.

    Notes:
    - This file wires module interfaces; heavy implementation is filled in by subsequent to-dos.
    - The CLI and JSON export schema depend on stable interfaces created here.
    """

    # Import lazily so the package can be installed without vision dependencies.
    from tackle_vision_ml.video_io import load_video
    from tackle_vision_ml.tracking import track_and_select_tackling_player
    from tackle_vision_ml.pose2d import estimate_pose2d_for_track
    from tackle_vision_ml.pose3d import lift_pose2d_to_pose3d
    from tackle_vision_ml.normalization import center_and_normalize_skeleton
    from tackle_vision_ml.smoothing import smooth_time_series
    from tackle_vision_ml.kinematics import compute_kinematics
    from tackle_vision_ml.impact import detect_impact_window
    from tackle_vision_ml.export import export_features_json

    frames, fps, frame_indices = load_video(input_path, frame_stride=config.frame_stride, max_frames=config.max_frames)

    selection = track_and_select_tackling_player(
        frames=frames,
        fps=fps,
        frame_indices=frame_indices,
        device=config.device,
        yolo_detect_model=config.yolo_detect_model,
    )

    pose2d = estimate_pose2d_for_track(
        frames=frames,
        fps=fps,
        frame_indices=frame_indices,
        device=config.device,
        pose2d_model=config.yolo_pose_model,
        track=selection,
    )

    pose3d = lift_pose2d_to_pose3d(
        pose2d=pose2d,
        fps=fps,
        pose3d_window_size=config.pose3d_window_size,
        pose3d_stride=config.pose3d_stride,
        device=config.device,
        pose3d_checkpoint=config.pose3d_checkpoint,
    )

    normalized = center_and_normalize_skeleton(pose3d, torso_length_aggregation=config.torso_length_aggregation)
    smoothed = smooth_time_series(
        normalized,
        method=config.smoothing_method,
        window=config.smoothing_window,
        polyorder=config.smoothing_polyorder,
        missing_max_gap=config.missing_max_gap,
    )

    kinematics = compute_kinematics(smoothed, fps=fps)
    impact = detect_impact_window(kinematics, radius=config.impact_window_radius)

    output_path, payload = export_features_json(
        input_path=input_path,
        output_dir=output_dir,
        fps=fps,
        frame_indices=frame_indices,
        frames=frames,
        selection=selection,
        pose2d=pose2d,
        pose3d=smoothed,
        kinematics=kinematics,
        impact=impact,
        output_annotated_video=config.output_annotated_video,
    )

    return output_path, payload

