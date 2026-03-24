from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuration knobs for the tackle-vision kinematic feature pipeline.
    """

    # Runtime
    device: str = "cpu"  # "cpu" or "cuda" (if supported by underlying libs)
    frame_stride: int = 2  # sample every Nth frame for speed
    max_frames: Optional[int] = None  # optional cap for smoke tests

    # Models (left as strings so users can pass local paths or standard names)
    yolo_detect_model: str = "yolov8n.pt"
    yolo_pose_model: str = "yolov8n-pose.pt"
    pose3d_checkpoint: Optional[str] = None  # expected local checkpoint path

    # Lifting / temporal stability
    pose3d_window_size: int = 27
    pose3d_stride: int = 5

    # Normalization
    torso_length_aggregation: str = "median"  # "median" or "per_frame"

    # Temporal smoothing
    smoothing_method: str = "savgol"  # "savgol" or "moving_average"
    smoothing_window: int = 11  # odd
    smoothing_polyorder: int = 2

    # Missing/noisy handling
    missing_max_gap: int = 5  # interpolate across gaps up to this many frames

    # Impact detection
    impact_window_radius: int = 15  # extract +/- radius frames around impact

    # Output
    output_annotated_video: bool = False

