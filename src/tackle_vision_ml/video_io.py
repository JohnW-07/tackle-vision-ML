from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_video(
    input_path: Path,
    *,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
) -> tuple[list[np.ndarray], float, list[int]]:
    """
    Load and sample frames from a video using OpenCV.

    Returns:
      frames_rgb: list of frames in RGB order (H, W, 3)
      fps: frames-per-second estimate from metadata (fallback to 30.0 if unavailable)
      frame_indices: original frame indices in the input video for each returned frame
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0

    frames_rgb: list[np.ndarray] = []
    frame_indices: list[int] = []

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if (frame_idx % frame_stride) == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
            frame_indices.append(frame_idx)
            if max_frames is not None and len(frames_rgb) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames_rgb, fps, frame_indices


def write_annotated_pose2d_video(
    *,
    output_path: Path,
    frames_rgb: list[np.ndarray],
    keypoints_2d: np.ndarray,
    keypoints_conf: np.ndarray,
    fps: float,
    conf_threshold: float = 0.2,
) -> Path:
    """
    Write a simple annotated video with 2D keypoints overlay.
    """
    if not frames_rgb:
        raise ValueError("frames_rgb is empty")

    H, W = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(max(1e-6, fps)), (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open writer for {output_path}")

    edges = [
        (5, 6),   # shoulders
        (5, 11),  # left torso
        (6, 12),  # right torso
        (11, 12), # hips
        (11, 13), # left thigh
        (12, 14), # right thigh
    ]

    for t, frame in enumerate(frames_rgb):
        out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pts = keypoints_2d[t]
        conf = keypoints_conf[t]

        for j in range(min(len(pts), len(conf))):
            if np.isnan(pts[j]).any() or float(conf[j]) < conf_threshold:
                continue
            x, y = int(pts[j, 0]), int(pts[j, 1])
            cv2.circle(out, (x, y), 3, (0, 255, 0), -1)

        for a, b in edges:
            if a >= len(pts) or b >= len(pts):
                continue
            if (
                np.isnan(pts[a]).any()
                or np.isnan(pts[b]).any()
                or float(conf[a]) < conf_threshold
                or float(conf[b]) < conf_threshold
            ):
                continue
            pa = (int(pts[a, 0]), int(pts[a, 1]))
            pb = (int(pts[b, 0]), int(pts[b, 1]))
            cv2.line(out, pa, pb, (255, 128, 0), 2)

        writer.write(out)

    writer.release()
    return output_path

