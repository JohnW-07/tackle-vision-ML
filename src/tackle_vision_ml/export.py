from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

def export_features_json(
    *,
    input_path: Path,
    output_dir: Path,
    fps: float,
    frame_indices: list[int],
    frames: list[np.ndarray],
    selection: dict[str, Any],
    pose2d: dict[str, Any],
    pose3d: dict[str, Any],
    kinematics: dict[str, Any],
    impact: dict[str, Any],
    output_annotated_video: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """
    Export final features for external safety classification.

    Output contract:
      - JSON file per clip
      - optional annotated video with skeleton overlay
    """
    joints = list(kinematics["joint_names"])
    pos = np.asarray(kinematics["pos"], dtype=np.float32)
    vel = np.asarray(kinematics["vel"], dtype=np.float32)
    acc = np.asarray(kinematics["acc"], dtype=np.float32)
    jerk = np.asarray(kinematics["jerk"], dtype=np.float32)
    acc_mag = np.asarray(kinematics["acc_mag"], dtype=np.float32)

    T, J, _ = pos.shape
    joint_payload: dict[str, Any] = {}
    for j_name, j_idx in zip(joints, range(J)):
        acc_j = acc_mag[:, j_idx]
        peak_i = int(np.nanargmax(np.nan_to_num(acc_j, nan=-np.inf)))
        joint_payload[j_name] = {
            "pos": pos[:, j_idx, :].tolist(),
            "vel": vel[:, j_idx, :].tolist(),
            "acc": acc[:, j_idx, :].tolist(),
            "jerk": jerk[:, j_idx, :].tolist(),
            "max_acceleration": float(acc_j[peak_i]),
            "time_of_peak_acceleration_seconds": float(peak_i / max(1e-6, fps)),
            "frame_of_peak_acceleration": int(frame_indices[peak_i]),
        }

    torso_decel = np.asarray(kinematics["torso_decel"], dtype=np.float32)
    torso_peak_i = int(np.nanargmax(np.nan_to_num(torso_decel, nan=-np.inf)))

    payload: dict[str, Any] = {
        "meta": {
            "clip_name": input_path.name,
            "frame_count": int(T),
            "fps": float(fps),
            "dt": float(1.0 / max(1e-6, fps)),
            "selected_track_id": int(selection.get("track_id", -1)),
            "models": {
                "detector_tracker": "YOLOv8+ByteTrack",
                "pose2d": "YOLOv8-pose/compatible",
                "pose3d": pose3d.get("source_lifting_meta", {}).get("method", "unknown"),
            },
        },
        "impact": {
            "impact_frame_index": int(impact["impact_frame_index"]),
            "impact_time_seconds": float(impact["impact_time_seconds"]),
            "window_frame_range": list(impact["window_frame_range"]),
        },
        "time_series": joint_payload,
        "summary": {
            "per_joint": {
                j: {
                    "max_acceleration": joint_payload[j]["max_acceleration"],
                    "time_of_peak_acceleration_seconds": joint_payload[j]["time_of_peak_acceleration_seconds"],
                }
                for j in joints
            },
            "torso_deceleration": {
                "max_deceleration": float(torso_decel[torso_peak_i]),
                "time_of_max_deceleration_seconds": float(torso_peak_i / max(1e-6, fps)),
                "frame_of_max_deceleration": int(frame_indices[torso_peak_i]),
            },
        },
    }

    if output_annotated_video:
        from tackle_vision_ml.video_io import write_annotated_pose2d_video

        annotated_path = output_dir / f"{input_path.stem}.annotated.mp4"
        write_annotated_pose2d_video(
            output_path=annotated_path,
            frames_rgb=frames,
            keypoints_2d=np.asarray(pose2d["keypoints_2d"], dtype=np.float32),
            keypoints_conf=np.asarray(pose2d["keypoints_conf"], dtype=np.float32),
            fps=fps,
        )
        payload["meta"]["annotated_video"] = str(annotated_path)

    output_path = output_dir / f"{input_path.stem}.kinematics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path, payload

