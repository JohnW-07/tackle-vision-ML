"""
Export frames from videos in `raws/` into `dataset/images/{train,val}/` for labeling.

Label files go in `dataset/labels/{train,val}/` with the same basename as each image
(YOLO format: one line per box: `class xc yc w h` normalized 0–1).

Example:
  python extract_dataset_frames.py
  python extract_dataset_frames.py --raws /path/to/videos --stride 3 --val-ratio 0.2
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _ensure_dirs(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _pick_val_videos(paths: list[Path], val_ratio: float, seed: int) -> set[Path]:
    if len(paths) <= 1:
        return set()
    n_val = max(1, int(round(len(paths) * val_ratio)))
    n_val = min(n_val, len(paths) - 1)
    rng = random.Random(seed)
    return set(rng.sample(paths, n_val))


def extract_frames(
    raws_dir: Path,
    dataset_root: Path,
    *,
    stride: int,
    val_ratio: float,
    seed: int,
    max_frames_per_video: int | None,
) -> int:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if not raws_dir.is_dir():
        print(f"Raw video directory not found: {raws_dir}", file=sys.stderr)
        return 1

    videos = sorted(
        p for p in raws_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        print(f"No videos found in {raws_dir} (supported: {sorted(VIDEO_SUFFIXES)})", file=sys.stderr)
        return 1

    _ensure_dirs(dataset_root)
    val_videos = _pick_val_videos(videos, val_ratio, seed)

    total_out = 0
    for video_path in videos:
        split = "val" if video_path in val_videos else "train"
        out_img_dir = dataset_root / "images" / split
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[skip] cannot open {video_path}", file=sys.stderr)
            continue

        stem = video_path.stem.replace(" ", "_")
        frame_idx = 0
        written = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            if max_frames_per_video is not None and written >= max_frames_per_video:
                break
            name = f"{stem}_{frame_idx:06d}.jpg"
            out_path = out_img_dir / name
            if not cv2.imwrite(str(out_path), frame_bgr):
                print(f"[skip] failed to write {out_path}", file=sys.stderr)
            else:
                total_out += 1
                written += 1
            frame_idx += 1
        cap.release()
        print(f"  {video_path.name} -> {split} ({written} frames)")

    print(f"Done. Wrote {total_out} images under {dataset_root / 'images'}.")
    print("Next: add matching .txt label files under dataset/labels/{train,val}/ then run train_tackler_detector.py")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export frames from raws/ into YOLO image folders.")
    p.add_argument("--raws", type=Path, default=PROJECT_ROOT / "raws", help="Folder of input videos")
    p.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "dataset",
        help="Dataset root (contains data.yaml, will get images/ and labels/)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Save every Nth frame (1 = all frames, 2 = half, etc.)",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of clips assigned to validation (by video, not by frame)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for val split")
    p.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Optional cap per video (after stride)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sys.exit(
        extract_frames(
            args.raws.expanduser().resolve(),
            args.dataset.expanduser().resolve(),
            stride=args.stride,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_frames_per_video=args.max_frames_per_video,
        )
    )


if __name__ == "__main__":
    main()
