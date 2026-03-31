"""
Video augmentation pipeline for the tackle dataset.

Generates three deterministic variations per raw video:
  A  – Mirror       : horizontal flip (opposite-direction tackle)
  B  – Broadcast    : Gaussian noise + motion blur (low-quality / high-speed footage)
  C  – Lighting     : brightness & contrast jitter (stadium / weather variation)

  to do:
  - frame skipping
  - bit perturbation 
  - rotations + blocks coming out 

  smoke test with our videos first
  

Temporal consistency is guaranteed by freezing all random parameters **once per
video** and re-applying the identical transform to every frame.

Usage (CLI):
    python -m tackle_vision_ml.augment                     # raws/ -> augmented/
    python -m tackle_vision_ml.augment --input my_raws --output my_aug
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Codec preference list – tried in order until one works on the current OS
# ---------------------------------------------------------------------------
_CODEC_CANDIDATES = ["avc1", "mp4v"]


def _open_writer(
    path: Path, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    """Try preferred codecs in order; raise RuntimeError if none open."""
    for codec in _CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"Could not open VideoWriter for {path} with codecs {_CODEC_CANDIDATES}"
    )


# ---------------------------------------------------------------------------
# Per-variation transform factories
# Each factory returns (transform, params_dict | None).
# For pixel-only transforms (B, C) albumentations handles randomness internally
# once we freeze the seed; for geometric transforms (A) we build a fixed pipeline.
# ---------------------------------------------------------------------------

def _make_variation_a() -> A.Compose:
    """Mirror: deterministic horizontal flip – no randomness needed."""
    return A.Compose([A.HorizontalFlip(p=1.0)])


def _make_variation_b(rng: random.Random) -> A.Compose:
    """
    Broadcast noise: Gaussian noise + motion blur.

    Parameters are sampled once per video (using the caller-supplied RNG) so
    every frame in the clip receives the same degradation level.
    """
    # Gaussian noise: variance uniformly sampled in [5, 25]
    var = rng.uniform(5.0, 25.0)
    # Motion blur: kernel size odd int in [3, 9]
    ksize = rng.choice([3, 5, 7, 9])

    return A.Compose(
        [
            A.GaussNoise(
                noise_scale_factor=1.0,   # use explicit std_range instead
                std_range=(var ** 0.5 / 255.0, var ** 0.5 / 255.0),
                p=1.0,
            ),
            A.MotionBlur(blur_limit=(ksize, ksize), p=1.0),
        ]
    )


def _make_variation_c(rng: random.Random) -> A.Compose:
    """
    Lighting: fixed brightness + contrast shift sampled once per video.

    brightness_limit and contrast_limit are offsets in [-limit, +limit].
    We pick a single value in those ranges and use [val, val] so the
    transform is deterministic across frames.
    """
    bright = rng.uniform(-0.3, 0.3)
    contrast = rng.uniform(-0.3, 0.3)

    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(bright, bright),
                contrast_limit=(contrast, contrast),
                p=1.0,
            )
        ]
    )


# ---------------------------------------------------------------------------
# Core per-file augmentation
# ---------------------------------------------------------------------------

def augment_video(src: Path, dst_dir: Path) -> None:
    """
    Read *src*, produce three augmented variants, write to *dst_dir*.

    Frame rate and dimensions are preserved exactly.
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {src}", file=sys.stderr)
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1e-6:
        fps = 30.0

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = src.stem  # original filename without extension

    # ------------------------------------------------------------------
    # Build transforms – all random parameters frozen before reading frames
    # ------------------------------------------------------------------
    # Use a deterministic seed derived from the filename so re-runs are
    # reproducible while different files get different augmentation params.
    file_seed = hash(src.name) & 0xFFFF_FFFF
    rng = random.Random(file_seed)

    variations: list[tuple[str, A.Compose]] = [
        ("A_mirror",     _make_variation_a()),
        ("B_broadcast",  _make_variation_b(rng)),
        ("C_lighting",   _make_variation_c(rng)),
    ]

    # ------------------------------------------------------------------
    # Open one writer per variation
    # ------------------------------------------------------------------
    writers: list[cv2.VideoWriter] = []
    for tag, _ in variations:
        out_path = dst_dir / f"{stem}_{tag}.mp4"
        writer = _open_writer(out_path, fps, width, height)
        writers.append(writer)
        print(f"  -> {out_path.name}")

    # ------------------------------------------------------------------
    # Stream frames: apply each transform, write to corresponding file
    # ------------------------------------------------------------------
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # Albumentations works in RGB; convert once
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        for writer, (_, transform) in zip(writers, variations):
            augmented = transform(image=frame_rgb)["image"]
            writer.write(cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        frame_idx += 1
        if total > 0 and frame_idx % max(1, total // 10) == 0:
            pct = 100 * frame_idx / total
            print(f"     {frame_idx}/{total} frames ({pct:.0f}%)")

    cap.release()
    for w in writers:
        w.release()

    print(f"  [DONE] {frame_idx} frames written for {src.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate augmented video variants for the tackle dataset."
    )
    parser.add_argument(
        "--input", default="raws",
        help="Directory containing raw .mp4 / .mov clips (default: raws/)"
    )
    parser.add_argument(
        "--output", default="augmented",
        help="Destination directory for augmented clips (default: augmented/)"
    )
    args = parser.parse_args()

    src_dir = Path(args.input)
    dst_dir = Path(args.output)

    if not src_dir.exists():
        sys.exit(f"Input directory not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        p for p in src_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".mov"}
    )

    if not video_files:
        sys.exit(f"No .mp4 or .mov files found in {src_dir}")

    print(f"Found {len(video_files)} video(s) in '{src_dir}'")
    print(f"Output directory: '{dst_dir}'\n")

    for i, vid in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Processing: {vid.name}")
        augment_video(vid, dst_dir)
        print()

    print(f"Augmentation complete. {len(video_files) * 3} files written to '{dst_dir}'.")


if __name__ == "__main__":
    main()
