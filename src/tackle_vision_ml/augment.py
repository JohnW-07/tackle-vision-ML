"""
Video augmentation pipeline for the tackle dataset.

Generates deterministic variations per raw video (same transform on every frame):

  A – Mirror        : horizontal flip (opposite-direction tackle)
  B – Broadcast     : Gaussian noise + motion blur (low-quality / high-speed footage)
  C – Lighting      : brightness & contrast jitter (stadium / weather variation)
  D – Rotation      : small in-plane rotation about frame center (fixed angle per clip)
  E – Shift         : 2D translation in pixels (fixed per clip)
  F – Rotate+shift  : rotation then translation (camera-style jitter)

Geometric variants (D–F) use one OpenCV affine matrix per clip, sampled once from
the file-derived RNG so temporal consistency is preserved.

Usage (CLI):
    python -m tackle_vision_ml.augment                     # ./raws -> ./output/
    python -m tackle_vision_ml.augment --raws my_raws --output my_aug
    python -m tackle_vision_ml.augment --max-rotation-deg 15 --max-shift-frac 0.1
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


def _affine_rotation_matrix(
    rng: random.Random, width: int, height: int, max_deg: float
) -> np.ndarray:
    """2×3 OpenCV affine matrix: rotation about image center (degrees)."""
    angle = rng.uniform(-float(max_deg), float(max_deg))
    cx, cy = width * 0.5, height * 0.5
    return cv2.getRotationMatrix2D((cx, cy), angle, 1.0).astype(np.float32)


def _affine_shift_matrix(
    rng: random.Random, width: int, height: int, max_frac: float
) -> np.ndarray:
    """2×3 pure translation; shifts are fractions of width/height."""
    tx = rng.uniform(-float(max_frac), float(max_frac)) * width
    ty = rng.uniform(-float(max_frac), float(max_frac)) * height
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)


def _affine_rotate_then_shift(
    rng: random.Random,
    width: int,
    height: int,
    max_deg: float,
    max_frac: float,
) -> np.ndarray:
    """Rotation about center, then add translation in output pixel space."""
    m = _affine_rotation_matrix(rng, width, height, max_deg)
    tx = rng.uniform(-float(max_frac), float(max_frac)) * width
    ty = rng.uniform(-float(max_frac), float(max_frac)) * height
    m[0, 2] += tx
    m[1, 2] += ty
    return m.astype(np.float32)


def _warp_affine_rgb(frame_rgb: np.ndarray, m: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.warpAffine(
        frame_rgb,
        m,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


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

def augment_video(
    src: Path,
    dst_dir: Path,
    *,
    max_rotation_deg: float = 12.0,
    max_shift_frac: float = 0.08,
) -> None:
    """
    Read *src*, produce augmented variants, write to *dst_dir*.

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

    variations_alb: list[tuple[str, A.Compose]] = [
        ("A_mirror", _make_variation_a()),
        ("B_broadcast", _make_variation_b(rng)),
        ("C_lighting", _make_variation_c(rng)),
    ]
    variations_affine: list[tuple[str, np.ndarray]] = [
        ("D_rotate", _affine_rotation_matrix(rng, width, height, max_rotation_deg)),
        ("E_shift", _affine_shift_matrix(rng, width, height, max_shift_frac)),
        (
            "F_rotate_shift",
            _affine_rotate_then_shift(rng, width, height, max_rotation_deg, max_shift_frac),
        ),
    ]

    # ------------------------------------------------------------------
    # Open one writer per variation
    # ------------------------------------------------------------------
    writers: list[cv2.VideoWriter] = []
    for tag, _ in variations_alb:
        out_path = dst_dir / f"{stem}_{tag}.mp4"
        writer = _open_writer(out_path, fps, width, height)
        writers.append(writer)
        print(f"  -> {out_path.name}")
    for tag, _ in variations_affine:
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

        w_idx = 0
        for _, transform in variations_alb:
            augmented = transform(image=frame_rgb)["image"]
            writers[w_idx].write(cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            w_idx += 1
        for _, m in variations_affine:
            warped = _warp_affine_rgb(frame_rgb, m, width, height)
            writers[w_idx].write(cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
            w_idx += 1

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
        "--input",
        "--raws",
        dest="input_dir",
        default="raws",
        help="Directory containing raw .mp4 / .mov clips (default: ./raws)",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default="output",
        help="Destination directory for augmented clips (default: ./output)",
    )
    parser.add_argument(
        "--max-rotation-deg",
        type=float,
        default=12.0,
        help="Max |rotation| in degrees for D and F (symmetric uniform draw per clip)",
    )
    parser.add_argument(
        "--max-shift-frac",
        type=float,
        default=0.08,
        help="Max |shift| as fraction of frame width/height for E and F (per axis)",
    )
    args = parser.parse_args()

    src_dir = Path(args.input_dir).expanduser().resolve()
    dst_dir = Path(args.output_dir).expanduser().resolve()

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
        augment_video(
            vid,
            dst_dir,
            max_rotation_deg=args.max_rotation_deg,
            max_shift_frac=args.max_shift_frac,
        )
        print()

    n_variants = 6
    print(f"Augmentation complete. {len(video_files) * n_variants} files written to '{dst_dir}'.")


if __name__ == "__main__":
    main()
