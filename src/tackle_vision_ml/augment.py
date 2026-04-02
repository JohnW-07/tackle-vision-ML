"""
Video augmentation pipeline for the tackle dataset.

The file is organized into three collaboration-friendly sections:
  1. Exposure       : brightness, contrast, gamma, saturation, white balance
  2. Random blocking: deterministic block dropout across frames
  3. Translation    : mirror, slight rotation, and slight translation

Temporal consistency is guaranteed by freezing all random parameters once per
video and re-applying the identical transform to every frame.

Usage (CLI):
    python3 augment.py
    python3 augment.py 3 -exposure
    python3 augment.py 10 -all
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


# Repository root: .../tackle-vision-ML
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Codec preference list - tried in order until one works on the current OS
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
# Section 1: Exposure
# ---------------------------------------------------------------------------

def _apply_color_temperature(
    image: np.ndarray, red_gain: float, blue_gain: float
) -> np.ndarray:
    """Approximate white-balance / color-temperature shifts with channel gains."""
    adjusted = image.astype(np.float32).copy()
    adjusted[..., 0] *= red_gain
    adjusted[..., 2] *= blue_gain
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _make_exposure_variation(rng: random.Random) -> list[A.BasicTransform]:
    """
    Exposure and color grading with fixed per-video parameters.

    We freeze the random values once so the entire clip keeps a consistent
    lighting and color look.
    """
    brightness_scale = rng.uniform(0.8, 1.2)
    brightness = brightness_scale - 1.0
    contrast_scale = rng.uniform(0.85, 1.15)
    contrast = contrast_scale - 1.0
    gamma_scale = rng.uniform(0.8, 1.2)
    gamma = gamma_scale * 100.0
    saturation_scale = rng.uniform(0.85, 1.15)
    saturation = (saturation_scale - 1.0) * 100.0
    hue = rng.uniform(-8.0, 8.0)
    temperature = rng.uniform(-0.08, 0.08)
    red_gain = 1.0 + max(0.0, temperature)
    blue_gain = 1.0 + max(0.0, -temperature)

    return [
        A.RandomBrightnessContrast(
            brightness_limit=(brightness, brightness),
            contrast_limit=(contrast, contrast),
            p=1.0,
        ),
        A.RandomGamma(gamma_limit=(gamma, gamma), p=1.0),
        A.HueSaturationValue(
            hue_shift_limit=(hue, hue),
            sat_shift_limit=(saturation, saturation),
            val_shift_limit=(0.0, 0.0),
            p=1.0,
        ),
        A.Lambda(
            image=lambda img, **kwargs: _apply_color_temperature(
                img, red_gain, blue_gain
            ),
            p=1.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Section 2: Random blocking
# ---------------------------------------------------------------------------

def _make_random_blocking_variation(
    rng: random.Random, width: int, height: int
) -> list[A.BasicTransform]:
    """
    Remove a few deterministic rectangular blocks from every frame in the clip.

    Blocks are sampled once per video so occlusions remain stable across time.
    """
    max_hole_height = max(12, height // 7)
    max_hole_width = max(12, width // 7)
    num_holes = rng.randint(1, 4)
    fill_value = rng.randint(0, 30)

    return [
        A.CoarseDropout(
            num_holes_range=(num_holes, num_holes),
            hole_height_range=(0.08, min(0.22, max_hole_height / height)),
            hole_width_range=(0.08, min(0.22, max_hole_width / width)),
            fill=fill_value,
            p=1.0,
        )
    ]


# ---------------------------------------------------------------------------
# Section 3: Translation
# ---------------------------------------------------------------------------

def _make_translation_variation(rng: random.Random) -> list[A.BasicTransform]:
    """
    Geometric perturbation: mirror plus a small rotation and XY shift.
    """
    rotate = rng.uniform(-5.0, 5.0)
    shift_x = rng.uniform(-0.05, 0.05)
    shift_y = rng.uniform(-0.05, 0.05)

    return [
        A.HorizontalFlip(p=1.0),
        A.Affine(
            scale=1.0,
            translate_percent={"x": (shift_x, shift_x), "y": (shift_y, shift_y)},
            rotate=(rotate, rotate),
            shear=0.0,
            p=1.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Core per-file augmentation
# ---------------------------------------------------------------------------

def _build_variations(
    rng: random.Random,
    width: int,
    height: int,
    num_variations: int,
    run_exposure: bool,
    run_blocking: bool,
    run_translation: bool,
) -> list[tuple[str, A.Compose]]:
    variations: list[tuple[str, A.Compose]] = []
    selected_sections = []

    if run_exposure:
        selected_sections.append("exposure")
    if run_blocking:
        selected_sections.append("blocking")
    if run_translation:
        selected_sections.append("translation")

    if not selected_sections:
        return variations

    mode_tag = "all" if len(selected_sections) > 1 else selected_sections[0]

    for idx in range(1, num_variations + 1):
        transforms: list[A.BasicTransform] = []

        if run_exposure:
            transforms.extend(_make_exposure_variation(rng))
        if run_blocking:
            transforms.extend(_make_random_blocking_variation(rng, width, height))
        if run_translation:
            transforms.extend(_make_translation_variation(rng))

        variations.append((f"{mode_tag}_{idx:02d}", A.Compose(transforms)))

    return variations


def augment_video(
    src: Path,
    dst_dir: Path,
    num_variations: int,
    run_exposure: bool,
    run_blocking: bool,
    run_translation: bool,
) -> int:
    """
    Read *src*, produce the requested augmented variants, write to *dst_dir*.

    Frame rate and dimensions are preserved exactly.
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {src}", file=sys.stderr)
        return 0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = src.stem
    file_seed = hash(src.name) & 0xFFFF_FFFF
    rng = random.Random(file_seed)

    variations = _build_variations(
        rng,
        width,
        height,
        num_variations=num_variations,
        run_exposure=run_exposure,
        run_blocking=run_blocking,
        run_translation=run_translation,
    )

    if not variations:
        print(f"  [SKIP] No augmentation sections selected for: {src.name}")
        cap.release()
        return 0

    writers: list[cv2.VideoWriter] = []
    for tag, _ in variations:
        out_path = dst_dir / f"{stem}_{tag}.mp4"
        writer = _open_writer(out_path, fps, width, height)
        writers.append(writer)
        print(f"  -> {out_path.name}")

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        for writer, (_, transform) in zip(writers, variations):
            augmented = transform(image=frame_rgb)["image"]
            writer.write(cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

        frame_idx += 1
        if total > 0 and frame_idx % max(1, total // 10) == 0:
            pct = 100 * frame_idx / total
            print(f"     {frame_idx}/{total} frames ({pct:.0f}%)")

    cap.release()
    for writer in writers:
        writer.release()

    print(f"  [DONE] {frame_idx} frames written for {src.name}")
    return len(variations)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate augmented video variants for the tackle dataset."
    )
    parser.add_argument(
        "num_variations",
        nargs="?",
        type=int,
        default=3,
        help="Number of augmented outputs to generate per raw video (default: 3).",
    )
    parser.add_argument(
        "--input",
        default=str(_PROJECT_ROOT / "raws"),
        help="Directory containing raw .mp4 / .mov clips (default: repo_root/raws/)",
    )
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "augmented"),
        help="Destination directory for augmented clips (default: repo_root/augmented/)",
    )
    parser.add_argument(
        "-exposure",
        dest="exposure",
        action="store_true",
        help="Generate exposure-only variants.",
    )
    parser.add_argument(
        "-blocking",
        dest="blocking",
        action="store_true",
        help="Generate random-blocking-only variants unless combined.",
    )
    parser.add_argument(
        "-translation",
        dest="translation",
        action="store_true",
        help="Generate translation-only variants unless combined.",
    )
    parser.add_argument(
        "-all",
        dest="all_sections",
        action="store_true",
        help="Apply exposure, blocking, and translation together.",
    )
    args = parser.parse_args()

    if args.num_variations < 1:
        sys.exit("num_variations must be at least 1")

    selected_any = (
        args.exposure or args.blocking or args.translation or args.all_sections
    )
    run_exposure = args.exposure or args.all_sections or not selected_any
    run_blocking = args.blocking or args.all_sections or not selected_any
    run_translation = args.translation or args.all_sections or not selected_any

    src_dir = Path(args.input)
    dst_dir = Path(args.output)

    if not src_dir.exists():
        sys.exit(f"Input directory not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        p for p in src_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov"}
    )

    if not video_files:
        sys.exit(f"No .mp4 or .mov files found in {src_dir}")

    print(f"Found {len(video_files)} video(s) in '{src_dir}'")
    print(f"Output directory: '{dst_dir}'")
    print(f"Variants per raw video: {args.num_variations}")

    selected_sections = []
    if run_exposure:
        selected_sections.append("exposure")
    if run_blocking:
        selected_sections.append("blocking")
    if run_translation:
        selected_sections.append("translation")
    print(f"Selected sections: {', '.join(selected_sections)}\n")

    total_outputs = 0
    for i, vid in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Processing: {vid.name}")
        total_outputs += augment_video(
            vid,
            dst_dir,
            num_variations=args.num_variations,
            run_exposure=run_exposure,
            run_blocking=run_blocking,
            run_translation=run_translation,
        )
        print()

    print(f"Augmentation complete. {total_outputs} files written to '{dst_dir}'.")


if __name__ == "__main__":
    main()
