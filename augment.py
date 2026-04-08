"""
Augmentation-focused starting point for tackle video experiments.

This module is intentionally self-contained so the rest of the repo can stay
small while you iterate on augmentation ideas.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODEC_CANDIDATES = ["avc1", "mp4v"]
VIDEO_SUFFIXES = {".mp4", ".mov"}


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    for codec in CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for {path} with codecs {CODEC_CANDIDATES}")


def assign_label(video_path: Path) -> int:
    """
    Placeholder for any filename- or metadata-based labeling scheme.

    Replace this with your own dataset logic when you rebuild the project.
    """
    _ = video_path
    return 0


def apply_color_temperature(image: np.ndarray, red_gain: float, blue_gain: float) -> np.ndarray:
    adjusted = image.astype(np.float32).copy()
    adjusted[..., 0] *= red_gain
    adjusted[..., 2] *= blue_gain
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def make_exposure_variation(rng: random.Random) -> list[A.BasicTransform]:
    brightness_scale = rng.uniform(0.8, 1.2)
    contrast_scale = rng.uniform(0.85, 1.15)
    gamma_scale = rng.uniform(0.8, 1.2)
    saturation_scale = rng.uniform(0.85, 1.15)
    hue = rng.uniform(-8.0, 8.0)
    temperature = rng.uniform(-0.08, 0.08)
    red_gain = 1.0 + max(0.0, temperature)
    blue_gain = 1.0 + max(0.0, -temperature)

    return [
        A.RandomBrightnessContrast(
            brightness_limit=(brightness_scale - 1.0, brightness_scale - 1.0),
            contrast_limit=(contrast_scale - 1.0, contrast_scale - 1.0),
            p=1.0,
        ),
        A.RandomGamma(gamma_limit=(gamma_scale * 100.0, gamma_scale * 100.0), p=1.0),
        A.HueSaturationValue(
            hue_shift_limit=(hue, hue),
            sat_shift_limit=((saturation_scale - 1.0) * 100.0, (saturation_scale - 1.0) * 100.0),
            val_shift_limit=(0.0, 0.0),
            p=1.0,
        ),
        A.Lambda(image=lambda img, **kwargs: apply_color_temperature(img, red_gain, blue_gain), p=1.0),
    ]


def make_blocking_variation(rng: random.Random, width: int, height: int) -> list[A.BasicTransform]:
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


def make_translation_variation(rng: random.Random) -> list[A.BasicTransform]:
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


def build_variations(
    rng: random.Random,
    *,
    width: int,
    height: int,
    num_variations: int,
    use_exposure: bool,
    use_blocking: bool,
    use_translation: bool,
) -> list[tuple[str, A.Compose]]:
    selected = []
    if use_exposure:
        selected.append("exposure")
    if use_blocking:
        selected.append("blocking")
    if use_translation:
        selected.append("translation")
    if not selected:
        return []

    mode_tag = "all" if len(selected) > 1 else selected[0]
    variations: list[tuple[str, A.Compose]] = []
    for index in range(1, num_variations + 1):
        transforms: list[A.BasicTransform] = []
        if use_exposure:
            transforms.extend(make_exposure_variation(rng))
        if use_blocking:
            transforms.extend(make_blocking_variation(rng, width, height))
        if use_translation:
            transforms.extend(make_translation_variation(rng))
        variations.append((f"{mode_tag}_{index:02d}", A.Compose(transforms)))
    return variations


def augment_video(
    src: Path,
    dst_dir: Path,
    *,
    num_variations: int,
    use_exposure: bool,
    use_blocking: bool,
    use_translation: bool,
) -> int:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {src}", file=sys.stderr)
        return 0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    label = assign_label(src)
    rng = random.Random(hash(src.name) & 0xFFFF_FFFF)

    variations = build_variations(
        rng,
        width=width,
        height=height,
        num_variations=num_variations,
        use_exposure=use_exposure,
        use_blocking=use_blocking,
        use_translation=use_translation,
    )
    if not variations:
        print(f"  [SKIP] No augmentation sections selected for: {src.name}")
        cap.release()
        return 0

    writers: list[cv2.VideoWriter] = []
    for tag, _ in variations:
        out_path = dst_dir / f"{src.stem}_{label}_{tag}.mp4"
        writers.append(_open_writer(out_path, fps, width, height))
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate temporally consistent video augmentations.")
    parser.add_argument("num_variations", nargs="?", type=int, default=3)
    parser.add_argument("--input", default=None, help="Input directory containing .mp4 / .mov clips.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "augmented"))
    parser.add_argument("-exposure", dest="exposure", action="store_true")
    parser.add_argument("-blocking", dest="blocking", action="store_true")
    parser.add_argument("-translation", dest="translation", action="store_true")
    parser.add_argument("-all", dest="all_sections", action="store_true")
    parser.add_argument("-upsampled", dest="use_upsampled", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_variations < 1:
        sys.exit("num_variations must be at least 1")

    selected_any = args.exposure or args.blocking or args.translation or args.all_sections
    use_exposure = args.exposure or args.all_sections or not selected_any
    use_blocking = args.blocking or args.all_sections or not selected_any
    use_translation = args.translation or args.all_sections or not selected_any

    default_input = PROJECT_ROOT / ("upsampled" if args.use_upsampled else "raws")
    src_dir = Path(args.input) if args.input is not None else default_input
    dst_dir = Path(args.output)

    if not src_dir.exists():
        sys.exit(f"Input directory not found: {src_dir}")

    video_files = sorted(path for path in src_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)
    if not video_files:
        sys.exit(f"No .mp4 or .mov files found in {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(video_files)} video(s) in '{src_dir}'")
    print(f"Output directory: '{dst_dir}'")
    print(f"Variants per source video: {args.num_variations}\n")

    total_outputs = 0
    for index, src in enumerate(video_files, start=1):
        print(f"[{index}/{len(video_files)}] Processing: {src.name}")
        total_outputs += augment_video(
            src,
            dst_dir,
            num_variations=args.num_variations,
            use_exposure=use_exposure,
            use_blocking=use_blocking,
            use_translation=use_translation,
        )
        print()

    print(f"Augmentation complete. {total_outputs} files written to '{dst_dir}'.")


if __name__ == "__main__":
    main()
