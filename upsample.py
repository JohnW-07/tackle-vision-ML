from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REAL_ESRGAN_PYTHON = PROJECT_ROOT / ".venv-realesrgan" / "bin" / "python"
RAW_DIR = PROJECT_ROOT / "raws"
UPSAMPLED_DIR = PROJECT_ROOT / "upsampled"
WEIGHTS_PATH = PROJECT_ROOT / "models" / "realesrgan" / "RealESRGAN_x4plus.pth"
VIDEO_SUFFIXES = {".mp4", ".mov"}
CODEC_CANDIDATES = ["avc1", "mp4v"]


def _ensure_realesrgan_python() -> None:
    expected = REAL_ESRGAN_PYTHON.resolve()
    current = Path(sys.executable).resolve()

    if current == expected:
        return

    if not REAL_ESRGAN_PYTHON.exists():
        sys.exit(
            "Real-ESRGAN environment not found. Expected "
            f"{REAL_ESRGAN_PYTHON}"
        )

    os.execv(str(REAL_ESRGAN_PYTHON), [str(REAL_ESRGAN_PYTHON), __file__, *sys.argv[1:]])


_ensure_realesrgan_python()

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    for codec in CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for {path}")


def build_upsampler(
    weights_path: Path,
    tile: int,
    tile_pad: int,
    outscale: float,
) -> RealESRGANer:
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Model weights: {weights_path}")
    print(f"Requested outscale: {outscale}")

    return RealESRGANer(
        scale=4,
        model_path=str(weights_path),
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=0,
        half=False,
        device=device,
    )


def upsample_video(
    src: Path,
    dst: Path,
    upsampler: RealESRGANer,
    outscale: float,
) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[SKIP] Cannot open: {src}", file=sys.stderr)
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_width = max(1, int(round(width * outscale)))
    out_height = max(1, int(round(height * outscale)))

    writer = open_writer(dst, fps, out_width, out_height)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        output, _ = upsampler.enhance(frame, outscale=outscale)
        writer.write(output)
        frame_idx += 1

        if total > 0 and frame_idx % max(1, total // 10) == 0:
            pct = 100 * frame_idx / total
            print(f"   {frame_idx}/{total} frames ({pct:.0f}%)")

    cap.release()
    writer.release()
    print(f"[DONE] {dst.name} ({frame_idx} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upsample all videos in raws/ with Real-ESRGAN."
    )
    parser.add_argument(
        "--input",
        default=str(RAW_DIR),
        help="Directory containing source .mp4 / .mov clips.",
    )
    parser.add_argument(
        "--output",
        default=str(UPSAMPLED_DIR),
        help="Directory where upsampled videos will be written.",
    )
    parser.add_argument(
        "--weights",
        default=str(WEIGHTS_PATH),
        help="Path to the RealESRGAN_x4plus model weights.",
    )
    parser.add_argument(
        "--outscale",
        type=float,
        default=4.0,
        help="Output scale multiplier passed to Real-ESRGAN.",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="Tile size for memory-constrained runs. Use 0 to disable tiling.",
    )
    parser.add_argument(
        "--tile-pad",
        type=int,
        default=10,
        help="Tile padding to reduce seam artifacts when tiling is enabled.",
    )
    args = parser.parse_args()

    src_dir = Path(args.input)
    dst_dir = Path(args.output)
    weights_path = Path(args.weights)

    if not src_dir.exists():
        sys.exit(f"Input directory not found: {src_dir}")
    if not weights_path.exists():
        sys.exit(f"Model weights not found: {weights_path}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        path for path in src_dir.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not video_files:
        sys.exit(f"No .mp4 or .mov files found in {src_dir}")

    upsampler = build_upsampler(
        weights_path=weights_path,
        tile=args.tile,
        tile_pad=args.tile_pad,
        outscale=args.outscale,
    )

    print(f"Found {len(video_files)} video(s) in '{src_dir}'")
    print(f"Writing Real-ESRGAN outputs to '{dst_dir}'\n")

    for index, src in enumerate(video_files, start=1):
        dst = dst_dir / f"{src.stem}_upsampled.mp4"
        print(f"[{index}/{len(video_files)}] Upsampling: {src.name}")
        print(f" -> {dst.name}")
        upsample_video(
            src=src,
            dst=dst,
            upsampler=upsampler,
            outscale=args.outscale,
        )
        print()


if __name__ == "__main__":
    main()
