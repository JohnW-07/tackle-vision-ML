"""
Upsample every video in `raws/` with SeedVR2.

By default this uses numz/ComfyUI-SeedVR2_VideoUpscaler's standalone CLI,
because that wrapper supports Apple Silicon MPS as well as CUDA. The original
ByteDance SeedVR torchrun path is still available with `--backend official`.

Outputs are written to `upsampled/` as:

  <video id>_up.<original suffix>

Example:

  raws/game_clip.m4v -> upsampled/game_clip_up.m4v

Note: the ComfyUI wrapper writes MP4 video streams. This launcher keeps the
requested output filename suffix for compatibility with the rest of this repo.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

COMFY_REPO_URL = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
DEFAULT_COMFY_REPO = Path(
    os.environ.get("SEEDVR2_UPSCALER_REPO", PROJECT_ROOT / "seedvr2_videoupscaler")
)
DEFAULT_COMFY_MODEL_DIR = PROJECT_ROOT / "models" / "SEEDVR2"
DEFAULT_COMFY_DIT_MODEL = "seedvr2_ema_7b-Q4_K_M.gguf"

OFFICIAL_REPO_URL = "https://github.com/ByteDance-Seed/SeedVR.git"
DEFAULT_OFFICIAL_REPO = Path(os.environ.get("SEEDVR_REPO", PROJECT_ROOT / "SeedVR"))
OFFICIAL_7B_REPO_ID = "ByteDance-Seed/SeedVR2-7B"
OFFICIAL_7B_CHECKPOINT = "seedvr2_ema_7b.pth"

VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
    ".wmv",
}

COMFY_REQUIRED_MODULES = {
    "cv2": "opencv-python",
    "diffusers": "diffusers",
    "einops": "einops",
    "gguf": "gguf",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "peft": "peft",
    "psutil": "psutil",
    "rotary_embedding_torch": "rotary-embedding-torch",
    "safetensors": "safetensors",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "tqdm": "tqdm",
}

OFFICIAL_REQUIRED_MODULES = {
    "diffusers": "diffusers",
    "einops": "einops",
    "mediapy": "mediapy",
    "omegaconf": "omegaconf",
    "rotary_embedding_torch": "rotary-embedding-torch",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
}


def find_videos(raws_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in raws_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_SUFFIXES
        and not p.stem.endswith("_up")
    )


def output_path_for(video_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{video_path.stem}_up{video_path.suffix}"


def run_command(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_repo(repo_url: str, target: Path, *, clone: bool) -> None:
    if target.exists():
        return
    if not clone:
        raise FileNotFoundError(
            f"Repository not found at {target}. Clone it there or rerun without --no-clone."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "clone", repo_url, str(target)])


def missing_modules(required: dict[str, str]) -> list[str]:
    return [
        package
        for module, package in required.items()
        if importlib.util.find_spec(module) is None
    ]


def available_torch_backend() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return None

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return None


def ensure_comfy_repo(comfy_repo: Path, *, clone: bool) -> None:
    clone_repo(COMFY_REPO_URL, comfy_repo, clone=clone)
    cli_path = comfy_repo / "inference_cli.py"
    if not cli_path.is_file():
        raise FileNotFoundError(f"Comfy SeedVR2 clone is missing {cli_path}")


def check_comfy_runtime(comfy_repo: Path) -> None:
    errors = []
    missing = missing_modules(COMFY_REQUIRED_MODULES)
    if missing:
        errors.append(
            "The active Python environment is missing Comfy SeedVR2 dependencies: "
            f"{', '.join(missing)}.\n"
            f"Install them with: python -m pip install -r {comfy_repo / 'requirements.txt'}"
        )

    backend = available_torch_backend()
    if backend is None:
        errors.append(
            "PyTorch does not report a usable CUDA or MPS backend. "
            "The Comfy SeedVR2 CLI needs CUDA on NVIDIA/Linux or MPS on Apple Silicon."
        )

    if errors:
        raise RuntimeError("Comfy SeedVR2 runtime is not ready:\n\n" + "\n\n".join(errors))


def ensure_official_repo(seedvr_repo: Path, *, clone: bool) -> None:
    clone_repo(OFFICIAL_REPO_URL, seedvr_repo, clone=clone)
    inference_script = seedvr_repo / "projects" / "inference_seedvr2_7b.py"
    if not inference_script.is_file():
        raise FileNotFoundError(f"SeedVR clone is missing {inference_script}")


def ensure_official_checkpoint(seedvr_repo: Path, *, download: bool) -> None:
    ckpt_dir = seedvr_repo / "ckpts"
    ckpt_path = ckpt_dir / OFFICIAL_7B_CHECKPOINT
    if ckpt_path.is_file():
        return

    if not download:
        raise FileNotFoundError(
            f"SeedVR2-7B checkpoint not found at {ckpt_path}. "
            "Download it there or rerun without --no-download."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download the official SeedVR2-7B checkpoint. "
            "Install it with: python -m pip install huggingface_hub"
        ) from exc

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {OFFICIAL_7B_REPO_ID}/{OFFICIAL_7B_CHECKPOINT} "
        f"to {ckpt_dir} (about 33 GB)."
    )
    snapshot_download(
        repo_id=OFFICIAL_7B_REPO_ID,
        local_dir=ckpt_dir,
        allow_patterns=[OFFICIAL_7B_CHECKPOINT],
        resume_download=True,
    )

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint download did not create {ckpt_path}")


def check_official_runtime(seedvr_repo: Path, *, skip_cuda_check: bool) -> None:
    errors = []
    missing = missing_modules(OFFICIAL_REQUIRED_MODULES)
    if missing:
        errors.append(
            "The active Python environment is missing official SeedVR dependencies: "
            f"{', '.join(missing)}.\n"
            f"Install them with: python -m pip install -r {seedvr_repo / 'requirements.txt'}\n"
            "The official path also needs CUDA-only packages such as flash-attn and apex."
        )

    if not skip_cuda_check:
        backend = available_torch_backend()
        if backend != "cuda":
            mps_hint = " MPS is available, but the official script does not support it." if backend == "mps" else ""
            errors.append(
                "CUDA is not available in this environment."
                f"{mps_hint}\n"
                "Use the default Comfy backend for macOS/MPS, or run the official backend "
                "from a Linux CUDA environment."
            )

    if errors:
        raise RuntimeError("Official SeedVR2-7B runtime is not ready:\n\n" + "\n\n".join(errors))


def link_video(video_path: Path, temp_input_dir: Path) -> None:
    linked_path = temp_input_dir / video_path.name
    try:
        os.symlink(video_path, linked_path)
    except FileExistsError:
        pass


def move_comfy_outputs(temp_output_dir: Path, output_dir: Path, videos: list[Path]) -> None:
    for video_path in videos:
        comfy_output = temp_output_dir / f"{video_path.stem}.mp4"
        final_output = output_path_for(video_path, output_dir)
        if not comfy_output.is_file():
            print(f"[warn] Comfy SeedVR2 did not produce {comfy_output}", file=sys.stderr)
            continue
        shutil.move(str(comfy_output), str(final_output))
        print(f"[done] {video_path.name} -> {final_output}")


def move_official_outputs(temp_output_dir: Path, output_dir: Path, videos: list[Path]) -> None:
    for video_path in videos:
        seedvr_output = temp_output_dir / video_path.name
        final_output = output_path_for(video_path, output_dir)
        if not seedvr_output.is_file():
            print(f"[warn] SeedVR did not produce {seedvr_output}", file=sys.stderr)
            continue
        shutil.move(str(seedvr_output), str(final_output))
        print(f"[done] {video_path.name} -> {final_output}")


def print_pending(videos: list[Path], output_dir: Path, *, backend: str) -> None:
    print(f"Pending SeedVR2 jobs ({backend} backend):")
    for video_path in videos:
        print(f"  {video_path.name} -> {output_path_for(video_path, output_dir).name}")


def run_comfy_backend(args: argparse.Namespace, pending: list[Path], output_dir: Path) -> None:
    comfy_repo = args.comfy_repo.expanduser().resolve()
    model_dir = args.comfy_model_dir.expanduser().resolve()

    ensure_comfy_repo(comfy_repo, clone=not args.no_clone)
    check_comfy_runtime(comfy_repo)

    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="seedvr2_comfy_inputs_") as temp_input:
        with tempfile.TemporaryDirectory(prefix="seedvr2_comfy_outputs_") as temp_output:
            temp_input_dir = Path(temp_input)
            temp_output_dir = Path(temp_output)

            for video_path in pending:
                link_video(video_path, temp_input_dir)

            cmd = [
                args.python,
                "inference_cli.py",
                str(temp_input_dir),
                "--output",
                str(temp_output_dir),
                "--output_format",
                "mp4",
                "--model_dir",
                str(model_dir),
                "--dit_model",
                args.dit_model,
                "--resolution",
                str(args.resolution),
                "--max_resolution",
                str(args.max_resolution),
                "--batch_size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                "--skip_first_frames",
                str(args.skip_first_frames),
                "--load_cap",
                str(args.load_cap),
                "--chunk_size",
                str(args.chunk_size),
                "--temporal_overlap",
                str(args.temporal_overlap),
                "--prepend_frames",
                str(args.prepend_frames),
                "--color_correction",
                args.color_correction,
                "--input_noise_scale",
                str(args.input_noise_scale),
                "--latent_noise_scale",
                str(args.latent_noise_scale),
                "--attention_mode",
                args.attention_mode,
                "--video_backend",
                args.video_backend,
                "--tensor_offload_device",
                args.tensor_offload_device,
            ]
            if args.vae_tiled:
                cmd.extend(
                    [
                        "--vae_encode_tiled",
                        "--vae_encode_tile_size",
                        str(args.vae_encode_tile_size),
                        "--vae_encode_tile_overlap",
                        str(args.vae_encode_tile_overlap),
                        "--vae_decode_tiled",
                        "--vae_decode_tile_size",
                        str(args.vae_decode_tile_size),
                        "--vae_decode_tile_overlap",
                        str(args.vae_decode_tile_overlap),
                    ]
                )
            if args.uniform_batch_size:
                cmd.append("--uniform_batch_size")
            if args.cache_models:
                cmd.extend(["--cache_dit", "--cache_vae"])
            if args.debug:
                cmd.append("--debug")

            run_command(cmd, cwd=comfy_repo)
            move_comfy_outputs(temp_output_dir, output_dir, pending)


def run_official_backend(args: argparse.Namespace, pending: list[Path], output_dir: Path) -> None:
    seedvr_repo = args.seedvr_repo.expanduser().resolve()

    ensure_official_repo(seedvr_repo, clone=not args.no_clone)
    ensure_official_checkpoint(seedvr_repo, download=not args.no_download)
    check_official_runtime(seedvr_repo, skip_cuda_check=args.skip_cuda_check)

    with tempfile.TemporaryDirectory(prefix="seedvr2_inputs_") as temp_input:
        with tempfile.TemporaryDirectory(prefix="seedvr2_outputs_") as temp_output:
            temp_input_dir = Path(temp_input)
            temp_output_dir = Path(temp_output)

            for video_path in pending:
                link_video(video_path, temp_input_dir)

            cmd = [
                args.torchrun,
                f"--nproc-per-node={args.num_gpus}",
                "projects/inference_seedvr2_7b.py",
                "--video_path",
                str(temp_input_dir),
                "--output_dir",
                str(temp_output_dir),
                "--seed",
                str(args.seed),
                "--res_h",
                str(args.res_h),
                "--res_w",
                str(args.res_w),
                "--sp_size",
                str(args.sp_size),
            ]
            if args.out_fps is not None:
                cmd.extend(["--out_fps", str(args.out_fps)])

            run_command(cmd, cwd=seedvr_repo)
            move_official_outputs(temp_output_dir, output_dir, pending)


def upsample(args: argparse.Namespace) -> int:
    raws_dir = args.raws.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not raws_dir.is_dir():
        print(f"Raw video directory not found: {raws_dir}", file=sys.stderr)
        return 1

    videos = find_videos(raws_dir)
    if not videos:
        print(f"No raw videos found in {raws_dir} (supported: {sorted(VIDEO_SUFFIXES)})")
        return 0

    pending = []
    for video_path in videos:
        final_output = output_path_for(video_path, output_dir)
        if final_output.exists() and not args.overwrite:
            print(f"[skip] {final_output.name} already exists")
            continue
        pending.append(video_path)

    if not pending:
        print("All raw videos already have upsampled outputs.")
        return 0

    if args.dry_run:
        print_pending(pending, output_dir, backend=args.backend)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    print_pending(pending, output_dir, backend=args.backend)

    if args.backend == "comfy":
        run_comfy_backend(args, pending, output_dir)
    else:
        run_official_backend(args, pending, output_dir)

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upsample all videos in raws/ with SeedVR2.")
    parser.add_argument(
        "--raws",
        type=Path,
        default=PROJECT_ROOT / "raws",
        help="Folder containing raw input videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "upsampled",
        help="Folder for *_up video outputs.",
    )
    parser.add_argument(
        "--backend",
        choices=["comfy", "official"],
        default="comfy",
        help="SeedVR2 runner to use. 'comfy' supports MPS; 'official' is CUDA-only.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing outputs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pending jobs without cloning, downloading, or running SeedVR.",
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="Do not auto-clone the selected SeedVR2 runner repo if it is missing.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for the Comfy standalone CLI.",
    )

    comfy = parser.add_argument_group("Comfy/MPS backend")
    comfy.add_argument(
        "--comfy-repo",
        type=Path,
        default=DEFAULT_COMFY_REPO,
        help="Path to numz/ComfyUI-SeedVR2_VideoUpscaler.",
    )
    comfy.add_argument(
        "--comfy-model-dir",
        type=Path,
        default=DEFAULT_COMFY_MODEL_DIR,
        help="Directory for Comfy SeedVR2 safetensors/GGUF model files.",
    )
    comfy.add_argument(
        "--dit-model",
        default=DEFAULT_COMFY_DIT_MODEL,
        help="Comfy SeedVR2 DiT model. Defaults to the 7B Q4 GGUF model for MPS practicality.",
    )
    comfy.add_argument("--resolution", type=int, default=720, help="Target short-side resolution.")
    comfy.add_argument("--max-resolution", type=int, default=1280, help="Maximum output edge, 0 disables.")
    comfy.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Frames per batch. Must be 4n+1: 1, 5, 9, 13, ...",
    )
    comfy.add_argument(
        "--chunk-size",
        type=int,
        default=65,
        help="Frames per streaming chunk. 0 loads each full video at once.",
    )
    comfy.add_argument("--temporal-overlap", type=int, default=0, help="Overlap frames between chunks.")
    comfy.add_argument("--prepend-frames", type=int, default=0, help="Prepended reversed frames.")
    comfy.add_argument("--skip-first-frames", type=int, default=0, help="Skip initial input frames.")
    comfy.add_argument("--load-cap", type=int, default=0, help="Maximum frames to process per video.")
    comfy.add_argument(
        "--color-correction",
        default="lab",
        choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
        help="Comfy SeedVR2 color correction method.",
    )
    comfy.add_argument("--input-noise-scale", type=float, default=0.0, help="Input noise scale.")
    comfy.add_argument("--latent-noise-scale", type=float, default=0.0, help="Latent noise scale.")
    comfy.add_argument(
        "--attention-mode",
        default="sdpa",
        choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"],
        help="Attention backend.",
    )
    comfy.add_argument(
        "--video-backend",
        default="opencv",
        choices=["opencv", "ffmpeg"],
        help="Comfy SeedVR2 video writer backend.",
    )
    comfy.add_argument(
        "--tensor-offload-device",
        default="cpu",
        help="Intermediate tensor offload device for Comfy SeedVR2.",
    )
    comfy.add_argument(
        "--no-vae-tiled",
        dest="vae_tiled",
        action="store_false",
        help="Disable VAE encode/decode tiling.",
    )
    comfy.set_defaults(vae_tiled=True)
    comfy.add_argument("--vae-encode-tile-size", type=int, default=512, help="VAE encode tile size.")
    comfy.add_argument("--vae-encode-tile-overlap", type=int, default=64, help="VAE encode tile overlap.")
    comfy.add_argument("--vae-decode-tile-size", type=int, default=512, help="VAE decode tile size.")
    comfy.add_argument("--vae-decode-tile-overlap", type=int, default=64, help="VAE decode tile overlap.")
    comfy.add_argument(
        "--uniform-batch-size",
        action="store_true",
        help="Pad final batch for more stable temporal behavior.",
    )
    comfy.add_argument(
        "--cache-models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep DiT/VAE loaded across files when the Comfy CLI can do so.",
    )
    comfy.add_argument("--debug", action="store_true", help="Enable Comfy SeedVR2 debug logs.")

    official = parser.add_argument_group("Official CUDA backend")
    official.add_argument(
        "--seedvr-repo",
        type=Path,
        default=DEFAULT_OFFICIAL_REPO,
        help="Path to the official ByteDance SeedVR repo.",
    )
    official.add_argument("--torchrun", default="torchrun", help="torchrun executable.")
    official.add_argument("--num-gpus", type=int, default=1, help="torchrun GPU process count.")
    official.add_argument("--sp-size", type=int, default=1, help="Sequence parallel size.")
    official.add_argument("--seed", type=int, default=666, help="SeedVR sampling seed.")
    official.add_argument("--res-h", type=int, default=720, help="Official backend output height.")
    official.add_argument("--res-w", type=int, default=1280, help="Official backend output width.")
    official.add_argument("--out-fps", type=float, default=None, help="Official backend FPS override.")
    official.add_argument(
        "--no-download",
        action="store_true",
        help="Do not auto-download the official SeedVR2-7B checkpoint if it is missing.",
    )
    official.add_argument(
        "--skip-cuda-check",
        action="store_true",
        help="Launch official SeedVR even when PyTorch reports CUDA is unavailable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        sys.exit(upsample(parse_args(argv)))
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
