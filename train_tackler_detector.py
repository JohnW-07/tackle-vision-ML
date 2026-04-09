"""
Fine-tune a YOLO detector on your tackler dataset (dataset/data.yaml).

Requires labeled images: for each image in dataset/images/{train,val}/,
a matching .txt in dataset/labels/{train,val}/ with YOLO lines:
  0 xc yc w h
(all normalized 0–1; class 0 = tackler).

Example:
  python train_tackler_detector.py --epochs 80 --batch 8
  python train_tackler_detector.py --model yolov8s.pt --imgsz 960 --device mps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install ultralytics: pip install ultralytics") from e

PROJECT_ROOT = Path(__file__).resolve().parent


def _count_labeled_images(images_dir: Path, labels_dir: Path) -> tuple[int, int]:
    """Return (n_images, n_with_non_empty_label)."""
    if not images_dir.is_dir():
        return 0, 0
    n_img = 0
    n_labeled = 0
    for img in images_dir.glob("*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        n_img += 1
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.is_file() and lbl.stat().st_size > 0:
            n_labeled += 1
    return n_img, n_labeled


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO tackler detector from dataset/data.yaml")
    p.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "data.yaml",
        help="Ultralytics data yaml",
    )
    p.add_argument("--model", default="yolov8n.pt", help="Starting checkpoint")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="Use -1 for auto")
    p.add_argument("--device", default=None, help="e.g. cpu, mps, cuda:0")
    p.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "detect")
    p.add_argument("--name", default="tackler", help="Run name under --project")
    p.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    p.add_argument(
        "--skip-count-check",
        action="store_true",
        help="Do not verify label files exist before training",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_yaml = args.data.expanduser().resolve()
    if not data_yaml.is_file():
        sys.exit(f"data yaml not found: {data_yaml}")

    dataset_root = data_yaml.parent
    if not args.skip_count_check:
        counts = []
        for split in ("train", "val"):
            img_dir = dataset_root / "images" / split
            lbl_dir = dataset_root / "labels" / split
            n_img, n_lbl = _count_labeled_images(img_dir, lbl_dir)
            counts.append((split, n_img, n_lbl))
            print(f"  {split}: {n_lbl} labeled images / {n_img} total images in {img_dir.name}/")
        train_labeled = next(n for s, _, n in counts if s == "train")
        if train_labeled < 1:
            sys.exit(
                "No training labels found. For each image in dataset/images/train/, "
                "add dataset/labels/train/<same_stem>.txt with YOLO lines (class 0 = tackler)."
            )

    model = YOLO(args.model)
    train_kw: dict = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": True,
    }
    if args.batch >= 0:
        train_kw["batch"] = args.batch
    if args.device:
        train_kw["device"] = args.device

    model.train(**train_kw)
    weights_dir = args.project / args.name / "weights"
    print(f"Training finished. Best weights: {weights_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
