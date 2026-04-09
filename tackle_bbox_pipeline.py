"""
Single-file pipeline: read a tackle clip, detect/track people with YOLO, infer
which figure is most likely the tackler (approach motion toward another player),
and write a video with a bounding box overlaid on that player.

Requires: pip install ultralytics  (also installs PyTorch)

Heuristic mode (default): COCO person + motion heuristic.
Trained mode: fine-tuned single-class `tackler` weights (see train_tackler_detector.py).

Examples:
  python tackle_bbox_pipeline.py path/to/clip.mp4 -o out_with_box.mp4
  python tackle_bbox_pipeline.py clip.mp4 --mode trained --weights runs/detect/tackler/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: install with `pip install ultralytics` "
        "(from the project venv after `pip install -e .` if you add it to pyproject)."
    ) from e


CODEC_CANDIDATES = ["avc1", "mp4v"]


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    for codec in CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for {path} with codecs {CODEC_CANDIDATES}")


def _box_center(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return float(inter / union) if union > 0 else 0.0


def _pair_score(
    box_i: np.ndarray, box_j: np.ndarray, frame_diag: float
) -> float:
    """Higher = more likely a interacting tackle pair (overlap + proximity)."""
    c_i = _box_center(box_i)
    c_j = _box_center(box_j)
    dist = float(np.linalg.norm(c_i - c_j)) / (frame_diag + 1e-6)
    overlap = _iou(box_i, box_j)
    proximity = max(0.0, 0.35 - dist)
    return overlap + 0.6 * proximity


def pick_interaction_pair(
    xyxy: np.ndarray,
    frame_diag: float,
    max_people: int = 4,
) -> tuple[int, int] | None:
    """Return indices of the two detections most likely in a tackle interaction."""
    n = len(xyxy)
    if n < 2:
        return None
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    order = np.argsort(-areas)
    top = order[: min(n, max_people)]
    best: tuple[int, int] | None = None
    best_s = -1.0
    diag = float(frame_diag) + 1e-6
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            i, j = int(top[a]), int(top[b])
            s = _pair_score(xyxy[i], xyxy[j], diag)
            if s > best_s:
                best_s = s
                best = (i, j)
    if best is None or best_s < 0.02:
        return None
    return best


def approach_score(
    track_id: int,
    self_center: np.ndarray,
    other_center: np.ndarray,
    history: dict[int, deque[tuple[float, float]]],
) -> float:
    """How much recent motion points from self toward the other player."""
    h = history.get(track_id)
    if h is None or len(h) < 2:
        return 0.0
    v = np.array(h[-1], dtype=np.float64) - np.array(h[-2], dtype=np.float64)
    toward = other_center - self_center
    n = float(np.linalg.norm(toward)) + 1e-6
    return float(np.dot(v, toward / n))


def choose_tackler_track(
    xyxy: np.ndarray,
    track_ids: np.ndarray,
    history: dict[int, deque[tuple[float, float]]],
    frame_diag: float,
) -> int | None:
    """
    Among tracked people, pick the track ID most likely to be the tackler.
    Uses the best interacting pair and compares approach motion toward the other.
    """
    n = len(xyxy)
    if n == 0:
        return None
    if n == 1:
        return int(track_ids[0])
    pair = pick_interaction_pair(xyxy, frame_diag)
    if pair is None:
        return None

    i, j = pair
    tid_i, tid_j = int(track_ids[i]), int(track_ids[j])
    c_i = _box_center(xyxy[i])
    c_j = _box_center(xyxy[j])
    si = approach_score(tid_i, c_i, c_j, history)
    sj = approach_score(tid_j, c_j, c_i, history)
    if abs(si - sj) < 0.5:
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        return tid_i if areas[i] < areas[j] else tid_j
    return tid_i if si > sj else tid_j


def draw_tackler_box(
    frame_bgr: np.ndarray,
    xyxy: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    label: str = "Tackler",
) -> None:
    x1, y1, x2, y2 = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        frame_bgr,
        label,
        (x1, max(24, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def run_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    weights: str,
    conf: float,
    device: str | None,
) -> None:
    model = YOLO(weights)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_writer(output_path, fps, width, height)
    frame_diag = float(np.hypot(width, height))

    history: dict[int, deque[tuple[float, float]]] = {}
    frame_i = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            kwargs: dict = {
                "persist": True,
                "classes": [0],
                "conf": conf,
                "verbose": False,
            }
            if device:
                kwargs["device"] = device

            results = model.track(frame_bgr, **kwargs)[0]
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                writer.write(frame_bgr)
                frame_i += 1
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            ids_t = boxes.id
            if ids_t is None:
                writer.write(frame_bgr)
                frame_i += 1
                continue

            track_ids = ids_t.cpu().numpy().astype(np.int64)
            for tid, box in zip(track_ids, xyxy, strict=True):
                tid = int(tid)
                if tid not in history:
                    history[tid] = deque(maxlen=10)
                c = _box_center(box)
                history[tid].append((float(c[0]), float(c[1])))

            tackler_tid = choose_tackler_track(xyxy, track_ids, history, frame_diag)
            if tackler_tid is not None:
                match = np.where(track_ids == tackler_tid)[0]
                if len(match) > 0:
                    draw_tackler_box(frame_bgr, xyxy[match[0]])

            writer.write(frame_bgr)
            frame_i += 1
    finally:
        cap.release()
        writer.release()

    print(f"Wrote {frame_i} frames to {output_path}")


def run_trained_tackler_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    weights: str,
    conf: float,
    device: str | None,
) -> None:
    """Draw the highest-confidence tackler box from a single-class fine-tuned model."""
    model = YOLO(weights)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_writer(output_path, fps, width, height)
    frame_i = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            kwargs: dict = {"persist": True, "conf": conf, "verbose": False}
            if device:
                kwargs["device"] = device

            results = model.track(frame_bgr, **kwargs)[0]
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                writer.write(frame_bgr)
                frame_i += 1
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            best = int(np.argmax(confs))
            draw_tackler_box(frame_bgr, xyxy[best])

            writer.write(frame_bgr)
            frame_i += 1
    finally:
        cap.release()
        writer.release()

    print(f"Wrote {frame_i} frames to {output_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay a tackler bounding box on football tackle video "
        "(COCO person + heuristic, or fine-tuned tackler detector)."
    )
    p.add_argument("input", type=Path, help="Input video path (.mp4, .mov, ...)")
    p.add_argument(
        "--mode",
        choices=("heuristic", "trained"),
        default="heuristic",
        help="heuristic: COCO person + motion rules; trained: single-class tackler checkpoint",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: <input_stem>_tackler_box.mp4)",
    )
    p.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="heuristic: COCO checkpoint; trained: your runs/.../best.pt",
    )
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument(
        "--device",
        default=None,
        help="torch device, e.g. mps, cuda:0, cpu (default: auto)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        sys.exit(f"Input not found: {inp}")
    out = args.output
    if out is None:
        out = inp.with_name(f"{inp.stem}_tackler_box.mp4")
    else:
        out = out.expanduser().resolve()

    if args.mode == "trained":
        run_trained_tackler_pipeline(
            inp,
            out,
            weights=args.weights,
            conf=args.conf,
            device=args.device,
        )
    else:
        run_pipeline(
            inp,
            out,
            weights=args.weights,
            conf=args.conf,
            device=args.device,
        )


if __name__ == "__main__":
    main()
