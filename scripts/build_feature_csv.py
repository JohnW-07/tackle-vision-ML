"""
Flatten out/training_set.json into a feature-matrix CSV for model training.

Column naming convention:
  Aggregate metrics  → <component>.<metric>_<stat>
                       e.g. head_neck.neck_flexion_deg_mean
                            lower_extremity.knee_flex_left_deg_at_poc
  Scalar metrics     → <component>.<metric>
                       e.g. head_neck.ear_below_shoulder_frac
                            upper_extremity.both_arms_extended_frac

Label columns (Y):
  label_head_neck, label_upper_extremity, label_com_spine,
  label_lower_extremity, label_overall_avg

The first column is always `video_id`.

Usage:
  python scripts/build_feature_csv.py \\
    --training-set out/training_set.json \\
    --output       out/features.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AGGREGATE_STATS = ("mean", "std", "min", "max", "p10", "p90", "at_poc")
LABEL_KEYS      = ("head_neck", "upper_extremity", "com_spine", "lower_extremity", "overall_avg")
COMPONENTS      = ("head_neck", "upper_extremity", "com_spine", "lower_extremity")


def _flatten_clip(video_id: str, clip: dict) -> dict[str, float | None]:
    row: dict[str, float | None] = {"video_id": video_id}

    for comp in COMPONENTS:
        comp_data = clip.get("components", {}).get(comp, {})
        summary   = comp_data.get("summary", {})
        for metric, val in summary.items():
            if isinstance(val, dict):
                for stat in AGGREGATE_STATS:
                    col = f"{comp}.{metric}_{stat}"
                    row[col] = val.get(stat)
            else:
                col = f"{comp}.{metric}"
                row[col] = val

    labels = clip.get("labels", {})
    for lk in LABEL_KEYS:
        row[f"label_{lk}"] = labels.get(lk)

    return row


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Flatten training_set.json → feature CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--training-set", type=Path, default=Path("out/training_set.json"))
    p.add_argument("--output",       type=Path, default=Path("out/features.csv"))
    args = p.parse_args(argv)

    with args.training_set.open() as f:
        data = json.load(f)

    clips = data["clips"]
    rows  = [_flatten_clip(vid, clip) for vid, clip in clips.items()]

    # Union of all column names (preserving insertion order across rows)
    all_cols: list[str] = ["video_id"]
    seen: set[str] = {"video_id"}
    for row in rows:
        for col in row:
            if col not in seen:
                all_cols.append(col)
                seen.add(col)

    # Separate feature cols from label cols
    feature_cols = [c for c in all_cols if not c.startswith("label_") and c != "video_id"]
    label_cols   = [c for c in all_cols if c.startswith("label_")]
    ordered_cols = ["video_id"] + feature_cols + label_cols

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Fill missing columns with empty string (→ NaN when read by pandas)
            full_row = {col: row.get(col, None) for col in ordered_cols}
            writer.writerow(full_row)

    # Summary
    non_null_counts = {col: sum(1 for r in rows if r.get(col) is not None) for col in feature_cols}
    n = len(rows)

    print(f"Clips:          {n}")
    print(f"Feature cols:   {len(feature_cols)}")
    print(f"Label cols:     {len(label_cols)}")
    print()
    print("Feature coverage (non-null / total):")
    prev_comp = None
    for col in feature_cols:
        comp = col.split(".")[0]
        if comp != prev_comp:
            print(f"  [{comp}]")
            prev_comp = comp
        nn = non_null_counts[col]
        bar = "#" * nn + "." * (n - nn)
        print(f"    {col:<55} {nn:>2}/{n}  [{bar}]")

    print()
    print("Label distribution:")
    for lk in LABEL_KEYS:
        col = f"label_{lk}"
        vals = [r[col] for r in rows if r.get(col) is not None]
        if vals:
            avg = sum(vals) / len(vals)
            lo, hi = min(vals), max(vals)
            print(f"  {col:<30} n={len(vals)}  mean={avg:.2f}  range=[{lo:.2f}, {hi:.2f}]")

    print(f"\nWrote {n} rows × {len(ordered_cols)} cols → {args.output}")


if __name__ == "__main__":
    main()
