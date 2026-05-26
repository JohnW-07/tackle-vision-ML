"""
Join combined_components.json with labeled_data_labels.json to produce
a training-ready dataset.

Filters applied:
  1. Only clips present in combined_components.json (have extracted biomechanics)
  2. Only clips that have at least one component with real (non-null) values
  3. Only clips that have a matching entry in the labels file

  Clips with partial null features (e.g. neck_flexion_deg null but other metrics present)
  are kept — the ML pipeline handles missing values via imputation.

Output (out/training_set.json):
  {
    "clip_count": N,
    "clips": {
      "<video_id>": {
        ...full component data from combined_components...,
        "labels": {
          "head_neck": float, "upper_extremity": float,
          "com_spine": float, "lower_extremity": float,
          "overall_avg": float
        }
      }
    }
  }

Usage:
  python scripts/build_training_set.py \\
    --combined   out/combined_components.json \\
    --labels     labeled_data_labels.json \\
    --output     out/training_set.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# ID normalisation (mirrors combine_components.py)
# ---------------------------------------------------------------------------

_SUFFIX_STRIP   = re.compile(r"[_\s]+(comp|wmv|m4v|mp4|mov|avi|mkv)$", re.IGNORECASE)
_DATE_PREFIX_RE = re.compile(r"^\d{1,4}[.\-_/]\d{1,2}[.\-_/]\d{2,4}[.\-_/\s]+")


def _normalise(vid: str) -> str:
    s = vid.strip()
    s = _DATE_PREFIX_RE.sub("", s)
    s = s.lower()
    s = re.sub(r"[\s\-\.]+", "_", s)
    s = _SUFFIX_STRIP.sub("", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _has_any_real_data(clip: dict) -> bool:
    """True if at least one component has at least one non-null metric value."""
    for comp_data in clip.get("components", {}).values():
        if not comp_data.get("metrics"):
            continue
        for val in comp_data.get("summary", {}).values():
            if isinstance(val, dict):
                if any(v is not None for v in val.values()):
                    return True
            elif val is not None:
                return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Join combined_components.json with labels to build training set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--combined", type=Path, default=Path("out/combined_components.json"))
    p.add_argument("--labels",   type=Path, default=Path("labeled_data_labels.json"))
    p.add_argument("--output",   type=Path, default=Path("out/training_set.json"))
    args = p.parse_args(argv)

    with args.combined.open() as f:
        all_clips: dict[str, dict] = json.load(f)["clips"]

    with args.labels.open() as f:
        raw_labels: dict[str, dict] = json.load(f)["labels"]

    total_clips = len(all_clips)

    # Step 1: drop clips with no real data at all
    with_data    = {vid: clip for vid, clip in all_clips.items() if _has_any_real_data(clip)}
    dropped_empty = total_clips - len(with_data)

    # Step 2: build normalised label lookup
    label_norm = {_normalise(k): (k, v) for k, v in raw_labels.items()}

    # Step 3: match and join
    training: dict[str, dict] = {}
    unmatched: list[str] = []

    for vid, clip in with_data.items():
        n = _normalise(vid)
        if n in label_norm:
            _, label_scores = label_norm[n]
            entry = dict(clip)
            entry["labels"] = label_scores
            training[vid] = entry
        else:
            unmatched.append(vid)

    # Report
    print(f"Combined clips total:           {total_clips}")
    print(f"Dropped (no component data):    {dropped_empty}")
    print(f"Remaining with data:            {len(with_data)}")
    print(f"Matched to labels:              {len(training)}")
    print(f"Dropped (no label found):       {len(unmatched)}")

    if unmatched:
        print("\nClips with biomechanical data but no label:")
        for v in sorted(unmatched):
            print(f"  {v}")

    unmatched_labels = [k for k in raw_labels if _normalise(k) not in {_normalise(v) for v in with_data}]
    print(f"\nLabels with no matching clip:   {len(unmatched_labels)}")
    if unmatched_labels:
        print("  (these clips were not in the extraction run)")

    print(f"\nFinal training set: {len(training)} clips")

    # Component population in training set
    components = ["head_neck", "upper_extremity", "com_spine", "lower_extremity"]
    print("\nComponent coverage in training set:")
    for comp in components:
        n = sum(1 for c in training.values() if c["components"].get(comp, {}).get("metrics"))
        print(f"  {comp:<22} {n}/{len(training)}")

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": "training_set.v1",
        "exported_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clip_count":     len(training),
        "clips":          training,
    }
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {len(training)} clips → {args.output}")


if __name__ == "__main__":
    main()
