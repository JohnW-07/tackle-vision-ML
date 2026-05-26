"""
Combine all four biomechanical components into a single unified JSON.

Sources (all optional):
  --all-clips      schema/all_clips.components.json
                     upper_extremity populated, head_neck/com_spine/lower_extremity empty
  --head-neck      output of extract_head_neck.py
                     head_neck populated
  --lower-ext      output of extract_lower_extremity.py
                     lower_extremity populated
  --com-spine-ndjson  schema/data (1).json  (newline-delimited JSON, one record per line)
                     thin com_spine metrics available; IDs may differ (see --id-map)

Output follows components.v1 schema with all four components merged per clip.

ID normalisation:
  By default, clip IDs are matched exactly.  If your sources use different naming
  conventions (e.g. "08.01.2023 Mp1042 Imp8_comp" vs "mp1042_imp8_comp_m4v"), pass
  an ID-map CSV (two columns: source_id, canonical_id) via --id-map.

Examples:
  # Merge all sources
  python scripts/combine_components.py \\
    --all-clips   schema/all_clips.components.json \\
    --head-neck   out/head_neck.json \\
    --lower-ext   out/lower_ext.json \\
    --com-spine-ndjson "schema/data (1).json" \\
    --output      out/combined_components.json

  # Merge only what you have
  python scripts/combine_components.py \\
    --all-clips  schema/all_clips.components.json \\
    --head-neck  out/head_neck.json \\
    --output     out/combined_components.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOL_VERSION = "combine-components 0.1.0"
EMPTY_COMPONENT = {"summary": {}, "metrics": [], "level": None, "time_series": None}
ALL_COMPONENTS   = ("head_neck", "upper_extremity", "com_spine", "lower_extremity")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_wrapped_json(path: Path) -> dict[str, dict]:
    """Load a multi-clip wrapper JSON → {video_id: clip_dict}."""
    with path.open() as f:
        data = json.load(f)
    clips = data.get("clips", {})
    if not clips:
        raise ValueError(f"No 'clips' key found in {path}")
    return dict(clips)


def _load_ndjson(path: Path) -> dict[str, dict]:
    """Load newline-delimited JSON (one record per line) → {video_id: clip_dict}."""
    result: dict[str, dict] = {}
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] {path.name} line {lineno}: parse error — {e}", file=sys.stderr)
                continue
            vid = rec.get("video_id")
            if not vid:
                continue
            result[vid] = rec
    return result


def _load_id_map(path: Path) -> dict[str, str]:
    """
    Load a two-column CSV (source_id, canonical_id) and return a lookup dict.
    This lets you reconcile IDs across sources with different naming conventions.
    """
    mapping: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                src, canonical = row[0].strip(), row[1].strip()
                if src and canonical:
                    mapping[src] = canonical
    return mapping


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def _empty_clip_skeleton(video_id: str) -> dict:
    return {
        "schema_version": "components.v1",
        "video_id": video_id,
        "meta": None,
        "keypoints": None,
        "components": {c: dict(EMPTY_COMPONENT) for c in ALL_COMPONENTS},
        "global": {"quality_flag": None},
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _pick_meta(existing: dict | None, incoming: dict | None) -> dict | None:
    """Keep whichever meta is non-None; prefer the one with more detail."""
    if existing and incoming:
        # Prefer the one from the detection pipeline (has poc_index etc.)
        return existing if existing.get("poc_index") is not None else incoming
    return existing or incoming


def _merge_clip(
    base: dict,
    source: dict,
    components_to_take: list[str],
) -> dict:
    """
    Merge component data from source into base.
    Only components listed in components_to_take are copied if they have metrics.
    Meta and keypoints from source fill gaps in base.
    """
    base["meta"]      = _pick_meta(base.get("meta"),      source.get("meta"))
    base["keypoints"] = base.get("keypoints") or source.get("keypoints")

    src_components = source.get("components", {})
    for comp in components_to_take:
        src_comp = src_components.get(comp)
        if src_comp is None:
            continue
        if not src_comp.get("metrics"):
            continue
        # Only overwrite if currently empty, OR incoming has more metrics
        existing = base["components"].get(comp, EMPTY_COMPONENT)
        if not existing.get("metrics") or len(src_comp["metrics"]) > len(existing.get("metrics", [])):
            base["components"][comp] = src_comp

    # quality_flag: True only when all 4 components have at least one metric
    populated = sum(
        1 for c in ALL_COMPONENTS if base["components"].get(c, {}).get("metrics")
    )
    base["global"]["quality_flag"] = (populated == len(ALL_COMPONENTS))

    return base


def collect_all_video_ids(*sources: dict[str, dict]) -> set[str]:
    ids: set[str] = set()
    for src in sources:
        ids.update(src.keys())
    return ids


def apply_id_map(clips: dict[str, dict], id_map: dict[str, str]) -> dict[str, dict]:
    """Remap keys in clips dict using id_map; unmapped keys stay unchanged."""
    remapped: dict[str, dict] = {}
    for vid, clip in clips.items():
        canonical = id_map.get(vid, vid)
        clip["video_id"] = canonical
        remapped[canonical] = clip
    return remapped


# ---------------------------------------------------------------------------
# Fuzzy ID normalisation
# ---------------------------------------------------------------------------

_SUFFIX_STRIP = re.compile(r"[_\s]+(comp|wmv|m4v|mp4|mov|avi|mkv)$", re.IGNORECASE)

# Matches date prefixes with any separator: 08.30.22, 08-30-22, 08_30_22, 2023.01.08 …
_DATE_PREFIX_RE = re.compile(
    r"^\d{1,4}[.\-_/]\d{1,2}[.\-_/]\d{2,4}[.\-_/\s]+"
)


def _normalise_id(vid: str) -> str:
    """
    Normalise a clip ID for fuzzy cross-source matching.
    1. Strip leading date prefix (handles  MM.DD.YY, MM_DD_YY, YYYY.MM.DD, etc.)
    2. Lowercase everything
    3. Replace separators (space, hyphen, dot) with underscores
    4. Strip common file-extension suffixes  (_wmv, _m4v, _comp, …)
    5. Collapse repeated underscores; strip edge underscores
    """
    s = vid.strip()
    s = _DATE_PREFIX_RE.sub("", s)
    s = s.lower()
    s = re.sub(r"[\s\-\.]+", "_", s)
    s = _SUFFIX_STRIP.sub("", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def build_fuzzy_id_map(
    *source_dicts: dict[str, dict],
) -> dict[str, str]:
    """
    Build a normalised-key → canonical-id mapping by gathering all raw IDs from
    every source, normalising each, and returning the mapping.
    Canonical ID = the first raw ID seen for each normalised key (preserves original).
    """
    norm_to_canonical: dict[str, str] = {}
    for src in source_dicts:
        for raw_id in src:
            norm = _normalise_id(raw_id)
            if norm not in norm_to_canonical:
                norm_to_canonical[norm] = raw_id
    return norm_to_canonical


def remap_with_fuzzy(
    clips: dict[str, dict],
    norm_to_canonical: dict[str, str],
) -> dict[str, dict]:
    """
    Remap clip IDs to canonical IDs via fuzzy normalisation.
    If a normalised key maps to a known canonical ID, use that; otherwise keep original.
    """
    remapped: dict[str, dict] = {}
    for raw_id, clip in clips.items():
        norm = _normalise_id(raw_id)
        canonical = norm_to_canonical.get(norm, raw_id)
        clip["video_id"] = canonical
        remapped[canonical] = clip
    return remapped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Merge head_neck, upper_extremity, com_spine, lower_extremity into one JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--all-clips",       type=Path, default=None,
                   help="schema/all_clips.components.json (upper_extremity source)")
    p.add_argument("--head-neck",       type=Path, default=None,
                   help="Output of extract_head_neck.py")
    p.add_argument("--lower-ext",       type=Path, default=None,
                   help="Output of extract_lower_extremity.py")
    p.add_argument("--com-spine-ndjson",type=Path, default=None,
                   help='NDJSON source for com_spine (e.g. schema/data (1).json)')
    p.add_argument("--output",          type=Path, required=True,
                   help="Output combined JSON file.")
    p.add_argument("--id-map",          type=Path, default=None,
                   help="Optional CSV (source_id, canonical_id) for reconciling clip IDs.")
    p.add_argument("--fuzzy-match",     action="store_true",
                   help="Auto-normalise clip IDs across sources (strips dates, lowercases, "
                        "collapses separators) to improve cross-source matching.")
    p.add_argument("--min-components",  type=int, default=1,
                   help="Only include clips with at least this many populated components (default: 1).")
    args = p.parse_args(argv)

    id_map: dict[str, str] = {}
    if args.id_map:
        id_map = _load_id_map(args.id_map)
        print(f"Loaded {len(id_map)} ID mappings from {args.id_map}")

    # --- Load raw source dicts (before any remapping) ---
    raw_sources: dict[str, dict[str, dict]] = {}
    if args.all_clips:
        raw_sources["all_clips"]       = _load_wrapped_json(args.all_clips)
    if args.head_neck:
        raw_sources["head_neck"]       = _load_wrapped_json(args.head_neck)
    if args.lower_ext:
        raw_sources["lower_ext"]       = _load_wrapped_json(args.lower_ext)
    if args.com_spine_ndjson:
        raw_sources["com_spine_ndjson"] = _load_ndjson(args.com_spine_ndjson)

    if not raw_sources:
        sys.exit("No source files provided.")

    # --- Build fuzzy normalisation map if requested ---
    norm_to_canonical: dict[str, str] = {}
    if args.fuzzy_match:
        norm_to_canonical = build_fuzzy_id_map(*raw_sources.values())
        print(f"Fuzzy matching: {len(norm_to_canonical)} normalised keys built")

    def _remap(clips: dict[str, dict]) -> dict[str, dict]:
        if id_map:
            clips = apply_id_map(clips, id_map)
        if args.fuzzy_match:
            clips = remap_with_fuzzy(clips, norm_to_canonical)
        return clips

    # --- Register sources with their target components ---
    sources: list[tuple[dict[str, dict], list[str]]] = []

    if "all_clips" in raw_sources:
        d = _remap(raw_sources["all_clips"])
        sources.append((d, ["upper_extremity", "com_spine"]))
        print(f"all_clips:        {len(d)} clips → upper_extremity, com_spine (if populated)")

    if "head_neck" in raw_sources:
        d = _remap(raw_sources["head_neck"])
        sources.append((d, ["head_neck"]))
        print(f"head_neck:        {len(d)} clips → head_neck")

    if "lower_ext" in raw_sources:
        d = _remap(raw_sources["lower_ext"])
        sources.append((d, ["lower_extremity"]))
        print(f"lower_extremity:  {len(d)} clips → lower_extremity")

    if "com_spine_ndjson" in raw_sources:
        d = _remap(raw_sources["com_spine_ndjson"])
        sources.append((d, ["com_spine"]))
        print(f"com_spine NDJSON: {len(d)} clips → com_spine")

    # --- Collect all video IDs ---
    all_ids = collect_all_video_ids(*[s for s, _ in sources])
    print(f"\nTotal unique clip IDs across all sources: {len(all_ids)}")

    # --- Merge ---
    combined: dict[str, dict] = {}
    for vid in sorted(all_ids):
        clip = _empty_clip_skeleton(vid)
        for src_clips, comps in sources:
            src_clip = src_clips.get(vid)
            if src_clip is None:
                continue
            clip = _merge_clip(clip, src_clip, comps)
        combined[vid] = clip

    # --- Filter by min_components ---
    kept: dict[str, dict] = {}
    for vid, clip in combined.items():
        populated = sum(
            1 for c in ALL_COMPONENTS if clip["components"].get(c, {}).get("metrics")
        )
        if populated >= args.min_components:
            kept[vid] = clip

    print(f"Clips with ≥ {args.min_components} component(s) populated: {len(kept)} / {len(combined)}")

    # Summary table
    print("\nComponent population summary:")
    for comp in ALL_COMPONENTS:
        n = sum(1 for v in kept.values() if v["components"].get(comp, {}).get("metrics"))
        print(f"  {comp:<20} {n}/{len(kept)} clips")

    # --- Write output ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": "components.v1",
        "tool_version":   TOOL_VERSION,
        "exported_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clip_count":     len(kept),
        "clips":          kept,
    }
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {len(kept)} clip(s) → {args.output}")


if __name__ == "__main__":
    main()
