"""
scripts/evaluation/js/compute_reference_agg.py

Compute reference aggregate muspy statistics over a directory of MIDI files.
Output JSON is consumed by compute_js_similarity() as ref_agg.

Usage:
    python scripts/evaluation/js/compute_reference_agg.py \
        --midi_dir data/midi_files/lmd_full/ \
        --output_json outputs/lmd_ref_agg.json \
        --max_files 1000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from muspy_metrics import compute_muspy_metrics

try:
    from js_similarity import aggregate
except ImportError:
    # js_similarity has a top-level import from lilybench which may not be
    # installed. Define aggregate inline — identical logic to js_similarity.aggregate.
    def aggregate(per_file: dict) -> dict:
        if not per_file:
            return {}
        all_metrics = list(next(iter(per_file.values())).keys())
        agg: dict[str, dict] = {}
        for metric in all_metrics:
            values = [
                v[metric]
                for v in per_file.values()
                if v.get(metric) is not None
                and isinstance(v[metric], (int, float))
                and not isinstance(v[metric], bool)
            ]
            n = len(values)
            if n == 0:
                agg[metric] = {"mean": None, "std": None, "n": 0}
            elif n == 1:
                agg[metric] = {"mean": float(values[0]), "std": None, "n": 1}
            else:
                mean = sum(values) / n
                variance = sum((x - mean) ** 2 for x in values) / (n - 1)
                agg[metric] = {"mean": float(mean), "std": math.sqrt(variance), "n": n}
        return agg


_MIDI_SUFFIXES = {".mid", ".midi"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute reference aggregate muspy metrics over a MIDI directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--midi_dir",    required=True,          help="Directory of reference MIDI files (searched recursively).")
    parser.add_argument("--output_json", required=True,          help="Path to save the aggregate JSON.")
    parser.add_argument("--max_files",   type=int, default=1000, help="Max files to process.")
    parser.add_argument("--seed",        type=int, default=42,   help="Random seed for sampling.")
    args = parser.parse_args()

    midi_dir    = Path(args.midi_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    if not midi_dir.exists():
        print(f"Error: --midi_dir '{midi_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Glob all MIDI files recursively
    all_files = sorted([
        p for p in midi_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _MIDI_SUFFIXES
    ])
    n_found = len(all_files)

    if n_found == 0:
        print(f"Error: no .mid/.midi files found under '{midi_dir}'.", file=sys.stderr)
        sys.exit(1)

    # Sample if more than max_files available
    if n_found > args.max_files:
        random.seed(args.seed)
        all_files = sorted(random.sample(all_files, args.max_files))

    n_total = len(all_files)

    print(f"\n[compute_reference_agg]")
    print(f"  midi_dir    : {midi_dir}")
    print(f"  output_json : {output_json}")
    print(f"  found       : {n_found} MIDI files")
    print(f"  processing  : {n_total} (max_files={args.max_files}, seed={args.seed})")
    print()

    # Process each file
    per_file: dict[str, dict] = {}
    n_ok = 0
    n_fail = 0

    for i, path in enumerate(all_files):
        metrics = compute_muspy_metrics(path)
        per_file[str(path)] = metrics

        if any(v is not None for v in metrics.values()):
            n_ok += 1
        else:
            n_fail += 1

        if (i + 1) % 100 == 0 or (i + 1) == n_total:
            print(f"  [{i + 1:>5}/{n_total}]  ok={n_ok}  all-None={n_fail}")

    # Aggregate across all files
    agg = aggregate(per_file)

    # Save JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    # Print summary
    print(f"\nSaved: {output_json}")
    print(f"Files: {n_total}  (ok={n_ok}, all-None={n_fail})\n")
    print(f"{'Metric':<35} {'mean':>10}  {'std':>10}  {'n':>6}")
    print("-" * 68)
    for metric, stats in agg.items():
        mean = stats.get("mean")
        std  = stats.get("std")
        n    = stats.get("n", 0)
        mean_s = f"{mean:.4f}" if mean is not None else "None"
        std_s  = f"{std:.4f}"  if std  is not None else "None"
        print(f"  {metric:<33} {mean_s:>10}  {std_s:>10}  {n:>6}")


if __name__ == "__main__":
    main()