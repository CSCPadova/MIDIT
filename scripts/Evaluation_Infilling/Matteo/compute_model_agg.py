"""
scripts/Evaluation_Infilling/Matteo/compute_model_agg.py

Compute model aggregate muspy statistics over generated infilling samples.
Output JSON is consumed by compute_js_similarity() as model_agg.

Mirrors compute_reference_agg.py but walks seed_N/reconstructed.mid
subdirs instead of a flat MIDI directory.

Usage:
    python scripts/Evaluation_Infilling/Matteo/compute_model_agg.py \
        --samples_dir /nfsd/.../outputs/samples_1000 \
        --output_json outputs/model_agg.json \
        --max_files   1000
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from muspy_metrics import compute_muspy_metrics

try:
    from js_similarity import aggregate
except ImportError:
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


def _seed_number(path: Path) -> int:
    """Extract integer seed from 'seed_N/reconstructed.mid' path for sorting."""
    m = re.search(r"seed_(\d+)", path.parts[-2])
    return int(m.group(1)) if m else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute model aggregate muspy metrics over generated infilling samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples_dir", required=True,          help="Directory containing seed_N/ subdirs with reconstructed.mid.")
    parser.add_argument("--output_json", required=True,          help="Path to save the aggregate JSON.")
    parser.add_argument("--max_files",   type=int, default=1000, help="Max seeds to process.")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    if not samples_dir.exists():
        print(f"Error: --samples_dir '{samples_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    all_files = sorted(
        samples_dir.glob("seed_*/reconstructed.mid"),
        key=_seed_number,
    )
    n_found = len(all_files)

    if n_found == 0:
        print(f"Error: no seed_*/reconstructed.mid found under '{samples_dir}'.", file=sys.stderr)
        sys.exit(1)

    all_files = all_files[: args.max_files]
    n_total = len(all_files)

    print(f"\n[compute_model_agg]")
    print(f"  samples_dir : {samples_dir}")
    print(f"  output_json : {output_json}")
    print(f"  found       : {n_found} seed_*/reconstructed.mid")
    print(f"  processing  : {n_total} (max_files={args.max_files})")
    print()

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

    agg = aggregate(per_file)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    print(f"\nSaved: {output_json}")
    print(f"Seeds: {n_total}  (ok={n_ok}, all-None={n_fail})\n")
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
