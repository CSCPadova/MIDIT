"""
scripts/Evaluation_Infilling/Matteo/compute_js_score.py

Final step of the JS similarity pipeline.
Loads model_agg.json + ref_agg.json and computes JS Divergence Similarity.

JS Similarity = 100 × exp(−2 × mean_JS) over 3 metrics:
  - muspy_polyphony_rate
  - muspy_groove_consistency
  - muspy_scale_consistency

Usage:
    python scripts/Evaluation_Infilling/Matteo/compute_js_score.py \
        --model_agg  outputs/model_agg.json \
        --ref_agg    outputs/ref_agg.json \
        --model_name MIDIT \
        --output_json outputs/js_score.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from js_similarity import JS_METRICS, _js_divergence_gaussian


def _load_json(path: Path) -> dict:
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute JS Divergence Similarity between model and reference distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_agg",   required=True,   help="Path to model_agg.json (output of compute_model_agg.py).")
    parser.add_argument("--ref_agg",     required=True,   help="Path to ref_agg.json (output of compute_reference_agg.py).")
    parser.add_argument("--output_json", default=None,    help="Optional path to save results JSON.")
    parser.add_argument("--model_name",  default="MIDIT", help="Label for the model in output.")
    args = parser.parse_args()

    model_agg = _load_json(Path(args.model_agg).expanduser().resolve())
    ref_agg   = _load_json(Path(args.ref_agg).expanduser().resolve())

    print(f"\n{'='*60}")
    print(f"  JS Similarity — {args.model_name}")
    print(f"{'='*60}")
    print(f"  model_agg : {args.model_agg}")
    print(f"  ref_agg   : {args.ref_agg}")
    print()

    print(f"  {'Metric':<35} {'model μ':>9}  {'model σ':>9}  {'ref μ':>9}  {'ref σ':>9}  {'JS div':>9}")
    print(f"  {'-'*35} {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")

    js_values: list[float] = []
    skipped: list[str] = []
    per_metric: dict[str, dict] = {}

    for metric in JS_METRICS:
        m_stats = model_agg.get(metric, {})
        r_stats = ref_agg.get(metric, {})

        mu1, s1 = m_stats.get("mean"), m_stats.get("std")
        mu2, s2 = r_stats.get("mean"), r_stats.get("std")
        n_model = m_stats.get("n", 0)
        n_ref   = r_stats.get("n", 0)

        mu1_s = f"{mu1:.4f}" if mu1 is not None else "None"
        s1_s  = f"{s1:.4f}"  if s1  is not None else "None"
        mu2_s = f"{mu2:.4f}" if mu2 is not None else "None"
        s2_s  = f"{s2:.4f}"  if s2  is not None else "None"

        if any(v is None for v in (mu1, s1, mu2, s2)) or s1 <= 0 or s2 <= 0:
            reason = "missing mean/std" if any(v is None for v in (mu1, s1, mu2, s2)) else "zero std"
            print(f"  {metric:<35} {mu1_s:>9}  {s1_s:>9}  {mu2_s:>9}  {s2_s:>9}  {'SKIP':>9}  ({reason})")
            skipped.append(metric)
            per_metric[metric] = {
                "model_mean": mu1, "model_std": s1, "model_n": n_model,
                "ref_mean": mu2,   "ref_std": s2,   "ref_n": n_ref,
                "js_divergence": None, "skipped": True, "skip_reason": reason,
            }
            continue

        js = _js_divergence_gaussian(mu1, s1, mu2, s2)
        js_values.append(js)
        print(f"  {metric:<35} {mu1_s:>9}  {s1_s:>9}  {mu2_s:>9}  {s2_s:>9}  {js:>9.6f}")
        per_metric[metric] = {
            "model_mean": mu1, "model_std": s1, "model_n": n_model,
            "ref_mean": mu2,   "ref_std": s2,   "ref_n": n_ref,
            "js_divergence": js, "skipped": False,
        }

    print()

    if skipped:
        print(f"  WARNING: {len(skipped)} metric(s) skipped — JS score is None.")
        for m in skipped:
            print(f"    - {m}")
        score = None
    else:
        mean_js = sum(js_values) / len(js_values)
        score = 100.0 * math.exp(-2.0 * mean_js)

    print(f"{'='*60}")
    if score is not None:
        print(f"  JS Similarity ({args.model_name}) = {score:.4f}")
    else:
        print(f"  JS Similarity ({args.model_name}) = None  (incomplete metrics)")
    print(f"{'='*60}\n")

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "model_name": args.model_name,
            "js_similarity": score,
            "metrics_used": [m for m in JS_METRICS if m not in skipped],
            "metrics_skipped": skipped,
            "per_metric": per_metric,
        }
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved: {output_path}\n")


if __name__ == "__main__":
    main()
