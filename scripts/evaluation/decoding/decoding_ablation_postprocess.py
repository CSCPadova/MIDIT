"""
scripts/evaluation/decoding/decoding_ablation_postprocess.py

Post-processes the decoding ablation: adds S-I metrics (pitch_class_entropy,
groove_consistency, scale_consistency, pitch_range, n_pitches_used) for each
reconstruction AND the original-reference S-I values (+ deltas), merging them
with the existing decoding_ablation_results.csv (condition, seed_idx, cp, gs, pche).

CPU only, no model, no GPU. Reuses the project's own S-I computation
(_compute_si_metrics) with the SAME bar_range for original and reconstruction so
the comparison is consistent. Does NOT modify any existing file; writes only
decoding_ablation_full_results.csv into --ablation_dir.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))                       # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "evaluation"))

from compute_muspy_metrics import _compute_si_metrics
from compute_infilling_metrics import _bar_boundaries

import symusic
import csv
import argparse
import math
import numpy as np
from collections import defaultdict

# muspy S-I dict key -> our column names (recon) and orig_* columns
SI_KEYS = ["pitch_class_entropy", "groove_consistency", "scale_consistency",
           "pitch_range", "n_pitches_used"]
ORIG_COL = {  # orig_* column name per S-I key
    "pitch_class_entropy": "orig_pce",
    "groove_consistency":  "orig_gc",
    "scale_consistency":   "orig_sc",
    "pitch_range":         "orig_pitch_range",
    "n_pitches_used":      "orig_n_pitches",
}
FULL_COLUMNS = [
    "condition", "seed_idx",
    "cp", "gs", "pche",
    "pitch_class_entropy", "groove_consistency", "scale_consistency",
    "pitch_range", "n_pitches_used",
    "orig_pce", "orig_gc", "orig_sc", "orig_pitch_range", "orig_n_pitches",
    "delta_pce", "delta_pitch_range", "delta_n_pitches",
]


def _f(x):
    """Parse to float; return nan on failure/empty."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _delta(a, b):
    """a - b, propagating nan."""
    if a is None or b is None or math.isnan(a) or math.isnan(b):
        return float("nan")
    return a - b


def _ms(values):
    """mean +/- std (ddof=1) over non-nan values; 'N/A' if none."""
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], float)
    if v.size == 0:
        return "N/A"
    if v.size == 1:
        return f"{v[0]:.3f} ± 0.000"
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def detect_conditions(ablation_dir, seed_dirs):
    """Auto-detect condition names from seed_0 (or the first available seed dir)."""
    ref = ablation_dir / "seed_0"
    if not (ref.exists() and ref.is_dir()):
        ref = seed_dirs[0] if seed_dirs else None
    if ref is None:
        return []
    return sorted([d.name for d in ref.iterdir() if d.is_dir()])


def main():
    ap = argparse.ArgumentParser(description="Add S-I metrics + reference deltas to the decoding ablation.")
    ap.add_argument("--ablation_dir", required=True,
                    help="Dir with seed_N/ subdirs and decoding_ablation_results.csv.")
    ap.add_argument("--mask_bars", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Condition names; if omitted, auto-detected from seed_0 subdirs.")
    args = ap.parse_args()

    ablation_dir = Path(args.ablation_dir)
    if not ablation_dir.exists():
        raise SystemExit(f"ERROR: {ablation_dir} does not exist.")

    def _seed_num(p):
        try:
            return int(p.name.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    seed_dirs = sorted([d for d in ablation_dir.glob("seed_*") if d.is_dir()], key=_seed_num)
    if not seed_dirs:
        raise SystemExit(f"ERROR: no seed_*/ dirs under {ablation_dir}.")

    conditions = args.conditions or detect_conditions(ablation_dir, seed_dirs)
    if not conditions:
        raise SystemExit("ERROR: no conditions found/auto-detected.")
    print(f"[postprocess] seeds={len(seed_dirs)}  conditions={len(conditions)}: {conditions}")

    # ── read existing P-I CSV (condition, seed_idx -> cp/gs/pche) ─────────────
    pi_csv = ablation_dir / "decoding_ablation_results.csv"
    pi = {}
    if pi_csv.exists():
        with open(pi_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pi[(r["condition"], str(r["seed_idx"]))] = {
                    "cp": _f(r.get("cp")), "gs": _f(r.get("gs")), "pche": _f(r.get("pche"))}
        print(f"  loaded existing P-I CSV: {len(pi)} rows")
    else:
        print(f"  WARNING: {pi_csv} not found — cp/gs/pche will be NaN")

    # ── compute S-I per (seed, condition) ─────────────────────────────────────
    rows = []
    n_skip_seed = n_skip_cond = 0
    for d in seed_dirs:
        seed_idx = _seed_num(d)
        orig_path = d / "original.mid"
        if not orig_path.exists():
            print(f"  Skipping seed {seed_idx}: no original.mid")
            n_skip_seed += 1
            continue

        try:
            orig_score = symusic.Score(str(orig_path))
            boundaries = _bar_boundaries(orig_score)
        except Exception as e:  # noqa: BLE001
            print(f"  Skipping seed {seed_idx}: could not read/boundaries original ({e})")
            n_skip_seed += 1
            continue

        bar_range = [(boundaries[b], boundaries[b + 1])
                     for b in args.mask_bars if b + 1 < len(boundaries)]
        if not bar_range:
            print(f"  Skipping seed {seed_idx}: empty bar_range (score too short)")
            n_skip_seed += 1
            continue

        # ORIGINAL reference S-I (same bar_range)
        orig_si = _compute_si_metrics(orig_path, bar_range=bar_range)

        for cond in conditions:
            recon_path = d / cond / "reconstructed.mid"
            if not recon_path.exists():
                print(f"  Skipping seed {seed_idx} / {cond}: no reconstructed.mid")
                n_skip_cond += 1
                continue

            recon_si = _compute_si_metrics(recon_path, bar_range=bar_range)

            pikey = pi.get((cond, str(seed_idx)), {"cp": float("nan"),
                                                   "gs": float("nan"), "pche": float("nan")})
            row = {"condition": cond, "seed_idx": seed_idx,
                   "cp": pikey["cp"], "gs": pikey["gs"], "pche": pikey["pche"]}
            for k in SI_KEYS:
                rv = _f(recon_si.get(k))
                ov = _f(orig_si.get(k))
                row[k] = rv
                row[ORIG_COL[k]] = ov
            row["delta_pce"] = _delta(row["pitch_class_entropy"], row["orig_pce"])
            row["delta_pitch_range"] = _delta(row["pitch_range"], row["orig_pitch_range"])
            row["delta_n_pitches"] = _delta(row["n_pitches_used"], row["orig_n_pitches"])
            rows.append(row)

    # ── write merged CSV ──────────────────────────────────────────────────────
    out_csv = ablation_dir / "decoding_ablation_full_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FULL_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved: {out_csv}  ({len(rows)} rows; "
          f"skipped {n_skip_seed} seeds, {n_skip_cond} (seed,cond) pairs)")

    # ── aggregate per condition ───────────────────────────────────────────────
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r["condition"]
        for col in ("pitch_class_entropy", "n_pitches_used", "pitch_range",
                    "delta_pce", "delta_n_pitches", "gs"):
            agg[c][col].append(r[col])

    ordered = (["argmax"] if "argmax" in conditions else []) + \
              [c for c in conditions if c != "argmax"]

    def label(c):
        return "argmax (baseline)" if c == "argmax" else c

    print("\n" + "=" * 116)
    print(f"  {'condition':<18} | {'PCE':<14} | {'n_pitches':<14} | {'pitch_range':<14} | "
          f"{'delta_pce':<14} | {'delta_n_pitches'}")
    print("  " + "-" * 112)
    for c in ordered:
        a = agg.get(c, {})
        print(f"  {label(c):<18} | {_ms(a.get('pitch_class_entropy', [])):<14} | "
              f"{_ms(a.get('n_pitches_used', [])):<14} | {_ms(a.get('pitch_range', [])):<14} | "
              f"{_ms(a.get('delta_pce', [])):<14} | {_ms(a.get('delta_n_pitches', []))}")
    print("=" * 116)

    # ── GS == 1.0 rate per condition ──────────────────────────────────────────
    print("\n" + "=" * 44)
    print(f"  {'condition':<18} | {'GS=1.0 rate (%)'}")
    print("  " + "-" * 40)
    for c in ordered:
        gvals = [g for g in agg.get(c, {}).get("gs", []) if not math.isnan(g)]
        if gvals:
            rate = 100.0 * sum(1 for g in gvals if g == 1.0) / len(gvals)
            print(f"  {label(c):<18} | {rate:5.1f}%   (n={len(gvals)})")
        else:
            print(f"  {label(c):<18} | N/A")
    print("=" * 44)


if __name__ == "__main__":
    main()
