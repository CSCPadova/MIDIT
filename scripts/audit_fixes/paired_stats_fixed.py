"""
scripts/audit_fixes/paired_stats_fixed.py

THROWAWAY stats (read-only). Paired comparison MIDIT FT vs Anticipatory MT
(time-crop fixed) to update thesis Table tab:anticipatory_direct.

Inner-joins two per-seed CSVs on 'experiment', keeps seeds present in BOTH, and
for CP / GS / PCHE reports: mean+/-std per model, paired t-test, Wilcoxon
signed-rank, Cohen's d_z (paired), and win fractions (direction-aware:
CP,GS higher=better; PCHE lower=better).

MIDIT source must be the SAME frozen CSV behind the current table (e.g.
thesis_freeze_20jul/metrics_midit.csv). Anticipatory = metrics_anticipatory_fixed.csv.

CAVEAT (printed): MIDIT frozen numbers use the _detect_masked_bars crop;
Anticipatory-fixed uses the time-based bars-3-5 crop (per approved design).
The two windowing definitions coincide for clean 4/4 8-bar files and differ
otherwise; MIDIT is not recomputed. GS is reported but flagged not strictly
comparable across paradigms.

Usage:
    python scripts/audit_fixes/paired_stats_fixed.py \
        --midit_csv /nfsd/.../thesis_freeze_20jul/metrics_midit.csv \
        --fixed_csv /nfsd/.../metrics_anticipatory_fixed.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as scipy_stats

METRICS = ("cp", "gs", "pche")
HIGHER_BETTER = {"cp": True, "gs": True, "pche": False}

# OLD table values (from thesis, for the before/after summary)
OLD = {
    "cp":   {"midit": 0.729, "ant": 0.435, "d": 1.16},
    "pche": {"midit": 0.323, "ant": 0.790, "d": 0.57},
}


def load_csv(path: Path) -> Dict[str, Dict[str, float]]:
    """Return {experiment: {metric: float}} (NaN if unparseable/missing)."""
    if not path.exists():
        raise SystemExit(f"ERROR: file not found: {path}")
    out: Dict[str, Dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for r in reader:
            exp = r.get("experiment")
            if exp is None:
                continue
            row = {}
            for m in METRICS:
                try:
                    row[m] = float(r[m])
                except (KeyError, ValueError, TypeError):
                    row[m] = float("nan")
            out[exp] = row
    print(f"  loaded {path}")
    print(f"    header={header}")
    print(f"    rows={len(out)}")
    return out


def paired(midit: Dict, ant: Dict, metric: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Aligned arrays over experiments present in both with non-NaN metric."""
    a, b, keys = [], [], []
    for exp in midit:
        if exp not in ant:
            continue
        x, y = midit[exp][metric], ant[exp][metric]
        if math.isnan(x) or math.isnan(y):
            continue
        a.append(x); b.append(y); keys.append(exp)
    return np.asarray(a, float), np.asarray(b, float), keys


def stats_block(m: str, midit: Dict, ant: Dict) -> None:
    a, b, keys = paired(midit, ant, m)          # a=MIDIT, b=Anticipatory
    n = len(a)
    print(f"\n  [{m.upper()}]  ({'higher' if HIGHER_BETTER[m] else 'lower'} is better)   matched n={n}")
    if n < 2:
        print("    insufficient paired data")
        return
    diff = a - b                                 # MIDIT - Anticipatory
    mean_a, std_a = float(np.mean(a)), float(np.std(a, ddof=1))
    mean_b, std_b = float(np.mean(b)), float(np.std(b, ddof=1))
    print(f"    MIDIT        : {mean_a:.4f} +/- {std_a:.4f}")
    print(f"    Anticipatory : {mean_b:.4f} +/- {std_b:.4f}")

    # paired t-test
    t, p_t = scipy_stats.ttest_rel(a, b)
    # Wilcoxon signed-rank (guard: needs some non-zero diffs)
    try:
        W, p_w = scipy_stats.wilcoxon(a, b)
        w_s = f"W={W:.1f}  p={p_w:.2e}"
    except Exception as e:  # noqa: BLE001
        w_s = f"n/a ({type(e).__name__})"
    # Cohen's d_z (paired)
    sd = float(np.std(diff, ddof=1))
    dz = float(np.mean(diff) / sd) if sd > 0 else float("nan")

    print(f"    paired t     : t={t:.3f}  p={p_t:.2e}")
    print(f"    Wilcoxon     : {w_s}")
    print(f"    Cohen d_z    : {dz:+.3f}  (diff=MIDIT-Ant; sign>0 => MIDIT higher)")

    # direction-aware wins
    if HIGHER_BETTER[m]:
        midit_wins = int(np.sum(a > b)); ant_wins = int(np.sum(b > a))
    else:
        midit_wins = int(np.sum(a < b)); ant_wins = int(np.sum(b < a))
    ties = n - midit_wins - ant_wins
    print(f"    wins         : MIDIT {midit_wins} ({100*midit_wins/n:.1f}%)  "
          f"Ant {ant_wins} ({100*ant_wins/n:.1f}%)  ties {ties}")

    better = "MIDIT" if ((mean_a > mean_b) == HIGHER_BETTER[m]) else "Anticipatory"
    sig = "significant" if p_t < 0.05 else "NOT significant"
    print(f"    -> winner (mean): {better}   [{sig} at p<0.05]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired MIDIT vs Anticipatory-fixed stats.")
    ap.add_argument("--midit_csv", required=True, help="Frozen MIDIT per-seed CSV (unchanged).")
    ap.add_argument("--fixed_csv", required=True, help="metrics_anticipatory_fixed.csv (time-crop).")
    args = ap.parse_args()

    print("[paired_stats_fixed] loading inputs (read-only)")
    midit = load_csv(Path(args.midit_csv))
    ant   = load_csv(Path(args.fixed_csv))

    matched = [e for e in midit if e in ant]
    print(f"\n  inner-join matched experiments: {len(matched)}")

    for m in METRICS:
        stats_block(m, midit, ant)
    print("\n  NOTE: GS remains not strictly comparable across paradigms "
          "(mask-preserve vs re-encode); reported for completeness only.")

    # ── before/after summary ──────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  BEFORE / AFTER  (thesis Table tab:anticipatory_direct)")
    print("=" * 68)
    for m in ("cp", "pche"):
        a, b, _ = paired(midit, ant, m)
        if len(a) < 2:
            continue
        diff = a - b
        sd = float(np.std(diff, ddof=1))
        dz = float(np.mean(diff) / sd) if sd > 0 else float("nan")
        o = OLD[m]
        arrow = "higher=better" if HIGHER_BETTER[m] else "lower=better"
        print(f"\n  {m.upper()} ({arrow})")
        print(f"    OLD : MIDIT {o['midit']:.3f} vs Ant {o['ant']:.3f}   (d={o['d']:.2f})")
        print(f"    NEW : MIDIT {np.mean(a):.3f} vs Ant {np.mean(b):.3f}   (d_z={dz:+.2f})")
    print("=" * 68)

    print("\n  CAVEAT: MIDIT uses the _detect_masked_bars crop; Anticipatory-fixed "
          "uses the time-based bars-3-5 crop. They coincide for clean 4/4 8-bar "
          "files and differ otherwise; MIDIT is not recomputed (unaffected by the "
          "tick-bar artifact). CP higher=better, PCHE lower=better.")


if __name__ == "__main__":
    main()
