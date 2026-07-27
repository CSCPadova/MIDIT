"""
scripts/audit_fixes/recompute_anticipatory_metrics_fixed.py

THROWAWAY audit fix (read-only inputs). Recomputes CP/GS/PCHE for the
Anticipatory MT baseline with a PER-FILE-CONSISTENT crop, fixing the tpq
mismatch: the original CP/GS/PCHE crop the recon with a tick window derived
from original.mid (tpq=16) applied verbatim to Anticipatory's recon (tpq=50),
misaligning the window. Here each file's masked-bar window is computed on its
OWN bar grid (its own tpq/time-signature), at the SAME detected masked bar
INDICES (matching MIDIT's own crop protocol).

Scope: CP/GS/PCHE only. muspy S-I / JS anticipatory fix is a separate task.

Reuses primitives UNCHANGED from compute_infilling_metrics.py:
  _load, _bar_boundaries, _notes_in_range, _chroma, _cosine,
  _onset_grid, _jaccard, _pitch_histogram, _entropy.
New here: masked-bar-INDEX detection (returns indices, not tick tuples) and a
per-file windowing loop (compute_CP/GS/PCHE hardcode a single shared bar_range,
so they cannot be reused for a two-grid crop).

Inputs (all read-only):
  --recon_dir   samples_anticipatory_1000  (seed_N/reconstructed.mid)
  --orig_dir    samples_1000               (seed_N/original.mid + masked.mid)
  --manifest    scripts/audit_fixes/sample_manifest.csv (restrict to 973 included)

Modes:
  sanity (default): --sanity_seeds N N ...  -> print seconds cross-check, note
    counts (old-misaligned vs new-fixed), old-vs-new CP/GS/PCHE, and a non-4/4
    count over all included originals. Does NOT write a CSV.
  full: --run_full  -> write metrics_anticipatory_fixed.csv over all included
    seeds. (Do not run until sanity is reviewed.)

Never writes into outputs/thesis_freeze_20jul/; never edits pipeline files.
Pure CPU (symusic + numpy); no model, no GPU.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_EVAL = _ROOT / "scripts" / "Evaluation_Infilling"
sys.path.insert(0, str(_EVAL))

from compute_infilling_metrics import (          # noqa: E402
    _load, _bar_boundaries, _notes_in_range,
    _chroma, _cosine, _onset_grid, _jaccard,
    _pitch_histogram, _entropy,
)

MAX_BARS = 64
GRID_RES = 16   # 16th-note grid, same as compute_GS default


# ── masked-bar INDEX detection (parallel to _detect_masked_bars, but indices) ─

def detect_masked_bar_indices(orig, masked) -> List[int]:
    """Bar INDICES masked in original vs masked (orig has notes, masked has none).

    Same criterion as compute_muspy_metrics._detect_masked_bars, but returns
    0-based bar indices on the ORIGINAL's grid instead of tick tuples.
    """
    boundaries = _bar_boundaries(orig, max_bars=MAX_BARS)
    idx: List[int] = []
    for b in range(len(boundaries) - 1):
        t0, t1 = boundaries[b], boundaries[b + 1]
        if len(_notes_in_range(orig, t0, t1)) > 0 and len(_notes_in_range(masked, t0, t1)) == 0:
            idx.append(b)
    return idx


# ── tempo / time-signature helpers ────────────────────────────────────────────

def first_qpm(score) -> float:
    """Quarter-notes per minute of the first tempo (default 120 if absent)."""
    try:
        tempos = score.tempos
        if tempos is not None and len(tempos) > 0:
            t0 = tempos[0]
            q = getattr(t0, "qpm", None)
            if q:
                return float(q)
            mspq = getattr(t0, "mspq", None)   # microseconds per quarter
            if mspq:
                return 60_000_000.0 / float(mspq)
    except Exception:
        pass
    return 120.0


def is_non_44(score) -> bool:
    """True if any time signature is not 4/4 (empty TS -> treated as 4/4)."""
    try:
        for ts in score.time_signatures:
            if (ts.numerator, ts.denominator) != (4, 4):
                return True
    except Exception:
        pass
    return False


# ── CP/GS/PCHE with per-file windows ──────────────────────────────────────────

def compute_triplet(
    orig, recon, indices: List[int],
    bo: List[int], br: List[int], tpq_o: int, tpq_r: int,
) -> Tuple[float, float, float]:
    """CP, GS, PCHE over `indices`, cropping orig on grid `bo` and recon on `br`.

    Pass br=bo, tpq_r=tpq_o to reproduce the OLD (shared-grid) buggy behaviour.
    """
    grid_o = max(1, int(round(tpq_o * 4 / GRID_RES)))
    grid_r = max(1, int(round(tpq_r * 4 / GRID_RES)))
    cps: List[float] = []
    gss: List[float] = []
    pool_o: List = []
    pool_i: List = []
    for b in indices:
        o0, o1 = bo[b], bo[b + 1]
        r0, r1 = br[b], br[b + 1]
        no = _notes_in_range(orig, o0, o1)
        ni = _notes_in_range(recon, r0, r1)
        cps.append(_cosine(_chroma(no, o0, o1), _chroma(ni, r0, r1)))
        gss.append(_jaccard(_onset_grid(no, o0, o1, grid_o), _onset_grid(ni, r0, r1, grid_r)))
        pool_o += no
        pool_i += ni
    cp = float(np.mean(cps)) if cps else 0.0
    gs = float(np.mean(gss)) if gss else 0.0
    pche = abs(_entropy(_pitch_histogram(pool_o)) - _entropy(_pitch_histogram(pool_i)))
    return cp, gs, pche


def _recon_note_count(recon, indices: List[int], boundaries: List[int]) -> int:
    return sum(len(_notes_in_range(recon, boundaries[b], boundaries[b + 1])) for b in indices)


# ── manifest ──────────────────────────────────────────────────────────────────

def included_seeds(manifest: Path) -> List[str]:
    if not manifest.exists():
        raise SystemExit(f"ERROR: manifest not found: {manifest}")
    out = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] == "included":
                out.append(r["seed"])
    return out


# ── per-seed compute (returns a record or None) ───────────────────────────────

def process_seed(name: str, orig_dir: Path, recon_dir: Path) -> Optional[dict]:
    orig_p   = orig_dir / name / "original.mid"
    masked_p = orig_dir / name / "masked.mid"
    recon_p  = recon_dir / name / "reconstructed.mid"
    if not orig_p.exists() or not masked_p.exists() or not recon_p.exists():
        return {"seed": name, "status": "missing_file"}

    orig   = _load(orig_p)
    masked = _load(masked_p)
    recon  = _load(recon_p)

    indices = detect_masked_bar_indices(orig, masked)
    if not indices:
        return {"seed": name, "status": "no_masked_bars"}

    tpq_o = orig.ticks_per_quarter
    tpq_r = recon.ticks_per_quarter
    bo = _bar_boundaries(orig,  max_bars=MAX_BARS)
    br = _bar_boundaries(recon, max_bars=MAX_BARS)

    # NEW (per-file) and OLD (shared orig grid on recon = the bug)
    cp_new, gs_new, pche_new = compute_triplet(orig, recon, indices, bo, br, tpq_o, tpq_r)
    cp_old, gs_old, pche_old = compute_triplet(orig, recon, indices, bo, bo, tpq_o, tpq_o)

    return {
        "seed": name, "status": "ok", "indices": indices,
        "tpq_o": tpq_o, "tpq_r": tpq_r,
        "cp_new": cp_new, "gs_new": gs_new, "pche_new": pche_new,
        "cp_old": cp_old, "gs_old": gs_old, "pche_old": pche_old,
        "n_orig":       _recon_note_count(orig,  indices, bo),
        "n_recon_old":  _recon_note_count(recon, indices, bo),   # orig-grid window (bug)
        "n_recon_new":  _recon_note_count(recon, indices, br),   # recon-grid window (fixed)
        "orig": orig, "recon": recon, "bo": bo, "br": br,
    }


# ── sanity report ─────────────────────────────────────────────────────────────

def sanity(args) -> None:
    orig_dir  = Path(args.orig_dir)
    recon_dir = Path(args.recon_dir)
    seeds = [f"seed_{s}" for s in args.sanity_seeds]

    print(f"\n[recompute_anticipatory_metrics_fixed] SANITY")
    print(f"  orig_dir  : {orig_dir}")
    print(f"  recon_dir : {recon_dir}")
    print(f"  seeds     : {seeds}\n")

    for name in seeds:
        rec = process_seed(name, orig_dir, recon_dir)
        print("=" * 78)
        print(f"  {name}")
        print("=" * 78)
        if rec["status"] != "ok":
            print(f"  SKIP: {rec['status']}")
            continue

        idx = rec["indices"]
        first, last = min(idx), max(idx)
        bo, br = rec["bo"], rec["br"]
        qpm_o = first_qpm(rec["orig"])
        qpm_r = first_qpm(rec["recon"])
        spt_o = 60.0 / (qpm_o * rec["tpq_o"])   # seconds per tick
        spt_r = 60.0 / (qpm_r * rec["tpq_r"])
        bar_sec = 240.0 / qpm_o                 # 4 beats * (60/qpm)
        gen_t1, gen_t2 = 3 * bar_sec, 6 * bar_sec   # generator fixed span [3,6)

        print(f"  detected masked bar indices: {idx}   tpq_o={rec['tpq_o']} tpq_r={rec['tpq_r']}")
        print(f"  qpm_o={qpm_o:.2f}  qpm_r={qpm_r:.2f}")
        print("  [1] seconds cross-check (H1):")
        print(f"      orig  window : [{bo[first]*spt_o:7.3f}, {bo[last+1]*spt_o:7.3f}] s")
        print(f"      recon window : [{br[first]*spt_r:7.3f}, {br[last+1]*spt_r:7.3f}] s")
        print(f"      generator    : [{gen_t1:7.3f}, {gen_t2:7.3f}] s   (bars 3-6, orig tempo)")
        print("  [2] note counts (H4):")
        print(f"      orig (masked bars)          : {rec['n_orig']}")
        print(f"      recon OLD (orig-grid window): {rec['n_recon_old']}  <- misaligned")
        print(f"      recon NEW (recon-grid window): {rec['n_recon_new']}  <- fixed")
        print("  [3] CP / GS / PCHE  (old -> new):")
        print(f"      CP   : {rec['cp_old']:.4f} -> {rec['cp_new']:.4f}")
        print(f"      GS   : {rec['gs_old']:.4f} -> {rec['gs_new']:.4f}")
        print(f"      PCHE : {rec['pche_old']:.4f} -> {rec['pche_new']:.4f}")
        print()

    # [4] non-4/4 count over ALL included originals (H2 disclosure prep)
    inc = included_seeds(Path(args.manifest))
    print("=" * 78)
    print(f"  [4] non-4/4 scan over {len(inc)} included originals (H2)")
    print("=" * 78)
    n_non44 = n_missing = n_scanned = 0
    for i, name in enumerate(inc):
        p = orig_dir / name / "original.mid"
        if not p.exists():
            n_missing += 1
            continue
        try:
            if is_non_44(_load(p)):
                n_non44 += 1
            n_scanned += 1
        except Exception:
            n_missing += 1
        if (i + 1) % 200 == 0:
            print(f"    ...{i+1}/{len(inc)} scanned")
    print(f"  scanned={n_scanned}  non-4/4={n_non44}  missing/err={n_missing}")
    print(f"  non-4/4 fraction: {100*n_non44/max(1,n_scanned):.1f}%")
    print("\n[SANITY DONE] Review before running --run_full.")


# ── full CSV pass (do NOT run until sanity approved) ──────────────────────────

def run_full(args) -> None:
    orig_dir  = Path(args.orig_dir)
    recon_dir = Path(args.recon_dir)
    out_csv   = Path(args.output_csv)
    inc = included_seeds(Path(args.manifest))

    print(f"\n[recompute_anticipatory_metrics_fixed] FULL over {len(inc)} included seeds")
    rows: List[dict] = []
    n_ok = n_skip = 0
    for i, name in enumerate(inc):
        rec = process_seed(name, orig_dir, recon_dir)
        if rec.get("status") != "ok":
            n_skip += 1
            continue
        rows.append({"experiment": name, "cp": rec["cp_new"],
                     "gs": rec["gs_new"], "pche": rec["pche_new"]})
        n_ok += 1
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(inc)}  ok={n_ok} skip={n_skip}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "cp", "gs", "pche"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved: {out_csv}  (ok={n_ok}, skipped={n_skip})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute Anticipatory CP/GS/PCHE with per-file crop.")
    ap.add_argument("--recon_dir", required=True, help="Dir with seed_N/reconstructed.mid (anticipatory).")
    ap.add_argument("--orig_dir", default="/nfsd/tesi/zanettisim/MIDIT/outputs/samples_1000",
                    help="Dir with seed_N/original.mid + masked.mid.")
    ap.add_argument("--manifest", default=str(Path(__file__).parent / "sample_manifest.csv"))
    ap.add_argument("--sanity_seeds", type=int, nargs="+", default=[5, 79, 216, 500, 900])
    ap.add_argument("--run_full", action="store_true", help="Run full pass and write CSV.")
    ap.add_argument("--output_csv", default="outputs/metrics_anticipatory_fixed.csv")
    args = ap.parse_args()

    if args.run_full:
        run_full(args)
    else:
        sanity(args)


if __name__ == "__main__":
    main()
