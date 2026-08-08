"""
scripts/audit_fixes/recompute_anticipatory_metrics_fixed.py

THROWAWAY audit fix (read-only inputs). Recomputes CP/GS/PCHE for the
Anticipatory MT baseline with a TIME-BASED crop, fixing the tpq mismatch AND
the token/tick bar-index divergence.

WHY time-based (not bar-index):
  The original CP/GS/PCHE crop the recon with a tick window derived from
  original.mid (tpq=16) applied verbatim to Anticipatory's recon (tpq=50) ->
  misaligned. A bar-INDEX match is ALSO invalid, because _detect_masked_bars
  returns tick-space (time-signature grid) bars that do NOT equal the REMI
  token-space masked bars [3,4,5] for most files (e.g. seed_79 -> [5,6,7],
  seed_500 -> 17 scattered indices). Those indices are artifacts of a fixed
  4/4-ish grid over non-8-bar / TS-changing originals and cannot be
  transplanted onto a different-tpq reconstruction.

FIX: crop by the ABSOLUTE TIME span the generator actually told Anticipatory to
  infill: [t1, t2] = [mask_bars[0]*bar_sec, (mask_bars[-1]+1)*bar_sec], with
  bar_sec = 240/qpm_orig (== generator's (tempo/1e6)*4, 4/4 assumed, first
  tempo only; see generate_anticipatory_samples.py:110-121). Convert the SAME
  seconds to EACH file's own ticks (orig: tpq_o/qpm_o; recon: tpq_r/qpm_r).
  This is immune to both the tpq mismatch and the token/tick bar divergence.

METHODOLOGICAL NOTE (disclosure): this crops Anticipatory and its reference
  original to fixed bars-3-5 BY TIME (the generator's constant-4/4 convention),
  whereas MIDIT's own metrics crop to the _detect_masked_bars region. The two
  windowing definitions coincide for clean 4/4 8-bar files and differ for
  odd-TS / non-8-bar files; MIDIT's numbers are unaffected by the tick-bar
  artifact (same grid on both sides) and are NOT recomputed here.

Scope: CP/GS/PCHE only. muspy S-I / JS anticipatory fix is a separate task.

Reuses primitives UNCHANGED from compute_infilling_metrics.py:
  _load, _notes_in_range, _chroma, _cosine, _onset_grid, _jaccard,
  _pitch_histogram, _entropy  (+ _bar_boundaries ONLY to reproduce the OLD
  buggy number for side-by-side comparison).

Inputs (all read-only):
  --recon_dir   samples_anticipatory_1000  (seed_N/reconstructed.mid)
  --orig_dir    samples_1000               (seed_N/original.mid + masked.mid)
  --manifest    scripts/audit_fixes/sample_manifest.csv (restrict to 973 included)

Modes:
  sanity (default): --sanity_seeds ... -> seconds cross-check, note counts
    (old vs new), CP/GS/PCHE (old vs new), end-tick guard, non-4/4 disclosure.
  full: --run_full -> write metrics_anticipatory_fixed.csv over all included
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
GRID_RES = 16                 # 16th-note grid, same as compute_GS default
MASK_BARS = [3, 4, 5]         # generator's fixed convention (SLURM --mask_bars 3 4 5)

Window = Tuple[int, int]      # (t_start_tick, t_end_tick)


# ── tempo / time-signature / duration helpers ─────────────────────────────────

def first_qpm(score) -> float:
    """Quarter-notes per minute of the first tempo (default 120 if absent).

    Matches the generator's read_first_tempo semantics: first tempo, else the
    120 BPM default (generate_anticipatory_samples.py:33,67-73).
    """
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


def end_tick(score) -> int:
    return max((n.time + n.duration for t in score.tracks for n in t.notes), default=0)


def sec_to_ticks(sec: float, tpq: int, qpm: float) -> int:
    """Absolute seconds -> ticks on a file's own grid.

    seconds_per_tick = 60 / (qpm * tpq)  ->  ticks = sec * qpm * tpq / 60.
    """
    return int(round(sec * qpm * tpq / 60.0))


def time_windows(mask_bars: List[int], bar_sec: float, tpq: int, qpm: float) -> List[Window]:
    """Per-masked-bar (t_start, t_end) tick tuples on this file's own grid.

    Bar k spans absolute seconds [k*bar_sec, (k+1)*bar_sec]; converted to this
    file's ticks. bar_sec is derived from the ORIGINAL's tempo (generator
    convention), so all files share the same absolute time span.
    """
    return [
        (sec_to_ticks(k * bar_sec, tpq, qpm), sec_to_ticks((k + 1) * bar_sec, tpq, qpm))
        for k in mask_bars
    ]


# ── OLD (buggy) window reproduction: _detect_masked_bars tick tuples ──────────

def old_masked_windows(orig, masked) -> List[Window]:
    """Reproduce the frozen pipeline's crop: _detect_masked_bars tick tuples
    (original grid), later applied to BOTH orig and recon (the bug).

    Identical criterion to compute_muspy_metrics._detect_masked_bars.
    """
    bnd = _bar_boundaries(orig, max_bars=MAX_BARS)
    out: List[Window] = []
    for b in range(len(bnd) - 1):
        t0, t1 = bnd[b], bnd[b + 1]
        if len(_notes_in_range(orig, t0, t1)) > 0 and len(_notes_in_range(masked, t0, t1)) == 0:
            out.append((t0, t1))
    return out


# ── CP/GS/PCHE over explicit per-file windows ─────────────────────────────────

def compute_triplet(
    orig, recon,
    orig_windows: List[Window], recon_windows: List[Window],
    grid_o: int, grid_r: int,
) -> Tuple[float, float, float]:
    """CP, GS, PCHE pairing orig_windows[i] with recon_windows[i].

    orig_windows and recon_windows must be the same length (same masked bars,
    each converted to its own file's ticks).
    """
    cps: List[float] = []
    gss: List[float] = []
    pool_o: List = []
    pool_i: List = []
    for (o0, o1), (r0, r1) in zip(orig_windows, recon_windows):
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


def _count_notes(score, windows: List[Window]) -> int:
    return sum(len(_notes_in_range(score, a, b)) for a, b in windows)


def _grid_ticks(tpq: int) -> int:
    return max(1, int(round(tpq * 4 / GRID_RES)))


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


# ── per-seed compute ──────────────────────────────────────────────────────────

def process_seed(name: str, orig_dir: Path, recon_dir: Path) -> dict:
    orig_p   = orig_dir / name / "original.mid"
    masked_p = orig_dir / name / "masked.mid"
    recon_p  = recon_dir / name / "reconstructed.mid"
    if not (orig_p.exists() and masked_p.exists() and recon_p.exists()):
        return {"seed": name, "status": "missing_file"}

    orig   = _load(orig_p)
    masked = _load(masked_p)
    recon  = _load(recon_p)

    tpq_o, tpq_r = orig.ticks_per_quarter, recon.ticks_per_quarter
    qpm_o, qpm_r = first_qpm(orig), first_qpm(recon)
    grid_o, grid_r = _grid_ticks(tpq_o), _grid_ticks(tpq_r)

    # Generator's exact span (bar_sec from ORIGINAL tempo, 4/4 assumed)
    bar_sec = 240.0 / qpm_o
    t1, t2 = MASK_BARS[0] * bar_sec, (MASK_BARS[-1] + 1) * bar_sec

    # NEW: time-based windows, each file on its own grid
    ow_new = time_windows(MASK_BARS, bar_sec, tpq_o, qpm_o)
    rw_new = time_windows(MASK_BARS, bar_sec, tpq_r, qpm_r)
    cp_new, gs_new, pche_new = compute_triplet(orig, recon, ow_new, rw_new, grid_o, grid_r)

    # OLD: reproduce the frozen bug (orig-grid detected tuples on BOTH sides)
    old_win = old_masked_windows(orig, masked)
    if old_win:
        cp_old, gs_old, pche_old = compute_triplet(orig, recon, old_win, old_win, grid_o, grid_o)
    else:
        cp_old = gs_old = pche_old = float("nan")

    et_o, et_r = end_tick(orig), end_tick(recon)
    bars_in_orig  = sum(1 for (_, e) in ow_new if e <= et_o)
    bars_in_recon = sum(1 for (_, e) in rw_new if e <= et_r)

    return {
        "seed": name, "status": "ok",
        "tpq_o": tpq_o, "tpq_r": tpq_r, "qpm_o": qpm_o, "qpm_r": qpm_r,
        "bar_sec": bar_sec, "t1": t1, "t2": t2,
        "ow_new": ow_new, "rw_new": rw_new, "old_win": old_win,
        "cp_new": cp_new, "gs_new": gs_new, "pche_new": pche_new,
        "cp_old": cp_old, "gs_old": gs_old, "pche_old": pche_old,
        "n_orig_new":  _count_notes(orig,  ow_new),
        "n_recon_new": _count_notes(recon, rw_new),
        "n_recon_old": _count_notes(recon, old_win),   # orig-grid window on recon (bug)
        "et_o": et_o, "et_r": et_r,
        "bars_in_orig": bars_in_orig, "bars_in_recon": bars_in_recon,
        "non44_orig": is_non_44(orig),
    }


# ── sanity report ─────────────────────────────────────────────────────────────

def sanity(args) -> None:
    orig_dir  = Path(args.orig_dir)
    recon_dir = Path(args.recon_dir)
    seeds = [f"seed_{s}" for s in args.sanity_seeds]

    print("\n[recompute_anticipatory_metrics_fixed] SANITY (time-based crop)")
    print(f"  orig_dir  : {orig_dir}")
    print(f"  recon_dir : {recon_dir}")
    print(f"  mask_bars : {MASK_BARS} (fixed; generator convention)")
    print(f"  seeds     : {seeds}\n")

    for name in seeds:
        rec = process_seed(name, orig_dir, recon_dir)
        print("=" * 80)
        print(f"  {name}")
        print("=" * 80)
        if rec["status"] != "ok":
            print(f"  SKIP: {rec['status']}")
            continue

        spt_o = 60.0 / (rec["qpm_o"] * rec["tpq_o"])
        spt_r = 60.0 / (rec["qpm_r"] * rec["tpq_r"])
        ow, rw = rec["ow_new"], rec["rw_new"]
        o_start, o_end = ow[0][0], ow[-1][1]
        r_start, r_end = rw[0][0], rw[-1][1]

        print(f"  tpq_o={rec['tpq_o']} qpm_o={rec['qpm_o']:.2f}   "
              f"tpq_r={rec['tpq_r']} qpm_r={rec['qpm_r']:.2f}   "
              f"non4/4_orig={rec['non44_orig']}")
        print("  [1] seconds cross-check (H1 — should all ~match [t1,t2]):")
        print(f"      generator    : [{rec['t1']:7.3f}, {rec['t2']:7.3f}] s")
        print(f"      orig  window : [{o_start*spt_o:7.3f}, {o_end*spt_o:7.3f}] s   ticks[{o_start},{o_end}]")
        print(f"      recon window : [{r_start*spt_r:7.3f}, {r_end*spt_r:7.3f}] s   ticks[{r_start},{r_end}]")
        print("  [2] note counts:")
        print(f"      orig  NEW window : {rec['n_orig_new']}")
        print(f"      recon NEW window : {rec['n_recon_new']}   <- fixed (time-based)")
        print(f"      recon OLD window : {rec['n_recon_old']}   <- misaligned (orig-grid on recon)")
        print("  [3] CP / GS / PCHE  (old -> new):")
        print(f"      CP   : {rec['cp_old']:.4f} -> {rec['cp_new']:.4f}")
        print(f"      GS   : {rec['gs_old']:.4f} -> {rec['gs_new']:.4f}")
        print(f"      PCHE : {rec['pche_old']:.4f} -> {rec['pche_new']:.4f}")
        print("  [5] end-tick guard:")
        print(f"      orig  end_tick={rec['et_o']}  masked bars fully inside: {rec['bars_in_orig']}/{len(ow)}")
        print(f"      recon end_tick={rec['et_r']}  masked bars fully inside: {rec['bars_in_recon']}/{len(rw)}")
        if rec["bars_in_recon"] < len(rw) or rec["bars_in_orig"] < len(ow):
            print("      WARNING: t2 exceeds a file's own end_tick (short file)")
        print()

    # [4] non-4/4 disclosure over ALL included originals
    inc = included_seeds(Path(args.manifest))
    print("=" * 80)
    print(f"  [4] non-4/4 scan over {len(inc)} included originals (H2 disclosure)")
    print("=" * 80)
    n_non44 = n_missing = n_scanned = 0
    for i, nm in enumerate(inc):
        p = orig_dir / nm / "original.mid"
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
    frac = 100.0 * n_non44 / max(1, n_scanned)
    print(f"  scanned={n_scanned}  non-4/4={n_non44}  missing/err={n_missing}  ({frac:.1f}% non-4/4)")

    print("\n[DISCLOSURE] Anticipatory is cropped to fixed bars-3-5 BY TIME "
          "(generator's constant-4/4 convention); MIDIT uses its _detect_masked_bars "
          "region. They coincide for clean 4/4 8-bar files and differ otherwise; "
          "MIDIT's numbers are unaffected by the tick-bar artifact and are not recomputed.")
    print("[SANITY DONE] Review before running --run_full.")


# ── full CSV pass (do NOT run until sanity approved) ──────────────────────────

def _agg(values: List[float]) -> Tuple[float, float, int]:
    """(mean, std ddof=1, n) over non-NaN values."""
    v = [x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else float("nan")
    return mean, std, n


def _load_old_csv(path: Path, keep: set) -> dict:
    """Load frozen metrics_anticipatory.csv -> {experiment: {cp,gs,pche}} for kept seeds."""
    out: dict = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            exp = r.get("experiment")
            if exp in keep:
                def _f(k):
                    try:
                        return float(r[k])
                    except (KeyError, ValueError, TypeError):
                        return float("nan")
                out[exp] = {"cp": _f("cp"), "gs": _f("gs"), "pche": _f("pche")}
    return out


def run_full(args) -> None:
    orig_dir  = Path(args.orig_dir)
    recon_dir = Path(args.recon_dir)
    out_csv   = Path(args.output_csv)
    inc = included_seeds(Path(args.manifest))

    print(f"\n[recompute_anticipatory_metrics_fixed] FULL over {len(inc)} included seeds (time-based crop)")
    rows: List[dict] = []
    acc = {k: [] for k in ("cp_new", "gs_new", "pche_new", "cp_old", "gs_old", "pche_old")}
    n_ok = n_skip = 0
    for i, name in enumerate(inc):
        rec = process_seed(name, orig_dir, recon_dir)
        if rec.get("status") != "ok":
            n_skip += 1
            continue
        rows.append({"experiment": name, "cp": rec["cp_new"],
                     "gs": rec["gs_new"], "pche": rec["pche_new"]})
        for k in acc:
            acc[k].append(rec[k])
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

    # ── aggregate comparison ──────────────────────────────────────────────────
    done_seeds = {r["experiment"] for r in rows}
    frozen = _load_old_csv(Path(args.old_csv), done_seeds) if args.old_csv else {}

    print("\n" + "=" * 72)
    print(f"  AGGREGATE  (mean +/- std, ddof=1)  over n_ok={n_ok} seeds")
    print("=" * 72)
    print(f"  {'metric':<6} {'NEW (time-crop)':>22} {'OLD (in-script bug repro)':>28}")
    print("  " + "-" * 62)
    for m in ("cp", "gs", "pche"):
        mn, sn, nn = _agg(acc[f"{m}_new"])
        mo, so, no = _agg(acc[f"{m}_old"])
        print(f"  {m.upper():<6} {mn:8.4f} +/- {sn:7.4f} (n={nn:>3}) "
              f"   {mo:8.4f} +/- {so:7.4f} (n={no:>3})")

    if frozen:
        print("\n  OLD (frozen metrics_anticipatory.csv, restricted to the same seeds):")
        for m in ("cp", "gs", "pche"):
            mf, sf, nf = _agg([frozen[s][m] for s in frozen])
            print(f"    {m.upper():<6} {mf:8.4f} +/- {sf:7.4f} (n={nf:>3})")
    else:
        print("\n  (pass --old_csv <frozen metrics_anticipatory.csv> to also show the "
              "frozen aggregate for the same seeds)")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute Anticipatory CP/GS/PCHE with time-based crop.")
    ap.add_argument("--recon_dir", required=True, help="Dir with seed_N/reconstructed.mid (anticipatory).")
    ap.add_argument("--orig_dir", default="/nfsd/tesi/zanettisim/MIDIT/outputs/samples_1000",
                    help="Dir with seed_N/original.mid + masked.mid.")
    ap.add_argument("--manifest", default=str(Path(__file__).parent / "sample_manifest.csv"))
    ap.add_argument("--sanity_seeds", type=int, nargs="+",
                    default=[5, 79, 216, 500, 900, 100, 300, 700])
    ap.add_argument("--run_full", action="store_true", help="Run full pass and write CSV.")
    ap.add_argument("--output_csv", default="outputs/metrics_anticipatory_fixed.csv")
    ap.add_argument("--old_csv", default=None,
                    help="Frozen metrics_anticipatory.csv; if given, its aggregate over "
                         "the same seeds is printed alongside NEW/OLD for comparison.")
    args = ap.parse_args()

    if args.run_full:
        run_full(args)
    else:
        sanity(args)


if __name__ == "__main__":
    main()
