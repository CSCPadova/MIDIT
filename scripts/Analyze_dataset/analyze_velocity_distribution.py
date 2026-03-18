"""
Velocity Distribution Analysis -- MIDIT REMI+ V1 Config
========================================================
Samples 300 random MIDI files from data/midi_files/lmd_full/,
tokenizes each with the V1 config, and reports:

  1. Count + % per velocity bin (32 bins)
  2. Flag if top-5 bins cover >70% of all velocity tokens
  3. Cumulative distribution (top-N bins)
  4. Per-file variability (std of bin distributions)
  5. Near-constant-velocity files (top-1 bin >80%)

Run from MIDIT project root:
    python scripts/Analyze_dataset/analyze_velocity_distribution.py
"""

from __future__ import annotations

import random
import statistics
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

DATASET_DIR    = Path("data/midi_files/lmd_full")
LOGS_DIR       = Path("logs")
OUTPUT_FILE    = LOGS_DIR / "velocity_distribution_analysis.txt"
SAMPLE_SIZE    = 300
NUM_WORKERS    = 4
SEED           = 42

TOP5_FLAG_PCT  = 70.0   # flag if top-5 bins > this % of all velocity tokens
NEAR_CONST_PCT = 80.0   # flag file if top-1 bin > this % of its velocity tokens
NUM_VEL_BINS   = 32

# ── Worker (module-level for Windows spawn) ──────────────────────────────────

_tokenizer   = None
_id_to_vel   = None   # {token_id: int(bin_index)}  for Velocity_X tokens


def _init_worker():
    """Initialize tokenizer once per worker process."""
    global _tokenizer, _id_to_vel
    from miditok import REMI, TokenizerConfig
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8},
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_controls=False,
    )
    _tokenizer = REMI(config)
    _id_to_vel = {}
    for tok_name, tok_id in _tokenizer.vocab.items():
        if tok_name.startswith("Velocity_"):
            try:
                _id_to_vel[tok_id] = int(tok_name[9:])
            except ValueError:
                pass


def _analyze_file(midi_path: Path) -> dict | None:
    """
    Tokenize one MIDI file and extract per-file velocity bin counts.
    Returns a dict with 'counts' (Counter bin->n) or None on error.
    """
    try:
        tokens = _tokenizer(str(midi_path))
        # Flatten: may be list-of-sequences or single sequence
        if isinstance(tokens, list):
            all_ids: list[int] = []
            for seq in tokens:
                all_ids.extend(seq.ids)
        else:
            all_ids = tokens.ids

        counts: Counter = Counter()
        for tok_id in all_ids:
            vel = _id_to_vel.get(tok_id)
            if vel is not None:
                counts[vel] += 1

        return {"path": str(midi_path), "counts": counts}

    except Exception as e:
        return {"path": str(midi_path), "error": str(e)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total > 0 else 0.0


def _bar(pct: float, width: int = 28) -> str:
    filled = round(pct / 100.0 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ── Main ─────────────────────────────────────────────────────────────────────

def run_analysis() -> None:
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    t0 = time.time()

    # ── Header ────────────────────────────────────────────────────────────────
    import datetime
    log("=" * 72)
    log("VELOCITY DISTRIBUTION ANALYSIS -- MIDIT REMI+ V1 CONFIG")
    log(f"Dataset  : {DATASET_DIR.resolve()}")
    log(f"Date     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)
    log()

    # ── Sample files ─────────────────────────────────────────────────────────
    all_files = sorted(DATASET_DIR.rglob("*.mid"))
    total_files = len(all_files)
    log(f"  Total files in dataset : {total_files:,}")
    log(f"  Sample size            : {SAMPLE_SIZE}  (seed={SEED})")
    log()

    rng = random.Random(SEED)
    sample = rng.sample(all_files, min(SAMPLE_SIZE, total_files))

    # ── Tokenize ──────────────────────────────────────────────────────────────
    log(f"[1] TOKENIZING {len(sample)} FILES  ({NUM_WORKERS} workers)...")
    log("-" * 72)

    t_tok = time.time()
    with Pool(NUM_WORKERS, initializer=_init_worker) as pool:
        results = pool.map(_analyze_file, sample)
    tok_elapsed = time.time() - t_tok

    good   = [r for r in results if r and "error" not in r]
    failed = [r for r in results if r and "error" in r]

    log(f"  Done in {tok_elapsed:.1f}s  ({tok_elapsed/len(sample):.2f}s/file)")
    log(f"  OK     : {len(good):>3} / {len(sample)}")
    log(f"  Failed : {len(failed):>3} / {len(sample)}")
    log()

    if not good:
        log("ERROR: No files tokenized successfully. Exiting.")
        return

    # ── Aggregate across all files ────────────────────────────────────────────
    agg_counts: Counter = Counter()
    for r in good:
        agg_counts.update(r["counts"])

    total_vel_tokens = sum(agg_counts.values())
    # Velocity token names are Velocity_<raw_midi_value> (e.g. Velocity_63),
    # NOT bin indices 0..31. Use actual quantized values that appear in data.
    all_bins = sorted(agg_counts.keys())

    # ── [2] Per-bin distribution ──────────────────────────────────────────────
    log(f"[2] VELOCITY BIN DISTRIBUTION  ({len(all_bins)} distinct quantized values)")
    log("-" * 72)
    log()
    log(f"  Total Velocity tokens analyzed : {total_vel_tokens:,}")
    log(f"  Across {len(good)} files")
    log()
    log(f"  {'Vel':>4}  {'Count':>10}  {'%':>6}  Bar chart  (vel = raw MIDI value)")
    log(f"  {'-'*4}  {'-'*10}  {'-'*6}  {'-'*30}")

    sorted_bins = sorted(all_bins, key=lambda b: -agg_counts[b])

    for b in all_bins:
        cnt = agg_counts[b]
        pct = _pct(cnt, total_vel_tokens)
        bar = _bar(pct)
        log(f"  {b:>4}  {cnt:>10,}  {pct:>5.2f}%  {bar}")

    log()

    # ── [3] Top-5 flag ────────────────────────────────────────────────────────
    log("[3] TOP-5 BINS COVERAGE")
    log("-" * 72)
    log()

    top5 = sorted_bins[:5]
    top5_count = sum(agg_counts[b] for b in top5)
    top5_pct   = _pct(top5_count, total_vel_tokens)

    log(f"  Top 5 bins by note count: {top5}")
    log(f"  Top 5 combined           : {top5_count:,} notes  ({top5_pct:.2f}%)")

    flag = "*** FLAG: top-5 bins cover more than 70% of velocity tokens ***"
    if top5_pct >= TOP5_FLAG_PCT:
        log(f"  {flag}")
        log(f"  -> Velocity diversity is LOW. Dataset is dominated by a few bins.")
    else:
        log(f"  OK: top-5 bins below {TOP5_FLAG_PCT:.0f}% threshold ({top5_pct:.2f}%)")
    log()

    # ── [4] Cumulative distribution ───────────────────────────────────────────
    log("[4] CUMULATIVE DISTRIBUTION  (sorted by frequency)")
    log("-" * 72)
    log()
    log(f"  {'Top N bins':>10}  {'Cumul %':>8}  Bar chart")
    log(f"  {'-'*10}  {'-'*8}  {'-'*30}")

    cumul = 0
    for rank, b in enumerate(sorted_bins, 1):
        cumul += agg_counts[b]
        cum_pct = _pct(cumul, total_vel_tokens)
        bar = _bar(cum_pct)
        log(f"  {rank:>10}  {cum_pct:>7.2f}%  {bar}")

    log()

    # ── [5] Per-file variability ──────────────────────────────────────────────
    log("[5] PER-FILE VARIABILITY  (std of bin % distributions across files)")
    log("-" * 72)
    log()

    # Build per-file bin fraction vectors (bin -> % of that file's vel tokens)
    # Then for each bin, compute std across files (only files that have vel tokens)
    per_file_fracs: dict[int, list[float]] = {b: [] for b in all_bins}
    files_with_vel = 0

    for r in good:
        file_total = sum(r["counts"].values())
        if file_total == 0:
            continue
        files_with_vel += 1
        for b in all_bins:
            per_file_fracs[b].append(_pct(r["counts"][b], file_total))

    log(f"  Files with at least one Velocity token : {files_with_vel} / {len(good)}")
    log()

    if files_with_vel >= 2:
        bin_stds = {b: statistics.stdev(per_file_fracs[b]) for b in all_bins
                    if len(per_file_fracs[b]) >= 2}
        mean_std = statistics.mean(bin_stds.values())
        max_std_bin = max(bin_stds, key=bin_stds.get)
        min_std_bin = min(bin_stds, key=bin_stds.get)

        log(f"  Mean std across all bins  : {mean_std:.2f}%")
        log(f"  Most variable bin         : bin {max_std_bin}  (std={bin_stds[max_std_bin]:.2f}%)")
        log(f"  Least variable bin        : bin {min_std_bin}  (std={bin_stds[min_std_bin]:.2f}%)")
        log()
        log(f"  {'Bin':>4}  {'Mean%':>7}  {'Std':>7}  Interpretation")
        log(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*30}")

        for b in all_bins:
            mean_f = statistics.mean(per_file_fracs[b]) if per_file_fracs[b] else 0.0
            std_f  = bin_stds.get(b, 0.0)
            note   = "high variability" if std_f > 10.0 else ("low variability" if std_f < 3.0 else "")
            log(f"  {b:>4}  {mean_f:>6.2f}%  {std_f:>6.2f}%  {note}")

        log()
        if mean_std < 5.0:
            log("  *** FLAG: Mean per-bin std < 5% -- files have VERY SIMILAR velocity")
            log("      distributions. Possible quality issue (MIDI quantization artifact).")
        elif mean_std < 8.0:
            log("  NOTE: Moderate variability across files. Acceptable.")
        else:
            log("  OK: High variability -- files have diverse velocity patterns.")
    else:
        log("  Not enough files with velocity tokens to compute std.")
    log()

    # ── [6] Near-constant-velocity files ─────────────────────────────────────
    log("[6] NEAR-CONSTANT VELOCITY FILES  (top-1 bin > 80% of file's velocity tokens)")
    log("-" * 72)
    log()

    near_const: list[dict] = []
    for r in good:
        file_total = sum(r["counts"].values())
        if file_total == 0:
            continue
        top_bin   = max(r["counts"], key=r["counts"].get)
        top_count = r["counts"][top_bin]
        top_pct   = _pct(top_count, file_total)
        if top_pct >= NEAR_CONST_PCT:
            near_const.append({
                "path": r["path"],
                "top_bin": top_bin,
                "top_pct": top_pct,
                "total_vel": file_total,
            })

    near_const.sort(key=lambda x: -x["top_pct"])

    log(f"  Threshold : top-1 bin > {NEAR_CONST_PCT:.0f}% of file's velocity tokens")
    log(f"  Found     : {len(near_const)} / {files_with_vel} files  "
        f"({_pct(len(near_const), files_with_vel):.1f}%)")
    log()

    if near_const:
        pct_all = _pct(len(near_const), len(good))
        if pct_all >= 10.0:
            log(f"  *** FLAG: {pct_all:.1f}% of files have near-constant velocity "
                f"(>= {NEAR_CONST_PCT:.0f}% in one bin) ***")
        log()
        log(f"  {'Top-1 bin':>10}  {'Top-1 %':>8}  {'Vel tokens':>10}  File")
        log(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*40}")
        for entry in near_const[:20]:
            fname = Path(entry["path"]).name
            log(f"  {entry['top_bin']:>10}  {entry['top_pct']:>7.1f}%  "
                f"{entry['total_vel']:>10,}  {fname}")
        if len(near_const) > 20:
            log(f"  ... and {len(near_const) - 20} more files")
    else:
        log(f"  OK: No files with near-constant velocity distribution found.")
    log()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0

    log("=" * 72)
    log("SUMMARY")
    log("=" * 72)

    top1_bin = sorted_bins[0]
    top1_pct = _pct(agg_counts[top1_bin], total_vel_tokens)
    log(f"  Files analyzed              : {len(good):>8,}")
    log(f"  Total Velocity tokens       : {total_vel_tokens:>8,}")
    log(f"  Avg vel tokens/file         : {total_vel_tokens/len(good):>8,.0f}")
    log()
    log(f"  Most common bin             : bin {top1_bin}  ({top1_pct:.2f}%)")
    log(f"  Top-5 bins coverage         : {top5_pct:.2f}%  "
        f"{'*** FLAGGED ***' if top5_pct >= TOP5_FLAG_PCT else 'OK'}")
    log(f"  Near-const-vel files        : {len(near_const)} / {len(good)}"
        f"  ({_pct(len(near_const), len(good)):.1f}%)")
    log()

    # Preprocessing implication
    const_vel_111_files = sum(
        1 for r in good
        if sum(r["counts"].values()) > 0
        and max(r["counts"], key=r["counts"].get) == 111
        and _pct(r["counts"].get(111, 0), sum(r["counts"].values())) >= 80.0
    )
    log(f"  Near-const vel=111 files    : {const_vel_111_files} / {len(good)}"
        f"  ({_pct(const_vel_111_files, len(good)):.1f}%)")
    log(f"  (vel=111 is the raw MIDI value; maps to bin based on quantization)")
    log()

    log(f"  Analysis time  : {total_elapsed:.1f}s")
    log(f"  Report saved   : {OUTPUT_FILE.resolve()}")
    log("=" * 72)

    # ── Write report ──────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nReport written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_analysis()
