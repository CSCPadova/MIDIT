"""
Quality Filter Simulation -- MIDIT Lakh MIDI Dataset
=====================================================
Samples 500 random MIDI files from data/midi_files/lmd_full/
using miditoolkit (no tokenizer -- fast raw MIDI reading).

ANALYSIS A -- Bar count distribution:
  Estimates bar count as duration_beats / 4 (4/4 approximation).
  Reports distribution and % of files producing 0 chunks (< 8 bars).

ANALYSIS B -- Combined quality filter simulation:
  Filter 1: Constant velocity (all notes same velocity)
  Filter 2: Near-constant velocity (top-1 raw vel > 90% of notes)
  Filter 3: Duration < 15 seconds
  Filter 4: Note count < 50
  Filter 5: Bar count < 8 (produces 0 chunks with bars_per_chunk=8)

  Reports per-filter counts, union, and survival rate.

Run from MIDIT project root:
    python scripts/Analyze_dataset/analyze_quality_filter.py
"""

from __future__ import annotations

import datetime
import random
import statistics
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

DATASET_DIR  = Path("data/midi_files/lmd_full")
LOGS_DIR     = Path("logs")
OUTPUT_FILE  = LOGS_DIR / "quality_filter_simulation.txt"
SAMPLE_SIZE  = 500
NUM_WORKERS  = 4
SEED         = 42
TOTAL_FILES  = 168_385   # known full dataset size

# Filter thresholds
NEAR_CONST_VEL_THRESH = 0.90   # top-1 velocity bin > 90% -> near-constant
MIN_DURATION_SEC      = 15.0   # < 15s -> too short
MIN_NOTE_COUNT        = 50     # < 50 notes -> too sparse
MIN_BAR_COUNT         = 8      # < 8 bars -> 0 chunks produced


# ── Worker (module-level for Windows spawn) ──────────────────────────────────

def _analyze_file(midi_path: Path) -> dict:
    """
    Parse a single MIDI file with miditoolkit and return quality stats.
    All filter flags computed here; no tokenizer used.
    """
    result = {
        "path":          str(midi_path),
        "ok":            False,
        "error":         None,
        # raw metrics
        "num_notes":     0,
        "duration_sec":  0.0,
        "duration_beats": 0.0,
        "bar_count_est": 0,
        "top_vel_pct":   0.0,
        # filter flags (True = file should be removed)
        "f1_const_vel":      False,
        "f2_near_const_vel": False,
        "f3_short":          False,
        "f4_sparse":         False,
        "f5_few_bars":       False,
    }

    try:
        import miditoolkit

        midi = miditoolkit.MidiFile(str(midi_path))
        tpq  = midi.ticks_per_beat

        # First tempo (default 120 BPM if none)
        bpm = 120.0
        if midi.tempo_changes:
            bpm = float(midi.tempo_changes[0].tempo)

        # All notes across all tracks
        all_notes = [n for inst in midi.instruments for n in inst.notes]
        result["num_notes"] = len(all_notes)

        if all_notes:
            max_tick = max(n.end for n in all_notes)
            duration_beats = max_tick / tpq
            duration_sec   = duration_beats * 60.0 / bpm
            # Bar estimate: duration_beats / 4  (assumes 4/4 time)
            bar_count_est  = int(duration_beats / 4.0)

            result["duration_beats"] = duration_beats
            result["duration_sec"]   = duration_sec
            result["bar_count_est"]  = bar_count_est

            # Velocity distribution
            velocities = [n.velocity for n in all_notes]
            vel_counts = Counter(velocities)
            top_count  = max(vel_counts.values())
            top_vel_pct = top_count / len(velocities)
            result["top_vel_pct"] = top_vel_pct

            # Filter 1: constant velocity (all notes same value)
            result["f1_const_vel"] = (len(vel_counts) == 1)

            # Filter 2: near-constant velocity (top-1 bin > threshold)
            result["f2_near_const_vel"] = (top_vel_pct > NEAR_CONST_VEL_THRESH)

            # Filter 3: too short
            result["f3_short"] = (duration_sec < MIN_DURATION_SEC)

            # Filter 4: too sparse
            result["f4_sparse"] = (len(all_notes) < MIN_NOTE_COUNT)

            # Filter 5: too few bars for any chunk
            result["f5_few_bars"] = (bar_count_est < MIN_BAR_COUNT)

        else:
            # No notes at all -- fails all filters
            result["f3_short"]    = True
            result["f4_sparse"]   = True
            result["f5_few_bars"] = True

        result["ok"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total > 0 else 0.0


def _bar(pct: float, width: int = 26) -> str:
    filled = round(pct / 100.0 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _est(sample_rate: float, total: int = TOTAL_FILES) -> int:
    return round(sample_rate * total)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_analysis() -> None:
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    t0 = time.time()

    log("=" * 72)
    log("QUALITY FILTER SIMULATION -- MIDIT LAKH MIDI DATASET")
    log(f"Dataset  : {DATASET_DIR.resolve()}")
    log(f"Date     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)
    log()

    # ── Sample files ─────────────────────────────────────────────────────────
    all_files = sorted(DATASET_DIR.rglob("*.mid"))
    total_found = len(all_files)
    log(f"  Total files found        : {total_found:,}")
    log(f"  Reference (known total)  : {TOTAL_FILES:,}")
    log(f"  Sample size              : {SAMPLE_SIZE}  (seed={SEED})")
    log()

    rng = random.Random(SEED)
    sample = rng.sample(all_files, min(SAMPLE_SIZE, total_found))

    # ── Parse files ───────────────────────────────────────────────────────────
    log(f"[0] PARSING {len(sample)} FILES  ({NUM_WORKERS} workers, miditoolkit)...")
    log("-" * 72)

    t_parse = time.time()
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(_analyze_file, sample)
    parse_elapsed = time.time() - t_parse

    good   = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    log(f"  Done in {parse_elapsed:.1f}s  ({parse_elapsed/len(sample):.2f}s/file)")
    log(f"  OK     : {len(good):>3} / {len(sample)}")
    log(f"  Failed : {len(failed):>3} / {len(sample)}")
    if failed:
        for r in failed[:5]:
            log(f"    {Path(r['path']).name}  --  {r['error'][:60]}")
    log()

    N = len(good)

    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS A -- Bar count distribution
    # ═══════════════════════════════════════════════════════════════════════════
    log("=" * 72)
    log("ANALYSIS A -- BAR COUNT DISTRIBUTION")
    log(f"  (estimate: duration_beats / 4, assuming 4/4 time)")
    log("=" * 72)
    log()

    bar_counts = [r["bar_count_est"] for r in good]

    buckets = [
        ("< 8   bars",    lambda b: b < 8),
        ("8-15  bars",    lambda b: 8  <= b < 16),
        ("16-31 bars",    lambda b: 16 <= b < 32),
        ("32-63 bars",    lambda b: 32 <= b < 64),
        ("64-127 bars",   lambda b: 64 <= b < 128),
        (">= 128 bars",   lambda b: b >= 128),
    ]

    log(f"  {'Bucket':<16}  {'Count':>6}  {'Sample%':>8}  {'Est. full':>10}  Bar chart")
    log(f"  {'-'*16}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*26}")

    for label, cond in buckets:
        cnt  = sum(1 for b in bar_counts if cond(b))
        pct  = _pct(cnt, N)
        est  = _est(cnt / N)
        bar  = _bar(pct)
        log(f"  {label:<16}  {cnt:>6}  {pct:>7.1f}%  {est:>10,}  {bar}")

    log()

    # < 8 bars detail
    lost_lt8 = sum(1 for b in bar_counts if b < 8)
    lost_pct = _pct(lost_lt8, N)
    est_lost = _est(lost_lt8 / N)

    log(f"  Files with < 8 bars (0 chunks, auto-discarded): {lost_lt8} / {N}  ({lost_pct:.1f}%)")
    log(f"  Estimated lost in full {TOTAL_FILES:,}-file dataset : ~{est_lost:,}")
    log()

    if bar_counts:
        log(f"  Bar count stats (non-zero):")
        nonzero = [b for b in bar_counts if b > 0]
        if nonzero:
            log(f"    Mean   : {statistics.mean(nonzero):.1f} bars")
            log(f"    Median : {statistics.median(nonzero):.1f} bars")
            log(f"    Min    : {min(nonzero)}")
            log(f"    Max    : {max(nonzero)}")
    log()

    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS B -- Combined quality filter simulation
    # ═══════════════════════════════════════════════════════════════════════════
    log("=" * 72)
    log("ANALYSIS B -- COMBINED QUALITY FILTER SIMULATION")
    log("=" * 72)
    log()

    filters = [
        ("f1_const_vel",
         "F1: Constant velocity",
         "All notes have identical velocity value"),
        ("f2_near_const_vel",
         "F2: Near-constant velocity",
         f"Top-1 velocity bin > {NEAR_CONST_VEL_THRESH*100:.0f}% of notes"),
        ("f3_short",
         "F3: Duration < 15s",
         "File is too short to contain useful musical content"),
        ("f4_sparse",
         "F4: Note count < 50",
         "File has too few notes"),
        ("f5_few_bars",
         "F5: Bar count < 8",
         "Not enough bars to produce any 8-bar chunk"),
    ]

    log("[B1] PER-FILTER STATISTICS  (each filter applied independently)")
    log("-" * 72)
    log()
    log(f"  {'Filter':<30}  {'Removed':>7}  {'% sample':>8}  {'Est. full':>10}  Bar chart")
    log(f"  {'-'*30}  {'-'*7}  {'-'*8}  {'-'*10}  {'-'*26}")

    filter_sets: dict[str, set[int]] = {}

    for key, name, desc in filters:
        flagged_idx = {i for i, r in enumerate(good) if r[key]}
        filter_sets[key] = flagged_idx
        cnt  = len(flagged_idx)
        pct  = _pct(cnt, N)
        est  = _est(cnt / N)
        bar  = _bar(pct)
        log(f"  {name:<30}  {cnt:>7}  {pct:>7.1f}%  {est:>10,}  {bar}")
        log(f"    ({desc})")
        log()

    # ── Union (any filter triggered) ─────────────────────────────────────────
    log("[B2] COMBINED FILTER  (union -- removed if ANY filter triggers)")
    log("-" * 72)
    log()

    union_idx = set()
    for s in filter_sets.values():
        union_idx |= s

    n_removed = len(union_idx)
    n_survive = N - n_removed
    pct_removed = _pct(n_removed, N)
    pct_survive = _pct(n_survive, N)
    est_survive = _est(n_survive / N)
    est_removed = _est(n_removed / N)

    log(f"  Total removed (union)   : {n_removed:>5} / {N}  ({pct_removed:.1f}%)")
    log(f"  Surviving files         : {n_survive:>5} / {N}  ({pct_survive:.1f}%)")
    log()
    log(f"  Estimated in full {TOTAL_FILES:,}-file dataset:")
    log(f"    Removed : ~{est_removed:,}  ({pct_removed:.1f}%)")
    log(f"    Survive : ~{est_survive:,}  ({pct_survive:.1f}%)")
    log()

    # ── Overlap between filters ────────────────────────────────────────────
    log("[B3] FILTER OVERLAP  (files removed by multiple filters)")
    log("-" * 72)
    log()
    log("  How many files are caught by exactly N filters:")

    overlap_counts = Counter()
    for i in range(N):
        n_flags = sum(1 for key, _, _ in filters if i in filter_sets[key])
        overlap_counts[n_flags] += 1

    for n_flags in sorted(overlap_counts):
        cnt = overlap_counts[n_flags]
        pct = _pct(cnt, N)
        label = (
            "pass all filters" if n_flags == 0
            else f"caught by {n_flags} filter(s)"
        )
        log(f"    {n_flags} flag(s)  ({label:<22}): {cnt:>5}  ({pct:.1f}%)")
    log()

    # ── Incremental filter cascade ────────────────────────────────────────
    log("[B4] INCREMENTAL FILTER CASCADE  (filters applied in order)")
    log("-" * 72)
    log()
    log("  Shows how many files remain after each filter is added cumulatively.")
    log()
    log(f"  {'After filter':<35}  {'Remaining':>9}  {'% of original':>14}")
    log(f"  {'-'*35}  {'-'*9}  {'-'*14}")

    removed_so_far: set[int] = set()
    log(f"  {'(start -- no filters)':<35}  {N:>9}  {100.0:>13.1f}%")
    for key, name, _ in filters:
        removed_so_far |= filter_sets[key]
        remaining = N - len(removed_so_far)
        pct = _pct(remaining, N)
        log(f"  {name:<35}  {remaining:>9}  {pct:>13.1f}%")
    log()

    # ── Surviving files detail ─────────────────────────────────────────────
    log("[B5] SURVIVING FILES  (pass all 5 filters)")
    log("-" * 72)
    log()

    survivors = [r for i, r in enumerate(good) if i not in union_idx]

    if survivors:
        dur_vals  = [r["duration_sec"]  for r in survivors if r["duration_sec"] > 0]
        note_vals = [r["num_notes"]     for r in survivors]
        bar_vals  = [r["bar_count_est"] for r in survivors if r["bar_count_est"] > 0]
        vel_vals  = [r["top_vel_pct"]   for r in survivors]

        log(f"  Count   : {len(survivors)}")
        log()
        if dur_vals:
            log(f"  Duration (sec):")
            log(f"    Mean   {statistics.mean(dur_vals):.1f}s   "
                f"Median {statistics.median(dur_vals):.1f}s   "
                f"Min {min(dur_vals):.1f}s   Max {max(dur_vals):.1f}s")
        if note_vals:
            log(f"  Notes per file:")
            log(f"    Mean   {statistics.mean(note_vals):.0f}   "
                f"Median {statistics.median(note_vals):.0f}   "
                f"Min {min(note_vals)}   Max {max(note_vals)}")
        if bar_vals:
            log(f"  Bar count (est.):")
            log(f"    Mean   {statistics.mean(bar_vals):.1f}   "
                f"Median {statistics.median(bar_vals):.1f}   "
                f"Min {min(bar_vals)}   Max {max(bar_vals)}")
        if vel_vals:
            log(f"  Top-vel% (max velocity concentration):")
            log(f"    Mean   {statistics.mean(vel_vals)*100:.1f}%   "
                f"Median {statistics.median(vel_vals)*100:.1f}%   "
                f"Max {max(vel_vals)*100:.1f}%")
    log()

    # ── Chunk yield after filtering ────────────────────────────────────────
    log("[B6] ESTIMATED CHUNK YIELD AFTER FILTERING")
    log("-" * 72)
    log()
    log("  Chunk count per file = bar_count_est // 8  (bars_per_chunk=8)")
    log()

    total_chunks_sample = sum(r["bar_count_est"] // 8 for r in survivors)
    if survivors:
        avg_chunks = total_chunks_sample / len(survivors)
        est_total_chunks = _est(avg_chunks * n_survive / N)
        log(f"  Avg chunks/surviving file : {avg_chunks:.1f}")
        log(f"  Chunks in sample          : {total_chunks_sample}")
        log(f"  Est. total chunks (full)  : ~{est_total_chunks:,}")
        log()
        log(f"  With pitch augmentation x11 (+/-5 semitones):")
        est_aug_chunks = est_survive * round(avg_chunks) * 11
        log(f"    Est. augmented chunks   : ~{est_aug_chunks:,}")
    log()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0

    log("=" * 72)
    log("SUMMARY")
    log("=" * 72)
    log(f"  Files analyzed           : {N}")
    log()
    log("  Filter rates (independent):")
    for key, name, _ in filters:
        cnt = len(filter_sets[key])
        log(f"    {name:<30} : {cnt:>4} / {N}  ({_pct(cnt, N):.1f}%)")
    log()
    log(f"  Combined removal (union) : {n_removed:>4} / {N}  ({pct_removed:.1f}%)")
    log(f"  Survival rate            : {n_survive:>4} / {N}  ({pct_survive:.1f}%)")
    log()
    log(f"  Est. usable files (full) : ~{est_survive:,} / {TOTAL_FILES:,}  ({pct_survive:.1f}%)")
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
