"""
Pitch & Token Analysis -- MIDIT Lakh MIDI Dataset
==================================================
Samples 400 random MIDI files from data/midi_files/lmd_full/
using miditoolkit (no tokenizer -- fast raw MIDI reading).

Analysis 1: Pitch diversity per file (non-drum unique pitches)
Analysis 2: Token count estimate distribution
Analysis 3: Max-instruments filter simulation
Analysis 4: Silence ratio estimation

Run from MIDIT project root:
    python scripts/Analyze_dataset/analyze_pitch_and_tokens.py
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

DATASET_DIR = Path("data/midi_files/lmd_full")
LOGS_DIR    = Path("logs")
OUTPUT_FILE = LOGS_DIR / "pitch_and_token_analysis.txt"
SAMPLE_SIZE = 400
NUM_WORKERS = 4
SEED        = 42
TOTAL_FILES = 168_385

# Thresholds
LOW_PITCH_DIV_1  =  5   # < 5  unique pitches
LOW_PITCH_DIV_2  = 10   # < 10 unique pitches
MIN_TOKENS_FLAG  =  500
MAX_TOKENS_FLAG  = 50_000
SILENCE_FLAG     =  0.50  # > 50% silence
INSTR_THRESHOLDS = [4, 6, 8, 10, 12, 16]


# ── Worker (module-level for Windows spawn) ──────────────────────────────────

def _analyze_file(midi_path: Path) -> dict:
    """Parse one MIDI file with miditoolkit and return stats dict."""
    result = {
        "path":             str(midi_path),
        "ok":               False,
        "error":            None,
        # metrics
        "num_notes_total":  0,
        "num_notes_melodic": 0,
        "duration_beats":   0.0,
        "duration_ticks":   0,
        "unique_pitches":   0,
        "unique_programs":  0,
        "sum_note_dur_ticks": 0,
        "silence_ratio":    0.0,
        "est_tokens":       0,
    }

    try:
        import miditoolkit

        midi = miditoolkit.MidiFile(str(midi_path))
        tpq  = midi.ticks_per_beat

        # Separate drum and melodic instruments
        melodic_insts = [i for i in midi.instruments if not i.is_drum]
        drum_insts    = [i for i in midi.instruments if i.is_drum]

        melodic_notes = [n for i in melodic_insts for n in i.notes]
        drum_notes    = [n for i in drum_insts    for n in i.notes]
        all_notes     = melodic_notes + drum_notes

        result["num_notes_total"]   = len(all_notes)
        result["num_notes_melodic"] = len(melodic_notes)

        # Unique pitches (non-drum only)
        result["unique_pitches"] = len({n.pitch for n in melodic_notes})

        # Unique melodic programs
        result["unique_programs"] = len({i.program for i in melodic_insts})

        if all_notes:
            max_tick = max(n.end for n in all_notes)
            result["duration_ticks"]  = max_tick
            result["duration_beats"]  = max_tick / tpq

            # Sum of individual note durations (can exceed max_tick due to polyphony)
            sum_dur = sum(n.end - n.start for n in all_notes)
            result["sum_note_dur_ticks"] = sum_dur

            # Silence ratio: (total_dur - sum_note_dur) / total_dur
            # Clamped to [0, 1] -- can go negative with heavy polyphony
            raw_silence = (max_tick - sum_dur) / max_tick
            result["silence_ratio"] = max(0.0, min(1.0, raw_silence))

            # Token count estimate: notes * 5 + bars * 2
            bar_count_est = max(1, int(result["duration_beats"] / 4.0))
            result["est_tokens"] = len(all_notes) * 5 + bar_count_est * 2

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


def _est(rate: float, total: int = TOTAL_FILES) -> int:
    return round(rate * total)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_analysis() -> None:
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    t0 = time.time()

    log("=" * 72)
    log("PITCH & TOKEN ANALYSIS -- MIDIT LAKH MIDI DATASET")
    log(f"Dataset  : {DATASET_DIR.resolve()}")
    log(f"Date     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)
    log()

    # ── Sample files ─────────────────────────────────────────────────────────
    all_files = sorted(DATASET_DIR.rglob("*.mid"))
    log(f"  Total files found : {len(all_files):,}")
    log(f"  Sample size       : {SAMPLE_SIZE}  (seed={SEED})")
    log()

    rng = random.Random(SEED)
    sample = rng.sample(all_files, min(SAMPLE_SIZE, len(all_files)))

    # ── Parse ─────────────────────────────────────────────────────────────────
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
    # ANALYSIS 1 -- Pitch diversity
    # ═══════════════════════════════════════════════════════════════════════════
    log("=" * 72)
    log("ANALYSIS 1 -- PITCH DIVERSITY  (non-drum unique MIDI pitch values)")
    log("=" * 72)
    log()

    pitch_divs = [r["unique_pitches"] for r in good]
    melodic_only = [r for r in good if r["num_notes_melodic"] > 0]

    # Stats on files that actually have melodic notes
    pitch_divs_m = [r["unique_pitches"] for r in melodic_only]

    log(f"  Files with at least one melodic note : {len(melodic_only)} / {N}")
    log(f"  Files with 0 melodic notes (drums only): {N - len(melodic_only)} / {N}")
    log()

    if pitch_divs_m:
        log(f"  Unique pitch stats (melodic files only):")
        log(f"    Mean   : {statistics.mean(pitch_divs_m):.1f}")
        log(f"    Median : {statistics.median(pitch_divs_m):.1f}")
        log(f"    Min    : {min(pitch_divs_m)}")
        log(f"    Max    : {max(pitch_divs_m)}")
        log(f"    StdDev : {statistics.stdev(pitch_divs_m):.1f}")
    log()

    # Buckets
    pitch_buckets = [
        ("0 pitches  (no melody)",  lambda p: p == 0),
        ("1-4 pitches",             lambda p: 1  <= p < 5),
        ("5-9 pitches",             lambda p: 5  <= p < 10),
        ("10-19 pitches",           lambda p: 10 <= p < 20),
        ("20-35 pitches",           lambda p: 20 <= p < 36),
        ("36-60 pitches",           lambda p: 36 <= p < 61),
        ("> 60 pitches",            lambda p: p >= 61),
    ]

    log(f"  {'Bucket':<24}  {'Count':>6}  {'Sample%':>8}  {'Est.full':>9}  Bar chart")
    log(f"  {'-'*24}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*26}")
    for label, cond in pitch_buckets:
        cnt = sum(1 for p in pitch_divs if cond(p))
        pct = _pct(cnt, N)
        log(f"  {label:<24}  {cnt:>6}  {pct:>7.1f}%  {_est(cnt/N):>9,}  {_bar(pct)}")
    log()

    # Low-diversity flags
    low5  = sum(1 for p in pitch_divs if 0 < p < 5)
    low10 = sum(1 for p in pitch_divs if 0 < p < 10)
    log(f"  Files with 1-4 unique pitches  (loop/rhythmic) : {low5:>4} / {N}  "
        f"({_pct(low5, N):.1f}%)  -- est. {_est(low5/N):,} in full dataset")
    log(f"  Files with 1-9 unique pitches  (very low div.) : {low10:>4} / {N}  "
        f"({_pct(low10, N):.1f}%)  -- est. {_est(low10/N):,} in full dataset")
    log()

    # Examples with very low pitch diversity (1-4)
    low_div_examples = sorted(
        [r for r in good if 0 < r["unique_pitches"] < 5],
        key=lambda r: r["unique_pitches"]
    )
    if low_div_examples:
        log(f"  Example files with 1-4 unique pitches (first 10):")
        log(f"  {'Pitches':>7}  {'Notes':>7}  {'Bars':>6}  File")
        log(f"  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*40}")
        for r in low_div_examples[:10]:
            bar_est = int(r["duration_beats"] / 4.0) if r["duration_beats"] > 0 else 0
            log(f"  {r['unique_pitches']:>7}  {r['num_notes_melodic']:>7}  "
                f"{bar_est:>6}  {Path(r['path']).name}")
    log()

    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS 2 -- Token count estimate
    # ═══════════════════════════════════════════════════════════════════════════
    log("=" * 72)
    log("ANALYSIS 2 -- ESTIMATED TOKEN COUNT DISTRIBUTION")
    log(f"  (estimate: num_notes * 5 + num_bars * 2,  num_bars = duration_beats / 4)")
    log("=" * 72)
    log()

    tok_ests = [r["est_tokens"] for r in good]

    log(f"  Token count estimate stats:")
    if tok_ests:
        log(f"    Mean   : {statistics.mean(tok_ests):,.0f}")
        log(f"    Median : {statistics.median(tok_ests):,.0f}")
        log(f"    Min    : {min(tok_ests):,}")
        log(f"    Max    : {max(tok_ests):,}")
        log(f"    StdDev : {statistics.stdev(tok_ests):,.0f}")
    log()

    tok_buckets = [
        ("< 500",           lambda t: t < 500),
        ("500 - 1k",        lambda t: 500   <= t < 1_000),
        ("1k - 5k",         lambda t: 1_000 <= t < 5_000),
        ("5k - 20k",        lambda t: 5_000 <= t < 20_000),
        ("20k - 50k",       lambda t: 20_000 <= t < 50_000),
        ("> 50k",           lambda t: t >= 50_000),
    ]

    log(f"  {'Bucket':<14}  {'Count':>6}  {'Sample%':>8}  {'Est.full':>9}  Bar chart")
    log(f"  {'-'*14}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*26}")
    for label, cond in tok_buckets:
        cnt = sum(1 for t in tok_ests if cond(t))
        pct = _pct(cnt, N)
        log(f"  {label:<14}  {cnt:>6}  {pct:>7.1f}%  {_est(cnt/N):>9,}  {_bar(pct)}")
    log()

    below500 = sum(1 for t in tok_ests if t < MIN_TOKENS_FLAG)
    above50k = sum(1 for t in tok_ests if t >= MAX_TOKENS_FLAG)
    log(f"  Below {MIN_TOKENS_FLAG:,} tokens  (near-empty)  : {below500} / {N}  "
        f"({_pct(below500, N):.1f}%)  -- est. {_est(below500/N):,} in full dataset")
    log(f"  Above {MAX_TOKENS_FLAG:,} tokens (very long) : {above50k} / {N}  "
        f"({_pct(above50k, N):.1f}%)  -- est. {_est(above50k/N):,} in full dataset")
    log()

    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS 3 -- Max instruments filter simulation
    # ═══════════════════════════════════════════════════════════════════════════
    log("=" * 72)
    log("ANALYSIS 3 -- MAX MELODIC INSTRUMENTS FILTER SIMULATION")
    log("  (counts unique non-drum program numbers per file)")
    log("=" * 72)
    log()

    prog_counts = [r["unique_programs"] for r in good]

    log(f"  Unique melodic programs stats:")
    if prog_counts:
        log(f"    Mean   : {statistics.mean(prog_counts):.1f}")
        log(f"    Median : {statistics.median(prog_counts):.1f}")
        log(f"    Min    : {min(prog_counts)}")
        log(f"    Max    : {max(prog_counts)}")
    log()

    # Distribution by program count
    prog_dist = Counter(prog_counts)
    log(f"  Distribution of unique melodic programs per file (top 20):")
    log(f"  {'Programs':>9}  {'Files':>6}  {'%':>6}  Bar chart")
    log(f"  {'-'*9}  {'-'*6}  {'-'*6}  {'-'*26}")
    for n_prog in sorted(prog_dist.keys())[:20]:
        cnt = prog_dist[n_prog]
        pct = _pct(cnt, N)
        log(f"  {n_prog:>9}  {cnt:>6}  {pct:>5.1f}%  {_bar(pct)}")
    if max(prog_dist.keys(), default=0) >= 20:
        cnt_above = sum(v for k, v in prog_dist.items() if k >= 20)
        log(f"  {'20+':>9}  {cnt_above:>6}  {_pct(cnt_above, N):>5.1f}%")
    log()

    log(f"  Filter simulation (remove files with MORE than N melodic instruments):")
    log(f"  {'Max instr':>10}  {'Removed':>8}  {'Survives':>9}  {'Survival%':>10}  "
        f"{'Est.survive':>12}")
    log(f"  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*12}")
    for threshold in INSTR_THRESHOLDS:
        removed  = sum(1 for p in prog_counts if p > threshold)
        survives = N - removed
        pct      = _pct(survives, N)
        est      = _est(survives / N)
        log(f"  {threshold:>10}  {removed:>8}  {survives:>9}  {pct:>9.1f}%  {est:>12,}")
    log()

    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS 4 -- Silence ratio
    # ═══════════════════════════════════════════════════════════════════════════
    log("=" * 72)
    log("ANALYSIS 4 -- SILENCE RATIO ESTIMATION")
    log(f"  (estimate: (total_dur_ticks - sum_note_dur_ticks) / total_dur_ticks)")
    log(f"  Note: clamped to [0,1] -- polyphony causes sum_note_dur > total_dur")
    log("=" * 72)
    log()

    sil_vals = [r["silence_ratio"] for r in good if r["duration_ticks"] > 0]

    if sil_vals:
        log(f"  Silence ratio stats:")
        log(f"    Mean   : {statistics.mean(sil_vals)*100:.1f}%")
        log(f"    Median : {statistics.median(sil_vals)*100:.1f}%")
        log(f"    Min    : {min(sil_vals)*100:.1f}%")
        log(f"    Max    : {max(sil_vals)*100:.1f}%")
        log(f"    StdDev : {statistics.stdev(sil_vals)*100:.1f}%")
    log()

    sil_buckets = [
        ("0%   (heavy polyphony)",  lambda s: s == 0.0),
        ("0-10%",                   lambda s: 0.0  < s <= 0.10),
        ("10-25%",                  lambda s: 0.10 < s <= 0.25),
        ("25-50%",                  lambda s: 0.25 < s <= 0.50),
        ("50-75%",                  lambda s: 0.50 < s <= 0.75),
        ("> 75% silence",           lambda s: s > 0.75),
    ]

    log(f"  {'Bucket':<24}  {'Count':>6}  {'Sample%':>8}  {'Est.full':>9}  Bar chart")
    log(f"  {'-'*24}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*26}")
    for label, cond in sil_buckets:
        cnt = sum(1 for s in sil_vals if cond(s))
        pct = _pct(cnt, len(sil_vals))
        log(f"  {label:<24}  {cnt:>6}  {pct:>7.1f}%  {_est(cnt/len(sil_vals)):>9,}  {_bar(pct)}")
    log()

    high_silence = [r for r in good if r["silence_ratio"] > SILENCE_FLAG]
    pct_hs = _pct(len(high_silence), N)
    log(f"  Files with silence > {SILENCE_FLAG*100:.0f}% : {len(high_silence)} / {N}  "
        f"({pct_hs:.1f}%)  -- est. {_est(len(high_silence)/N):,} in full dataset")
    log()

    if pct_hs >= 5.0:
        log(f"  NOTE: {pct_hs:.1f}% of files have >50% silence. These may be sparse arrangements")
        log(f"  or files with mostly rests. Consider adding to quality filter if > 10%.")
    else:
        log(f"  OK: silence ratio is low across files.")
    log()

    if high_silence:
        high_silence_sorted = sorted(high_silence, key=lambda r: -r["silence_ratio"])
        log(f"  Top 10 highest-silence files:")
        log(f"  {'Silence%':>9}  {'Notes':>7}  {'Bars':>6}  File")
        log(f"  {'-'*9}  {'-'*7}  {'-'*6}  {'-'*40}")
        for r in high_silence_sorted[:10]:
            bar_est = int(r["duration_beats"] / 4.0) if r["duration_beats"] > 0 else 0
            log(f"  {r['silence_ratio']*100:>8.1f}%  {r['num_notes_total']:>7}  "
                f"{bar_est:>6}  {Path(r['path']).name}")
    log()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0

    log("=" * 72)
    log("SUMMARY")
    log("=" * 72)
    log(f"  Files analyzed           : {N}")
    log()
    log(f"  [1] Pitch diversity:")
    log(f"    Files with 0 melodic notes    : {sum(1 for p in pitch_divs if p == 0):>4} / {N}  "
        f"({_pct(sum(1 for p in pitch_divs if p==0), N):.1f}%)")
    log(f"    Files with 1-4 unique pitches : {low5:>4} / {N}  ({_pct(low5, N):.1f}%)")
    log(f"    Files with 1-9 unique pitches : {low10:>4} / {N}  ({_pct(low10, N):.1f}%)")
    if pitch_divs_m:
        log(f"    Median unique pitches (melodic): {statistics.median(pitch_divs_m):.0f}")
    log()
    log(f"  [2] Token count estimate:")
    if tok_ests:
        log(f"    Below {MIN_TOKENS_FLAG:,} (near-empty) : {below500:>4} / {N}  ({_pct(below500, N):.1f}%)")
        log(f"    Above {MAX_TOKENS_FLAG:,} (very long): {above50k:>4} / {N}  ({_pct(above50k, N):.1f}%)")
        log(f"    Median estimate           : {statistics.median(tok_ests):,.0f} tokens")
    log()
    log(f"  [3] Max instruments filter:")
    for threshold in [6, 10, 16]:
        removed  = sum(1 for p in prog_counts if p > threshold)
        survives = N - removed
        log(f"    Max {threshold:>2} instruments -> removes {removed:>4} / {N}  "
            f"({_pct(removed, N):.1f}%),  survives {_pct(survives, N):.1f}%")
    log()
    log(f"  [4] Silence ratio:")
    log(f"    Files with > 50% silence : {len(high_silence):>4} / {N}  ({pct_hs:.1f}%)")
    if sil_vals:
        log(f"    Median silence ratio     : {statistics.median(sil_vals)*100:.1f}%")
    log()
    log(f"  Analysis time  : {total_elapsed:.1f}s")
    log(f"  Report saved   : {OUTPUT_FILE.resolve()}")
    log("=" * 72)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nReport written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_analysis()
