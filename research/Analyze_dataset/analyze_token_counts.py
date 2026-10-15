"""
Token Count Analysis — Lakh MIDI Dataset
==========================================
Samples 200 random files from data/midi_files/lmd_full/, tokenizes each with
the EXACT MidiTok REMI config used in preprocess_batch_improved.py, and reports:

  1. Token count distribution (per file)
  2. Bar count distribution (per file)
  3. 8-bar chunk count distribution (non-overlapping, per file)
  4. Estimated total chunks for the full lmd_full dataset
  5. Estimated disk space for preprocessed .pt files

Chunking model:
  - Strategy: bar_based (matches lakh_dataset.py + config.yaml)
  - bars_per_chunk: 8
  - Chunks per file = num_bars // 8  (non-overlapping lower-bound)
  - Files with < 8 bars produce 0 usable chunks (filtered out at training time)

Disk space model:
  - Each .pt file stores: torch.tensor(ids, dtype=torch.long) — 8 bytes/token
  - Plus bar_positions list (~8 bytes/bar) + metadata + pickle overhead (~1 KB)
  - Estimate: num_tokens*8 + num_bars*8 + 1024  bytes per file

Usage:
  python scripts/Analyze_dataset/analyze_token_counts.py
  (run from MIDIT project root)

Requirements: miditok  (pip install miditok)
"""

import random
import statistics
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
MIDI_DIR    = Path("data/midi_files/lmd_full")
LOGS_DIR    = Path("logs")
REPORT_PATH = LOGS_DIR / "token_analysis.txt"

# ─── Config ───────────────────────────────────────────────────────────────────
SAMPLE_SIZE   = 200
NUM_WORKERS   = 4
RANDOM_SEED   = 42
BARS_PER_CHUNK = 8          # matches config.yaml chunking.bars_per_chunk
MAX_TOKENS     = 8192       # matches config.yaml chunking.max_tokens

# Full lmd_full dataset target for estimation (CLAUDE.md: "~160,000 files").
# The script will also compute the real total from a directory scan.
FULL_DATASET_APPROX = 160_000


# ─── Worker initializer — creates tokenizer once per worker process ───────────
# (defined at module level so it's picklable on Windows / spawn)

_tokenizer   = None
_id_to_token = None


def _init_worker():
    """Initialize the MidiTok tokenizer in each worker process."""
    global _tokenizer, _id_to_token
    from miditok import REMI, TokenizerConfig   # imported inside worker

    # EXACT config from scripts/Pre_Processing/preprocess_batch_improved.py
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
    _tokenizer   = REMI(config)
    _id_to_token = {v: k for k, v in _tokenizer.vocab.items()}


# ─── Per-file analysis (runs in worker processes) ─────────────────────────────

def analyze_file(midi_path: Path) -> dict:
    """Tokenize one MIDI file and return token/bar/chunk stats."""
    result = {
        "path":          str(midi_path),
        "filename":      midi_path.name,
        "success":       False,
        "error":         None,
        "num_tokens":    0,
        "num_bars":      0,
        "num_chunks":    0,
        "bytes_estimate": 0,
    }

    try:
        global _tokenizer, _id_to_token

        # ── Tokenize ────────────────────────────────────────────────────────
        tokens = _tokenizer(str(midi_path))

        # Handle both single TokSequence and list of TokSequence
        # (matches the exact pattern in preprocess_batch_improved.py)
        if isinstance(tokens, list):
            all_ids = []
            for tok_seq in tokens:
                all_ids.extend(tok_seq.ids)
        else:
            all_ids = tokens.ids

        num_tokens = len(all_ids)

        # ── Count bars (same logic as preprocess_batch_improved.py) ────────
        bar_positions = []
        for idx, token_id in enumerate(all_ids):
            token_name = _id_to_token.get(token_id, None)
            if token_name and "Bar" in token_name:
                bar_positions.append(idx)

        # Ensure the sequence starts at 0 (mirrors the preprocessing fix)
        if not bar_positions or bar_positions[0] != 0:
            bar_positions.insert(0, 0)
        bar_positions.append(num_tokens)

        num_bars = len(bar_positions) - 1

        # ── Non-overlapping 8-bar chunks ─────────────────────────────────
        # Files with < 8 bars are skipped by LakhDataset._chunk_by_bars().
        num_chunks = num_bars // BARS_PER_CHUNK

        # ── Disk space estimate ──────────────────────────────────────────
        # torch.tensor(ids, dtype=torch.long) = 8 bytes/token
        # bar_positions list ≈ 8 bytes/entry
        # metadata + pickle overhead ≈ 1 KB
        bytes_est = num_tokens * 8 + num_bars * 8 + 1024

        result.update({
            "success":        True,
            "num_tokens":     num_tokens,
            "num_bars":       num_bars,
            "num_chunks":     num_chunks,
            "bytes_estimate": bytes_est,
        })

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _stats(values: list, unit: str = "", fmt: str = ".1f") -> list:
    if not values:
        return ["  (no data)"]
    lines = [
        f"  Count  : {len(values):,}",
        f"  Mean   : {statistics.mean(values):{fmt}}{unit}",
        f"  Median : {statistics.median(values):{fmt}}{unit}",
        f"  Min    : {min(values):{fmt}}{unit}",
        f"  Max    : {max(values):{fmt}}{unit}",
    ]
    if len(values) > 1:
        lines.append(f"  StdDev : {statistics.stdev(values):{fmt}}{unit}")
    return lines


def _bar(count: int, total: int, width: int = 28) -> str:
    filled = int(width * count / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ─── Percentile helper ────────────────────────────────────────────────────────

def _percentile(sorted_values: list, pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = (len(sorted_values) - 1) * pct / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_analysis():
    output_lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        output_lines.append(msg)

    t0 = time.time()

    log("=" * 70)
    log("LAKH MIDI — TOKEN COUNT ANALYSIS")
    log(f"Tokenizer : MidiTok REMI  (config from preprocess_batch_improved.py)")
    log(f"Dataset   : {MIDI_DIR.resolve()}")
    log(f"Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # ── Validate dataset path ────────────────────────────────────────────────
    if not MIDI_DIR.exists():
        log(f"\nERROR: dataset directory not found: {MIDI_DIR.resolve()}")
        return

    # ── Count total files (fast: one glob per top-level subdir) ─────────────
    log("\n[1] DATASET SIZE SCAN")
    log("-" * 70)

    top_subdirs = sorted(p for p in MIDI_DIR.iterdir() if p.is_dir())
    all_midi: list[Path] = []
    for sd in top_subdirs:
        all_midi.extend(sd.glob("*.mid"))
        all_midi.extend(sd.glob("*.midi"))

    total_files = len(all_midi)
    log(f"  Top-level subdirs : {len(top_subdirs)}  ({', '.join(d.name for d in top_subdirs)})")
    log(f"  Total MIDI files  : {total_files:,}")

    # ── Sample ───────────────────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    sample_size = min(SAMPLE_SIZE, total_files)
    sample: list[Path] = random.sample(all_midi, sample_size)
    log(f"  Sample size       : {sample_size}  (seed={RANDOM_SEED})")

    # ── Tokenizer info (init once in main process for reporting) ─────────────
    log("\n[2] TOKENIZER CONFIG")
    log("-" * 70)
    from miditok import REMI, TokenizerConfig
    _cfg = TokenizerConfig(
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
    _tok = REMI(_cfg)
    vocab_size = len(_tok.vocab)
    del _tok, _cfg  # free before spawning workers

    log(f"  Type              : REMI")
    log(f"  pitch_range       : (21, 109)")
    log(f"  beat_res          : {{(0, 4): 8}}")
    log(f"  num_velocities    : 32")
    log(f"  use_programs      : True")
    log(f"  use_tempos        : True")
    log(f"  use_time_sigs     : True")
    log(f"  use_rests         : True")
    log(f"  use_chords        : False")
    log(f"  special_tokens    : [PAD, BOS, EOS, MASK]")
    log(f"  Vocab size        : {vocab_size}")

    # ── Parallel tokenization ────────────────────────────────────────────────
    log(f"\n[3] PARALLEL TOKENIZATION  ({NUM_WORKERS} workers, {sample_size} files)")
    log("-" * 70)

    with Pool(NUM_WORKERS, initializer=_init_worker) as pool:
        results = pool.map(analyze_file, sample)

    elapsed = time.time() - t0
    log(f"  Done in {elapsed:.1f}s  ({elapsed/sample_size:.2f}s/file)")

    good = [r for r in results if r["success"]]
    bad  = [r for r in results if not r["success"]]
    n_good = len(good)
    n_bad  = len(bad)

    log(f"  Tokenized OK : {n_good:4d} / {sample_size}")
    log(f"  Failed       : {n_bad:4d} / {sample_size}")

    if bad:
        log(f"\n  Failed files:")
        err_types = Counter((r["error"] or "unknown")[:70] for r in bad)
        for msg, cnt in err_types.most_common(5):
            log(f"    x{cnt}  {msg}")

    if n_good == 0:
        log("\nERROR: no files tokenized successfully. Cannot produce statistics.")
        return

    # ── Extract arrays ───────────────────────────────────────────────────────
    token_counts  = [r["num_tokens"]  for r in good]
    bar_counts    = [r["num_bars"]    for r in good]
    chunk_counts  = [r["num_chunks"]  for r in good]
    bytes_list    = [r["bytes_estimate"] for r in good]

    token_counts_s  = sorted(token_counts)
    bar_counts_s    = sorted(bar_counts)
    chunk_counts_s  = sorted(chunk_counts)

    # ── 4. Token count distribution ──────────────────────────────────────────
    log(f"\n[4] TOKEN COUNTS PER FILE")
    log("-" * 70)
    for line in _stats(token_counts, " tokens", fmt=".0f"):
        log(line)

    log(f"\n  Percentiles:")
    for p in (10, 25, 50, 75, 90, 95, 99):
        v = _percentile(token_counts_s, p)
        log(f"    p{p:2d}: {v:>8,.0f} tokens")

    # Token count buckets
    buckets_tokens = [
        ("< 500",          sum(1 for t in token_counts if t < 500)),
        ("500 – 1 000",    sum(1 for t in token_counts if 500    <= t < 1_000)),
        ("1 000 – 5 000",  sum(1 for t in token_counts if 1_000  <= t < 5_000)),
        ("5 000 – 10 000", sum(1 for t in token_counts if 5_000  <= t < 10_000)),
        ("10 000 – 20 000",sum(1 for t in token_counts if 10_000 <= t < 20_000)),
        ("> 20 000",       sum(1 for t in token_counts if t >= 20_000)),
    ]
    log(f"\n  Buckets:")
    for label, cnt in buckets_tokens:
        pct = 100 * cnt / n_good
        log(f"  {label:18s}: {cnt:4d}  ({pct:5.1f}%)  {_bar(cnt, n_good)}")

    # ── 5. Bar count distribution ─────────────────────────────────────────────
    log(f"\n[5] BAR COUNTS PER FILE")
    log("-" * 70)
    for line in _stats(bar_counts, " bars", fmt=".0f"):
        log(line)

    log(f"\n  Percentiles:")
    for p in (10, 25, 50, 75, 90, 95, 99):
        v = _percentile(bar_counts_s, p)
        log(f"    p{p:2d}: {v:>6,.0f} bars")

    # Tokens per bar (derived)
    tpb_list = [t / b for t, b in zip(token_counts, bar_counts) if b > 0]
    log(f"\n  Tokens per bar (derived):")
    for line in _stats(tpb_list, " tokens/bar", fmt=".1f"):
        log(line)

    buckets_bars = [
        ("< 8  (no chunks)", sum(1 for b in bar_counts if b < BARS_PER_CHUNK)),
        ("8 – 16",           sum(1 for b in bar_counts if BARS_PER_CHUNK <= b < 16)),
        ("16 – 32",          sum(1 for b in bar_counts if 16 <= b < 32)),
        ("32 – 64",          sum(1 for b in bar_counts if 32 <= b < 64)),
        ("64 – 128",         sum(1 for b in bar_counts if 64 <= b < 128)),
        ("> 128",            sum(1 for b in bar_counts if b >= 128)),
    ]
    log(f"\n  Bar count buckets:")
    for label, cnt in buckets_bars:
        pct = 100 * cnt / n_good
        log(f"  {label:20s}: {cnt:4d}  ({pct:5.1f}%)  {_bar(cnt, n_good)}")

    skipped = sum(1 for b in bar_counts if b < BARS_PER_CHUNK)
    log(f"\n  Files skipped (< {BARS_PER_CHUNK} bars): {skipped}  ({100*skipped/n_good:.1f}%)")

    # ── 6. Chunk count distribution ───────────────────────────────────────────
    log(f"\n[6] 8-BAR CHUNK COUNTS PER FILE  (non-overlapping: num_bars // {BARS_PER_CHUNK})")
    log("-" * 70)
    for line in _stats(chunk_counts, " chunks", fmt=".1f"):
        log(line)

    log(f"\n  Percentiles:")
    for p in (10, 25, 50, 75, 90, 95, 99):
        v = _percentile(chunk_counts_s, p)
        log(f"    p{p:2d}: {v:>5,.1f} chunks/file")

    chunk_dist = Counter(chunk_counts)
    log(f"\n  Most common chunk counts (top 15):")
    for n_ch, cnt in chunk_dist.most_common(15):
        pct = 100 * cnt / n_good
        log(f"  {n_ch:3d} chunks: {cnt:4d} files  ({pct:5.1f}%)  {_bar(cnt, n_good, 20)}")

    # ── 7. Full-dataset extrapolation ─────────────────────────────────────────
    log(f"\n[7] FULL DATASET EXTRAPOLATION")
    log("-" * 70)

    # Use the actual scanned total, not the approximation
    full_n = total_files

    # Rates from sample
    success_rate  = n_good / sample_size
    skip_rate     = skipped / n_good                   # fraction with < 8 bars
    mean_chunks   = statistics.mean(chunk_counts)
    mean_tokens   = statistics.mean(token_counts)
    mean_bytes    = statistics.mean(bytes_list)

    est_readable  = int(full_n * success_rate)
    est_usable    = int(est_readable * (1 - skip_rate))
    est_chunks    = int(est_usable  * mean_chunks)
    est_bytes     = est_readable * mean_bytes
    est_bytes_tot = full_n * mean_bytes * success_rate

    log(f"  Full dataset file count          : {full_n:>12,}")
    log(f"  Estimated readable files         : {est_readable:>12,}  ({100*success_rate:.1f}%)")
    log(f"  Estimated files with ≥8 bars     : {est_usable:>12,}  ({100*(1-skip_rate):.1f}% of readable)")
    log(f"  Estimated total 8-bar chunks     : {est_chunks:>12,}")
    log(f"    (= avg {mean_chunks:.1f} chunks/file × {est_usable:,} usable files)")

    # Breakdown by chunk count range
    log(f"\n  Chunk yield sensitivity (avg tokens/file = {mean_tokens:,.0f}):")
    for bars_assumed in (32, 64, 128):
        chunks_assumed = bars_assumed // BARS_PER_CHUNK
        est = int(full_n * success_rate * (1 - skip_rate) * chunks_assumed)
        log(f"    If avg = {bars_assumed:4d} bars/file → {chunks_assumed} chunks/file → {est:>10,} total chunks")

    # ── 8. Disk space estimate ─────────────────────────────────────────────────
    log(f"\n[8] DISK SPACE ESTIMATE  (.pt files, dtype=torch.long)")
    log("-" * 70)
    log(f"  Model:  num_tokens × 8 bytes  +  num_bars × 8 bytes  +  1 KB overhead")
    log(f"\n  Per file:")
    log(f"    Mean tokens/file          : {mean_tokens:>10,.0f}")
    log(f"    Mean bars/file            : {statistics.mean(bar_counts):>10,.1f}")
    log(f"    Estimated bytes/file      : {_human_bytes(mean_bytes):>15}")

    log(f"\n  For {est_readable:,} readable files (lmd_full):")
    log(f"    Total estimated size      : {_human_bytes(est_bytes_tot):>15}")
    log(f"    (i.e. {est_bytes_tot/1e9:.1f} GB)")

    log(f"\n  Sensitivity (bytes/token × num_tokens × num_files):")
    for tpf in (5_000, 10_000, 20_000, 40_000):
        b = tpf * 8 * est_readable
        log(f"    {tpf:>6,} tokens/file × {est_readable:,} files → {_human_bytes(b)}")

    log(f"\n  Note: actual .pt overhead includes Python pickle framing.")
    log(f"  Cross-check from CLAUDE.md: current 9,987-file subset ≈ 2.2 GB")
    implied_tpf = (2.2e9 / 9987) / 8
    log(f"  → implies avg {implied_tpf:,.0f} tokens/file for that subset")

    # ── Summary ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")
    log(f"  Files scanned (lmd_full)         : {total_files:>12,}")
    log(f"  Sample size                      : {sample_size:>12,}")
    log(f"  Tokenized OK                     : {n_good:>12,}")
    log(f"  Failed                           : {n_bad:>12,}")
    log(f"  Tokenizer vocab size             : {vocab_size:>12,}")
    log(f"  Mean tokens / file               : {mean_tokens:>12,.0f}")
    log(f"  Mean bars  / file                : {statistics.mean(bar_counts):>12,.1f}")
    log(f"  Mean 8-bar chunks / file         : {mean_chunks:>12,.1f}")
    log(f"  Files with < 8 bars (skipped)    : {100*skip_rate:>11.1f}%")
    log(f"  Est. total chunks (full dataset) : {est_chunks:>12,}")
    log(f"  Est. disk space   (full dataset) : {_human_bytes(est_bytes_tot):>15}")
    log(f"\n  Analysis time  : {total_elapsed:.1f}s")
    log(f"  Report saved   : {REPORT_PATH.resolve()}")
    log("=" * 70)

    # ── Save report ──────────────────────────────────────────────────────────
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(output_lines) + "\n")

    print(f"\nReport written to: {REPORT_PATH.resolve()}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Guard required on Windows (spawn-based multiprocessing)
    try:
        from miditok import REMI  # noqa: F401
    except ImportError:
        print("ERROR: miditok not found.\n  Install with:  pip install miditok")
        sys.exit(1)

    run_analysis()
