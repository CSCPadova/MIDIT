"""
Lakh MIDI Dataset Analysis Script
==================================
Analyzes a random sample of 500 files from data/midi_files/lmd_full/.

Reports:
  1. Directory structure and total file count
  2. Corrupted / unreadable files
  3. Constant-velocity files (low-quality signal, flagged for filtering)
  4. Distributions: duration, track count, tempo, time signatures
  5. Saves full report to logs/dataset_analysis.txt

Usage:
  python scripts/Analyze_dataset/analyze_lakh_dataset.py

Run from the MIDIT project root.
Requires: miditoolkit  (pip install miditoolkit)
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
REPORT_PATH = LOGS_DIR / "dataset_analysis.txt"

# ─── Config ───────────────────────────────────────────────────────────────────
SAMPLE_SIZE  = 500
NUM_WORKERS  = 4
RANDOM_SEED  = 42


# ─── Per-file analysis (runs in worker processes) ─────────────────────────────
def analyze_file(midi_path: Path) -> dict:
    """Parse a single MIDI file and return a stats dict."""
    result = {
        "path":                    str(midi_path),
        "filename":                midi_path.name,
        "success":                 False,
        "error":                   None,
        # populated on success:
        "file_size_kb":            0.0,
        "num_tracks":              0,
        "num_notes":               0,
        "duration_beats":          0.0,
        "duration_sec":            0.0,
        "tpq":                     0,
        "tempo_bpm":               None,
        "num_tempo_changes":       0,
        "time_signature":          None,
        "has_drums":               False,
        "constant_velocity":       False,
        "constant_velocity_value": None,
    }

    try:
        import miditoolkit  # imported here so workers don't need a global import

        result["file_size_kb"] = midi_path.stat().st_size / 1024.0
        midi = miditoolkit.MidiFile(str(midi_path))

        result["tpq"]               = midi.ticks_per_beat
        result["num_tracks"]        = len(midi.instruments)
        result["has_drums"]         = any(inst.is_drum for inst in midi.instruments)
        result["num_tempo_changes"] = len(midi.tempo_changes)

        # Tempo — miditoolkit stores BPM directly
        if midi.tempo_changes:
            result["tempo_bpm"] = midi.tempo_changes[0].tempo

        # Time signature (first event, default 4/4)
        if midi.time_signature_changes:
            ts = midi.time_signature_changes[0]
            result["time_signature"] = (ts.numerator, ts.denominator)
        else:
            result["time_signature"] = (4, 4)

        # All notes across all instruments
        all_notes = [note for inst in midi.instruments for note in inst.notes]
        result["num_notes"] = len(all_notes)

        if all_notes:
            max_tick = max(note.end for note in all_notes)
            result["duration_beats"] = max_tick / midi.ticks_per_beat
            bpm = result["tempo_bpm"] or 120.0
            result["duration_sec"]   = result["duration_beats"] * 60.0 / bpm

            # Constant-velocity check: all notes share the same velocity value
            velocities   = [n.velocity for n in all_notes]
            unique_vels  = set(velocities)
            if len(unique_vels) == 1:
                result["constant_velocity"]       = True
                result["constant_velocity_value"] = next(iter(unique_vels))

        result["success"] = True

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ─── Formatting helpers ───────────────────────────────────────────────────────
def _stats(values: list, unit: str = "") -> list:
    if not values:
        return ["  (no data)"]
    lines = [
        f"  Count:  {len(values):,}",
        f"  Mean:   {statistics.mean(values):.2f}{unit}",
        f"  Median: {statistics.median(values):.2f}{unit}",
        f"  Min:    {min(values):.2f}{unit}",
        f"  Max:    {max(values):.2f}{unit}",
    ]
    if len(values) > 1:
        lines.append(f"  StdDev: {statistics.stdev(values):.2f}{unit}")
    return lines


def _bar(count: int, total: int, width: int = 28) -> str:
    filled = int(width * count / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ─── Main analysis ─────────────────────────────────────────────────────────────
def run_analysis():
    output_lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        output_lines.append(msg)

    t0 = time.time()

    log("=" * 70)
    log("LAKH MIDI DATASET ANALYSIS")
    log(f"Dataset : {MIDI_DIR.resolve()}")
    log(f"Date    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # ── 1. Directory structure ───────────────────────────────────────────────
    log("\n[1] DIRECTORY STRUCTURE")
    log("-" * 70)

    if not MIDI_DIR.exists():
        log(f"  ERROR: dataset directory not found: {MIDI_DIR.resolve()}")
        return

    # lmd_full uses a two-level layout: lmd_full/{hex_char}/{md5}.mid
    top_subdirs = sorted(p for p in MIDI_DIR.iterdir() if p.is_dir())
    log(f"  Top-level subdirectories: {len(top_subdirs)}  ({', '.join(d.name for d in top_subdirs)})")

    all_midi: list[Path] = []
    subdir_counts: dict[str, int] = {}
    for sd in top_subdirs:
        files = list(sd.glob("*.mid")) + list(sd.glob("*.midi"))
        subdir_counts[sd.name] = len(files)
        all_midi.extend(files)

    total_files = len(all_midi)
    log(f"\n  Files per top-level subdir:")
    for name, cnt in sorted(subdir_counts.items()):
        log(f"    {name}/  {cnt:>8,} files")
    log(f"\n  TOTAL .mid/.midi files: {total_files:,}")

    # ── Sample ──────────────────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    sample_size = min(SAMPLE_SIZE, total_files)
    sample: list[Path] = random.sample(all_midi, sample_size)
    log(f"\n  Sampling {sample_size} random files (seed={RANDOM_SEED}) for analysis...")

    # ── 2. Parallel analysis ─────────────────────────────────────────────────
    log(f"\n[2] PARALLEL ANALYSIS  ({NUM_WORKERS} workers)")
    log("-" * 70)

    with Pool(NUM_WORKERS) as pool:
        results = pool.map(analyze_file, sample)

    elapsed = time.time() - t0
    log(f"  Done in {elapsed:.1f}s")

    good = [r for r in results if r["success"]]
    bad  = [r for r in results if not r["success"]]
    n_good = len(good)
    n_bad  = len(bad)

    log(f"  Read OK:  {n_good:4d} / {sample_size}")
    log(f"  Failed:   {n_bad:4d} / {sample_size}")

    # ── 3. Corrupted files ───────────────────────────────────────────────────
    log(f"\n[3] CORRUPTED / UNREADABLE FILES")
    log("-" * 70)

    if not bad:
        log("  None in sample — all files readable.")
    else:
        error_types = Counter()
        for r in bad:
            err_short = (r["error"] or "unknown error")[:60]
            error_types[err_short] += 1

        log(f"  {n_bad} file(s) failed.  Most common errors:")
        for err_msg, cnt in error_types.most_common(5):
            log(f"    x{cnt:3d}  {err_msg}")

        log(f"\n  First 10 failed files:")
        for r in bad[:10]:
            log(f"    {Path(r['path']).name:45s}  {(r['error'] or '')[:50]}")

        corrupt_rate = n_bad / sample_size
        est = int(corrupt_rate * total_files)
        log(f"\n  Estimated corrupt in full dataset: ~{est:,} ({100*corrupt_rate:.1f}%)")

    # ── 4. Constant-velocity detection ──────────────────────────────────────
    log(f"\n[4] CONSTANT VELOCITY FILES  (low-quality filter)")
    log("-" * 70)

    const_all  = [r for r in good if r["constant_velocity"]]
    const_111  = [r for r in const_all if r["constant_velocity_value"] == 111]
    n_const    = len(const_all)
    n_const111 = len(const_111)

    pct_all  = 100 * n_const    / max(1, n_good)
    pct_111  = 100 * n_const111 / max(1, n_good)

    log(f"  All notes same velocity:     {n_const:4d} / {n_good}  ({pct_all:.1f}%)")
    log(f"  ... of which velocity=111:   {n_const111:4d} / {n_good}  ({pct_111:.1f}%)")

    if const_all:
        vel_counter = Counter(r["constant_velocity_value"] for r in const_all)
        log(f"\n  Constant-velocity value breakdown (top 10):")
        for vel_val, cnt in vel_counter.most_common(10):
            pct = 100 * cnt / n_good
            log(f"    velocity={vel_val:3d}: {cnt:4d}  ({pct:.1f}%)  {_bar(cnt, n_good, 20)}")

        log(f"\n  Example files (first 5):")
        for r in const_all[:5]:
            vel = r["constant_velocity_value"]
            log(f"    vel={vel:3d}  {r['path'][-80:]}")

    est_const_total = int(n_const / max(1, n_good) * total_files)
    log(f"\n  Estimated constant-velocity files in full dataset: ~{est_const_total:,}")

    # ── 5. Distributions ────────────────────────────────────────────────────
    dur_sec     = [r["duration_sec"]    for r in good if r["duration_sec"]    > 0]
    dur_beats   = [r["duration_beats"]  for r in good if r["duration_beats"]  > 0]
    track_cnts  = [r["num_tracks"]      for r in good]
    note_cnts   = [r["num_notes"]       for r in good]
    tempos      = [r["tempo_bpm"]       for r in good if r["tempo_bpm"] is not None]
    time_sigs   = [r["time_signature"]  for r in good if r["time_signature"]  is not None]
    file_sizes  = [r["file_size_kb"]    for r in good]

    # 5a. Duration
    log(f"\n[5a] FILE DURATION")
    log("-" * 70)
    for line in _stats(dur_sec, "s"):
        log(line)
    log(f"\n  Buckets:")
    buckets = [
        ("<30s",       sum(1 for d in dur_sec if d < 30)),
        ("30s – 1min", sum(1 for d in dur_sec if 30  <= d < 60)),
        ("1 – 3min",   sum(1 for d in dur_sec if 60  <= d < 180)),
        ("3 – 5min",   sum(1 for d in dur_sec if 180 <= d < 300)),
        ("5 – 10min",  sum(1 for d in dur_sec if 300 <= d < 600)),
        (">10min",     sum(1 for d in dur_sec if d >= 600)),
    ]
    for label, cnt in buckets:
        pct = 100 * cnt / max(1, len(dur_sec))
        log(f"  {label:12s}: {cnt:4d}  ({pct:5.1f}%)  {_bar(cnt, len(dur_sec))}")

    # 5b. Track count
    log(f"\n[5b] NUMBER OF TRACKS PER FILE")
    log("-" * 70)
    for line in _stats([float(x) for x in track_cnts], " tracks"):
        log(line)

    track_dist = Counter(track_cnts)
    log(f"\n  Distribution:")
    for n_tracks, cnt in sorted(track_dist.items()):
        pct = 100 * cnt / max(1, len(track_cnts))
        log(f"  {n_tracks:3d} track(s): {cnt:4d}  ({pct:5.1f}%)  {_bar(cnt, len(track_cnts))}")

    has_drums_n = sum(1 for r in good if r["has_drums"])
    log(f"\n  Files with drum track: {has_drums_n}  ({100*has_drums_n/max(1,n_good):.1f}%)")

    # 5c. Tempo
    log(f"\n[5c] TEMPO  (BPM, first tempo event)")
    log("-" * 70)
    for line in _stats(tempos, " BPM"):
        log(line)

    log(f"\n  Buckets:")
    tempo_buckets = [
        ("Very Slow  <60",     [t for t in tempos if t < 60]),
        ("Slow    60-80",      [t for t in tempos if 60  <= t < 80]),
        ("Moderate 80-120",    [t for t in tempos if 80  <= t < 120]),
        ("Fast   120-160",     [t for t in tempos if 120 <= t < 160]),
        ("Very Fast >160",     [t for t in tempos if t >= 160]),
    ]
    for label, group in tempo_buckets:
        cnt = len(group)
        pct = 100 * cnt / max(1, len(tempos))
        log(f"  {label:18s}: {cnt:4d}  ({pct:5.1f}%)  {_bar(cnt, len(tempos))}")

    multi_tempo = sum(1 for r in good if r["num_tempo_changes"] > 1)
    log(f"\n  Files with >1 tempo change: {multi_tempo}  ({100*multi_tempo/max(1,n_good):.1f}%)")

    # 5d. Time signatures
    log(f"\n[5d] TIME SIGNATURES  (first event)")
    log("-" * 70)
    ts_counter = Counter(time_sigs)
    log(f"  Unique time signatures found: {len(ts_counter)}")
    log(f"\n  Top 10:")
    for (num, den), cnt in ts_counter.most_common(10):
        pct = 100 * cnt / max(1, len(time_sigs))
        log(f"  {num}/{den:2d}: {cnt:4d}  ({pct:5.1f}%)  {_bar(cnt, len(time_sigs))}")

    # 5e. Note count
    log(f"\n[5e] NOTE COUNT PER FILE")
    log("-" * 70)
    for line in _stats([float(x) for x in note_cnts], " notes"):
        log(line)

    empty_files = sum(1 for r in good if r["num_notes"] == 0)
    log(f"\n  Files with 0 notes: {empty_files}  ({100*empty_files/max(1,n_good):.1f}%)")

    # 5f. File size
    log(f"\n[5f] FILE SIZE")
    log("-" * 70)
    for line in _stats(file_sizes, " KB"):
        log(line)

    # ── Summary ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")
    log(f"  Total files in dataset   : {total_files:>10,}")
    log(f"  Sample analyzed          : {sample_size:>10,}")
    log(f"  Read OK                  : {n_good:>10,}  ({100*n_good/max(1,sample_size):.1f}%)")
    log(f"  Corrupted / unreadable   : {n_bad:>10,}  ({100*n_bad/max(1,sample_size):.1f}%)")
    log(f"  Constant velocity (any)  : {n_const:>10,}  ({100*n_const/max(1,n_good):.1f}% of readable)")
    log(f"  Constant velocity=111    : {n_const111:>10,}  ({100*n_const111/max(1,n_good):.1f}% of readable)")
    if dur_sec:
        log(f"  Avg duration             : {statistics.mean(dur_sec):>9.1f}s")
    if track_cnts:
        log(f"  Avg tracks per file      : {statistics.mean([float(x) for x in track_cnts]):>9.1f}")
    if tempos:
        log(f"  Avg tempo                : {statistics.mean(tempos):>9.1f} BPM")
    log(f"\n  Analysis time  : {total_elapsed:.1f}s")
    log(f"  Report saved   : {REPORT_PATH.resolve()}")
    log("=" * 70)

    # ── Save report ─────────────────────────────────────────────────────────
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(output_lines) + "\n")

    print(f"\nReport written to: {REPORT_PATH.resolve()}")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Guard required on Windows (spawn-based multiprocessing)
    try:
        import miditoolkit  # noqa: F401  — fail early with a clear message
    except ImportError:
        print("ERROR: miditoolkit not found.\n  Install with:  pip install miditoolkit")
        sys.exit(1)

    run_analysis()
