"""
Program / Instrument Distribution Analysis -- MIDIT REMI+ Tokenizer
====================================================================
Samples 300 random files from data/midi_files/lmd_full/ and tokenizes
each with the EXACT V1 config from preprocess_batch_improved.py.

In the REMI+ stream each note produces exactly one Program_X token,
so counting Program_X occurrences == note count for instrument X.

Reports:
  1. Total notes and per-family distribution (% of all notes)
  2. GM instrument family breakdown with bar charts
  3. Flags: any family > 40% of notes (dataset skew)
  4. Drums-only files, single-program files
  5. Top 10 most common individual Program values
  6. Average unique programs per file + distribution

Uses multiprocessing with 4 workers.
Saves report to logs/program_distribution_analysis.txt

Run from MIDIT project root:
  python scripts/Analyze_dataset/analyze_program_distribution.py

Requirements: miditok  (pip install miditok)
"""

import random
import statistics
import sys
import time
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
MIDI_DIR    = Path("data/midi_files/lmd_full")
LOGS_DIR    = Path("logs")
REPORT_PATH = LOGS_DIR / "program_distribution_analysis.txt"

# ─── Constants ────────────────────────────────────────────────────────────────
SAMPLE_SIZE        = 300
NUM_WORKERS        = 4
RANDOM_SEED        = 42
FAMILY_FLAG_PCT    = 40.0   # flag if any GM family exceeds this % of all notes

# ─── GM instrument family boundaries ─────────────────────────────────────────
FAMILY_ORDER = [
    "Drums", "Piano", "Chromatic Perc", "Organ", "Guitar",
    "Bass", "Strings", "Ensemble", "Brass", "Reed",
    "Pipe", "Synth Lead", "Synth Pad", "Other",
]


def _get_family(prog: int) -> str:
    if prog == -1: return "Drums"
    if prog <=  7: return "Piano"
    if prog <= 15: return "Chromatic Perc"
    if prog <= 23: return "Organ"
    if prog <= 31: return "Guitar"
    if prog <= 39: return "Bass"
    if prog <= 47: return "Strings"
    if prog <= 55: return "Ensemble"
    if prog <= 63: return "Brass"
    if prog <= 71: return "Reed"
    if prog <= 79: return "Pipe"
    if prog <= 87: return "Synth Lead"
    if prog <= 95: return "Synth Pad"
    return "Other"


# ─── GM instrument names (abridged; used for Top-10 display) ─────────────────
GM_NAMES = {
    -1:  "Drums/Percussion",
     0:  "Acoustic Grand Piano",   1:  "Bright Acoustic Piano",
     2:  "Electric Grand Piano",   3:  "Honky-tonk Piano",
     4:  "Electric Piano 1",       5:  "Electric Piano 2",
     6:  "Harpsichord",            7:  "Clavinet",
     8:  "Celesta",                9:  "Glockenspiel",
    10:  "Music Box",             11:  "Vibraphone",
    12:  "Marimba",               13:  "Xylophone",
    14:  "Tubular Bells",         15:  "Dulcimer",
    16:  "Drawbar Organ",         17:  "Percussive Organ",
    18:  "Rock Organ",            19:  "Church Organ",
    24:  "Nylon Guitar",          25:  "Steel Guitar",
    26:  "Jazz Guitar",           27:  "Clean Guitar",
    28:  "Muted Guitar",          29:  "Overdriven Guitar",
    30:  "Distortion Guitar",     31:  "Guitar Harmonics",
    32:  "Acoustic Bass",         33:  "Finger Bass",
    34:  "Pick Bass",             35:  "Fretless Bass",
    36:  "Slap Bass 1",           37:  "Slap Bass 2",
    38:  "Synth Bass 1",          39:  "Synth Bass 2",
    40:  "Violin",                41:  "Viola",
    42:  "Cello",                 43:  "Contrabass",
    44:  "Tremolo Strings",       45:  "Pizzicato Strings",
    46:  "Orchestral Harp",       47:  "Timpani",
    48:  "String Ensemble 1",     49:  "String Ensemble 2",
    50:  "Synth Strings 1",       51:  "Synth Strings 2",
    52:  "Choir Aahs",            53:  "Voice Oohs",
    54:  "Synth Voice",           55:  "Orchestra Hit",
    56:  "Trumpet",               57:  "Trombone",
    58:  "Tuba",                  59:  "Muted Trumpet",
    60:  "French Horn",           61:  "Brass Section",
    62:  "Synth Brass 1",         63:  "Synth Brass 2",
    64:  "Soprano Sax",           65:  "Alto Sax",
    66:  "Tenor Sax",             67:  "Baritone Sax",
    68:  "Oboe",                  69:  "English Horn",
    70:  "Bassoon",               71:  "Clarinet",
    72:  "Piccolo",               73:  "Flute",
    74:  "Recorder",              75:  "Pan Flute",
    76:  "Blown Bottle",          77:  "Shakuhachi",
    78:  "Whistle",               79:  "Ocarina",
    80:  "Square Lead",           81:  "Sawtooth Lead",
    82:  "Calliope Lead",         83:  "Chiff Lead",
    84:  "Charang Lead",          85:  "Voice Lead",
    86:  "Fifths Lead",           87:  "Bass+Lead",
    88:  "New Age Pad",           89:  "Warm Pad",
    90:  "Polysynth Pad",         91:  "Choir Pad",
    92:  "Bowed Pad",             93:  "Metallic Pad",
    94:  "Halo Pad",              95:  "Sweep Pad",
}


def _gm_name(prog: int) -> str:
    return GM_NAMES.get(prog, f"Program {prog}")


# ─── Worker initializer ── builds tokenizer + program ID lookup once per worker
_tokenizer    = None
_id_to_prog   = None   # token_id -> program_value  (Program_X tokens only)


def _init_worker():
    global _tokenizer, _id_to_prog
    from miditok import REMI, TokenizerConfig

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
    _tokenizer  = REMI(config)
    _id_to_prog = {}
    for tok_name, tok_id in _tokenizer.vocab.items():
        if tok_name.startswith("Program_"):
            try:
                _id_to_prog[tok_id] = int(tok_name[8:])
            except ValueError:
                pass


# ─── Per-file analysis (runs in worker processes) ─────────────────────────────
def _analyze_file(midi_path: Path) -> dict:
    """Tokenize one file and count notes per Program value."""
    result = {
        "path":            str(midi_path),
        "success":         False,
        "error":           None,
        "prog_counts":     {},   # program_int -> note_count
        "total_notes":     0,
        "unique_programs": 0,
        "has_drums":       False,
        "drums_only":      False,
        "no_programs":     False,
    }
    try:
        global _tokenizer, _id_to_prog

        tokens = _tokenizer(str(midi_path))
        if isinstance(tokens, list):
            ids = []
            for seq in tokens:
                ids.extend(seq.ids)
        else:
            ids = tokens.ids

        prog_counts: Counter = Counter()
        for tok_id in ids:
            prog = _id_to_prog.get(tok_id)
            if prog is not None:
                prog_counts[prog] += 1

        total      = sum(prog_counts.values())
        unique     = len(prog_counts)
        has_drums  = -1 in prog_counts
        drums_only = (unique >= 1 and set(prog_counts.keys()) == {-1})
        no_progs   = (unique == 0)

        result.update({
            "success":         True,
            "prog_counts":     dict(prog_counts),
            "total_notes":     total,
            "unique_programs": unique,
            "has_drums":       has_drums,
            "drums_only":      drums_only,
            "no_programs":     no_progs,
        })

    except Exception as exc:
        result["error"] = str(exc)
    return result


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _bar(count: int, total: int, width: int = 30) -> str:
    filled = int(width * count / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


# ─── Main analysis ─────────────────────────────────────────────────────────────
def run_analysis():
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(str(msg))

    t0 = time.time()

    log("=" * 72)
    log("PROGRAM / INSTRUMENT DISTRIBUTION ANALYSIS -- MIDIT REMI+ V1 CONFIG")
    log(f"Dataset  : {MIDI_DIR.resolve()}")
    log(f"Date     : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 72)

    # ── Dataset scan ─────────────────────────────────────────────────────────
    if not MIDI_DIR.exists():
        log(f"ERROR: {MIDI_DIR.resolve()} not found.")
        return

    top_subdirs = sorted(p for p in MIDI_DIR.iterdir() if p.is_dir())
    all_midi = []
    for sd in top_subdirs:
        all_midi.extend(sd.glob("*.mid"))
        all_midi.extend(sd.glob("*.midi"))

    total_dataset_files = len(all_midi)
    log(f"\n  Total files in dataset : {total_dataset_files:,}")

    random.seed(RANDOM_SEED)
    sample = random.sample(all_midi, min(SAMPLE_SIZE, total_dataset_files))
    log(f"  Sample size            : {len(sample)}  (seed={RANDOM_SEED})")

    # ── Parallel analysis ────────────────────────────────────────────────────
    log(f"\n[1] TOKENIZING {len(sample)} FILES  ({NUM_WORKERS} workers)...")
    log("-" * 72)

    with Pool(NUM_WORKERS, initializer=_init_worker) as pool:
        results = pool.map(_analyze_file, sample)

    elapsed = time.time() - t0
    good = [r for r in results if r["success"]]
    bad  = [r for r in results if not r["success"]]
    log(f"  Done in {elapsed:.1f}s  ({elapsed / len(sample):.2f}s/file)")
    log(f"  OK     : {len(good):4d} / {len(sample)}")
    log(f"  Failed : {len(bad):4d} / {len(sample)}")
    if bad:
        err_types = Counter((r["error"] or "unknown")[:60] for r in bad)
        for msg, cnt in err_types.most_common(3):
            log(f"    x{cnt}  {msg}")

    if not good:
        log("ERROR: no files processed.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total_notes = sum(r["total_notes"] for r in good)
    agg_prog    = Counter()
    for r in good:
        agg_prog.update(r["prog_counts"])

    # Family totals
    agg_family = Counter()
    for prog, count in agg_prog.items():
        agg_family[_get_family(prog)] += count

    # ── [2] GM family distribution ────────────────────────────────────────────
    log(f"\n[2] GM INSTRUMENT FAMILY DISTRIBUTION")
    log("-" * 72)
    log(f"\n  Total notes analyzed : {total_notes:,}  across {len(good)} files")
    log(f"  (One Program token per note in REMI+ stream)")
    log()

    log(f"  {'Family':<18}  {'Notes':>10}  {'%':>7}  {'Bar chart':<32}  Flag")
    log("  " + "-" * 72)

    for family in FAMILY_ORDER:
        cnt  = agg_family.get(family, 0)
        pct  = _pct(cnt, total_notes)
        bar  = _bar(cnt, total_notes, 28)
        flag = "  *** > 40% ***" if pct >= FAMILY_FLAG_PCT else ""
        log(f"  {family:<18}  {cnt:>10,}  {pct:>6.2f}%  {bar}  {flag}")

    log()
    # Verify totals add up
    family_total = sum(agg_family.values())
    log(f"  Sum check: {family_total:,} notes across all families (should equal {total_notes:,})")

    # ── [3] Flags ─────────────────────────────────────────────────────────────
    log(f"\n[3] FLAGS")
    log("-" * 72)

    dominant_families = [(f, agg_family.get(f, 0))
                         for f in FAMILY_ORDER
                         if _pct(agg_family.get(f, 0), total_notes) >= FAMILY_FLAG_PCT]

    if dominant_families:
        log(f"\n  *** DOMINANT FAMILY WARNING ***")
        for fam, cnt in dominant_families:
            pct = _pct(cnt, total_notes)
            log(f"  [{fam}] accounts for {pct:.1f}% of all notes.")
            log(f"  This family dominates the training distribution.")
            if fam == "Piano":
                log(f"  The Lakh dataset is heavily piano-biased -- expected.")
            elif fam == "Drums":
                log(f"  Drums-heavy dataset. Consider filtering drums-only files.")
    else:
        log(f"\n  No single GM family exceeds {FAMILY_FLAG_PCT}% of notes.")
        log(f"  Dataset appears reasonably balanced across instrument families.")

    # Top-N families (sorted)
    log(f"\n  Top 5 families by note count:")
    for fam, cnt in sorted(agg_family.items(), key=lambda x: -x[1])[:5]:
        pct = _pct(cnt, total_notes)
        log(f"    {fam:<18}  {cnt:>10,}  ({pct:.1f}%)")

    # ── [4] Single-program and drums-only files ───────────────────────────────
    log(f"\n[4] SINGLE-PROGRAM AND DRUMS-ONLY FILES")
    log("-" * 72)

    drums_only_files  = [r for r in good if r["drums_only"]]
    has_drums_files   = [r for r in good if r["has_drums"]]
    no_drums_files    = [r for r in good if not r["has_drums"]]
    no_program_files  = [r for r in good if r["no_programs"]]
    single_prog_files = [r for r in good if r["unique_programs"] == 1]

    log(f"\n  Files with drums      : {len(has_drums_files):4d} / {len(good)}"
        f"  ({_pct(len(has_drums_files), len(good)):.1f}%)")
    log(f"  Files without drums   : {len(no_drums_files):4d} / {len(good)}"
        f"  ({_pct(len(no_drums_files), len(good)):.1f}%)")
    log(f"  Drums-only files      : {len(drums_only_files):4d} / {len(good)}"
        f"  ({_pct(len(drums_only_files), len(good)):.1f}%)  (all notes = Program_-1)")
    log(f"  Single-program files  : {len(single_prog_files):4d} / {len(good)}"
        f"  ({_pct(len(single_prog_files), len(good)):.1f}%)  (1 unique instrument)")
    log(f"  No program tokens     : {len(no_program_files):4d} / {len(good)}"
        f"  ({_pct(len(no_program_files), len(good)):.1f}%)  (empty or unreadable)")

    # Breakdown of single-program files by program value
    if single_prog_files:
        prog_of_single = Counter()
        for r in single_prog_files:
            progs = list(r["prog_counts"].keys())
            if progs:
                prog_of_single[progs[0]] += 1

        log(f"\n  Single-program files breakdown (top 10 programs):")
        log(f"  {'Program':>8}  {'Name':<28}  {'Family':<18}  {'Files':>6}")
        log("  " + "-" * 64)
        for prog, cnt in prog_of_single.most_common(10):
            name   = _gm_name(prog)
            family = _get_family(prog)
            log(f"  {prog:>8}  {name:<28}  {family:<18}  {cnt:>6}")

    # Drums-only: estimate for full dataset
    drums_only_rate = _pct(len(drums_only_files), len(good))
    est_drums_only  = int(drums_only_rate / 100.0 * total_dataset_files)
    log(f"\n  Estimated drums-only files in full dataset: ~{est_drums_only:,}"
        f"  ({drums_only_rate:.1f}%)")

    # ── [5] Top 10 individual Program values ──────────────────────────────────
    log(f"\n[5] TOP 10 INDIVIDUAL PROGRAM VALUES  (by note count)")
    log("-" * 72)
    log()
    log(f"  {'Rank':<5}  {'Prog':>5}  {'Name':<28}  {'Family':<18}  "
        f"{'Notes':>10}  {'%':>7}  {'Files':>7}")
    log("  " + "-" * 80)

    for rank, (prog, cnt) in enumerate(agg_prog.most_common(10), 1):
        name     = _gm_name(prog)
        family   = _get_family(prog)
        pct      = _pct(cnt, total_notes)
        # files that contain this program
        files_with = sum(1 for r in good if prog in r["prog_counts"])
        log(f"  {rank:<5}  {prog:>5}  {name:<28}  {family:<18}  "
            f"{cnt:>10,}  {pct:>6.2f}%  {files_with:>5}/{len(good)}")

    # Programs present but rare
    total_unique_progs = len(agg_prog)
    rare_progs = [(p, c) for p, c in agg_prog.items()
                  if _pct(c, total_notes) < 0.1]
    log(f"\n  Total unique Program values seen : {total_unique_progs}")
    log(f"  Programs with < 0.1% of notes   : {len(rare_progs)}")
    if rare_progs:
        log(f"  (These instruments are very underrepresented in the sample)")

    # ── [6] Unique programs per file distribution ─────────────────────────────
    log(f"\n[6] UNIQUE PROGRAMS PER FILE")
    log("-" * 72)

    unique_counts = [r["unique_programs"] for r in good]
    log(f"\n  Mean   : {statistics.mean(unique_counts):.1f}")
    log(f"  Median : {statistics.median(unique_counts):.1f}")
    log(f"  Min    : {min(unique_counts)}")
    log(f"  Max    : {max(unique_counts)}")
    if len(unique_counts) > 1:
        log(f"  StdDev : {statistics.stdev(unique_counts):.1f}")

    log()
    dist = Counter(unique_counts)
    # Show up to 15 most common values
    log(f"  {'# programs':>12}  {'Files':>6}  {'%':>7}  Bar chart")
    log("  " + "-" * 52)
    for n_progs, cnt in sorted(dist.items()):
        pct  = _pct(cnt, len(good))
        bar  = _bar(cnt, len(good), 22)
        note = "  <- drums-only" if n_progs == 1 and any(
            r["drums_only"] for r in good if r["unique_programs"] == 1
        ) else ""
        log(f"  {n_progs:>12}  {cnt:>6}  {pct:>6.1f}%  {bar}{note}")
        if n_progs >= 20:
            remaining = sum(c for k, c in dist.items() if k > 20)
            if remaining:
                log(f"  {'> 20':>12}  {remaining:>6}  "
                    f"{_pct(remaining, len(good)):>6.1f}%")
            break

    # ── [7] Family co-occurrence heatmap (simplified) ─────────────────────────
    log(f"\n[7] FAMILY CO-OCCURRENCE  (how often families appear together)")
    log("-" * 72)
    log(f"  (% of files that contain at least one note from each family pair)")
    log()

    # Only show the top 6 most common families to keep output readable
    top_families = [f for f, _ in
                    sorted(agg_family.items(), key=lambda x: -x[1])[:6]]

    # File-level family presence
    file_families = []
    for r in good:
        present = set()
        for prog, cnt in r["prog_counts"].items():
            if cnt > 0:
                present.add(_get_family(prog))
        file_families.append(present)

    header = f"  {'Family':<18}"
    for f in top_families:
        header += f"  {f[:10]:>10}"
    log(header)
    log("  " + "-" * (18 + 12 * len(top_families)))

    for fa in top_families:
        row = f"  {fa:<18}"
        for fb in top_families:
            both = sum(1 for ff in file_families if fa in ff and fb in ff)
            pct  = _pct(both, len(good))
            row += f"  {pct:>9.1f}%"
        log(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0

    log(f"\n{'=' * 72}")
    log("SUMMARY")
    log(f"{'=' * 72}")
    log(f"  Files analyzed              : {len(good):>8,}")
    log(f"  Total notes                 : {total_notes:>8,}")
    log(f"  Avg notes/file              : {total_notes/len(good):>8,.0f}")
    log(f"  Unique Program values seen  : {total_unique_progs:>8}")
    log(f"  Avg unique programs/file    : {statistics.mean(unique_counts):>8.1f}")
    log()

    log(f"  Top 3 families by note count:")
    for fam, cnt in sorted(agg_family.items(), key=lambda x: -x[1])[:3]:
        pct = _pct(cnt, total_notes)
        flag = "  *** DOMINANT ***" if pct >= FAMILY_FLAG_PCT else ""
        log(f"    {fam:<18}  {pct:.1f}%{flag}")

    log()
    log(f"  Drums-only files            : {len(drums_only_files):>3} / {len(good)}"
        f"  ({_pct(len(drums_only_files), len(good)):.1f}%)")
    log(f"  Single-program files        : {len(single_prog_files):>3} / {len(good)}"
        f"  ({_pct(len(single_prog_files), len(good)):.1f}%)")
    log(f"  Files with drums            : {len(has_drums_files):>3} / {len(good)}"
        f"  ({_pct(len(has_drums_files), len(good)):.1f}%)")

    log()
    log(f"  Class imbalance (notes): "
        f"most={max(agg_family.values()):,}  least={min(agg_family.values()):,}"
        f"  ratio={max(agg_family.values())/max(1, min(agg_family.values())):.0f}x")

    if dominant_families:
        log(f"\n  *** DOMINANT FAMILY: "
            f"{', '.join(f[0] for f in dominant_families)} ***")

    log(f"\n  Analysis time  : {total_elapsed:.1f}s")
    log(f"  Report saved   : {REPORT_PATH.resolve()}")
    log("=" * 72)

    # Save
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(output_lines) + "\n")

    print(f"\nReport written to: {REPORT_PATH.resolve()}")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from miditok import REMI  # noqa
    except ImportError:
        print("ERROR: miditok not found. Install with: pip install miditok")
        sys.exit(1)
    run_analysis()
