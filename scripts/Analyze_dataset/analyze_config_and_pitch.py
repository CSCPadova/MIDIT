"""
Pre-Preprocessing Analysis:
  B -- Config v2 Vocab Impact (runs first)
  A -- Pitch Shift Validation for Augmentation

Saves:
  logs/config_v2_analysis.txt
  logs/pitch_shift_analysis.txt

Run from MIDIT project root:
  python scripts/Analyze_dataset/analyze_config_and_pitch.py

Requirements: miditok, miditoolkit
"""

import inspect
import random
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

warnings.filterwarnings("ignore")   # suppress miditok UserWarnings

# ─── Paths ────────────────────────────────────────────────────────────────────
MIDI_DIR = Path("data/midi_files/lmd_full")
LOGS_DIR = Path("logs")
REPORT_B = LOGS_DIR / "config_v2_analysis.txt"
REPORT_A = LOGS_DIR / "pitch_shift_analysis.txt"

# ─── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
CONFIG_SAMPLE  = 10    # files tokenized for B6 comparison
PITCH_SAMPLE   = 200   # files for pitch analysis
NUM_WORKERS    = 4
PITCH_MIN      = 21    # tokenizer pitch_range lower bound (inclusive)
PITCH_MAX      = 108   # tokenizer pitch_range upper bound (inclusive; config: 109 exclusive)
SHIFTS         = [-5, -4, -3, -2, -1, +1, +2, +3, +4, +5]

# ─── V1 config ── EXACT copy from preprocess_batch_improved.py ───────────────
V1_PARAMS = dict(
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

# ─── V2 config ── proposed in CLAUDE.md Task 1 ───────────────────────────────
# NOTE: max_bar_embedding is NOT a TokenizerConfig parameter in miditok 3.x.
#       It was available in older versions. This script will flag it accordingly.
V2_PARAMS_FULL = dict(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8, (4, 12): 4},          # extended range -- adds Duration tokens
    num_velocities=32,
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_programs=True,
    special_tokens=["PAD", "BOS", "EOS", "MASK"],
    use_controls=False,
    max_bar_embedding=8,                         # from CLAUDE.md -- may not be supported
    delete_equal_successive_tempo_changes=True,  # from CLAUDE.md -- reduces redundant tokens
    delete_equal_successive_time_sig_changes=True,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _tok_type(name: str) -> str:
    return name.split("_")[0] if "_" in name else name


def _bar_chart(count: int, total: int, width: int = 28) -> str:
    filled = int(width * count / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def _get_ids(tokenizer, midi_path: Path) -> list:
    tokens = tokenizer(str(midi_path))
    if isinstance(tokens, list):
        ids = []
        for seq in tokens:
            ids.extend(seq.ids)
        return ids
    return tokens.ids


# ─── Worker for pitch extraction (Analysis A) -- must be module-level ─────────
def _extract_pitches(midi_path: Path) -> dict:
    """Extract raw non-drum MIDI pitches from one file."""
    result = {
        "path":        str(midi_path),
        "success":     False,
        "error":       None,
        "pitches":     [],
        "total_notes": 0,
    }
    try:
        import miditoolkit
        midi = miditoolkit.MidiFile(str(midi_path))
        pitches = []
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                pitches.append(note.pitch)
        result["pitches"]     = pitches
        result["total_notes"] = len(pitches)
        result["success"]     = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


# ─── Analysis B ───────────────────────────────────────────────────────────────
def run_analysis_b(config_files: list) -> list:
    from miditok import REMI, TokenizerConfig
    import importlib.metadata

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("ANALYSIS B -- CONFIG V2 VOCAB IMPACT")
    log(f"Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        ver = importlib.metadata.version("miditok")
    except Exception:
        ver = "unknown"
    log(f"MidiTok   : {ver}")
    log("=" * 70)

    # ── [B1] Parameter support check ─────────────────────────────────────────
    log("\n[B1] V2 PARAMETER SUPPORT CHECK")
    log("-" * 70)

    sig           = inspect.signature(TokenizerConfig.__init__)
    supported_set = set(sig.parameters.keys()) - {"self", "kwargs"}

    v2_new_params = {k: v for k, v in V2_PARAMS_FULL.items() if k not in V1_PARAMS}
    log(f"  {'Parameter':<52}  Value                   Status")
    log("  " + "-" * 68)

    v2_applicable = {}
    v2_unsupported = {}
    for param, value in v2_new_params.items():
        if param in supported_set:
            status = "SUPPORTED"
            v2_applicable[param] = value
        else:
            status = "NOT SUPPORTED in miditok " + ver
            v2_unsupported[param] = value
        log(f"  {param:<52}  {str(value):<22}  [{status}]")

    if v2_unsupported:
        log(f"\n  NOTE: {len(v2_unsupported)} param(s) from CLAUDE.md are absent from")
        log(f"  TokenizerConfig in miditok {ver}:")
        for k, v in v2_unsupported.items():
            log(f"    - {k} = {v}")
            if k == "max_bar_embedding":
                log(f"      (This param existed in older miditok versions. In v3.x the")
                log(f"       bar embedding is handled differently. Will be skipped.)")

    # Build the V2 config we can actually create
    v2_buildable = {k: v for k, v in V2_PARAMS_FULL.items()
                    if k in supported_set or k in V1_PARAMS}

    # ── [B2] Create both tokenizers ───────────────────────────────────────────
    log(f"\n[B2] TOKENIZER CREATION")
    log("-" * 70)

    tok_v1 = REMI(TokenizerConfig(**V1_PARAMS))
    vocab_v1 = len(tok_v1.vocab)
    log(f"  V1 tokenizer : OK  (vocab = {vocab_v1})")

    try:
        tok_v2 = REMI(TokenizerConfig(**v2_buildable))
        vocab_v2 = len(tok_v2.vocab)
        log(f"  V2 tokenizer : OK  (vocab = {vocab_v2})")
        v2_ok = True
    except Exception as exc:
        log(f"  V2 tokenizer : FAILED -- {exc}")
        tok_v2 = None
        vocab_v2 = None
        v2_ok = False

    # ── [B3] Vocab size comparison ────────────────────────────────────────────
    log(f"\n[B3] VOCAB SIZE COMPARISON")
    log("-" * 70)
    log(f"  V1 vocab size : {vocab_v1}")

    if v2_ok:
        delta = vocab_v2 - vocab_v1
        log(f"  V2 vocab size : {vocab_v2}")
        log(f"  Delta         : {delta:+d} tokens")

        if delta != 0:
            log(f"\n  {'*' * 64}")
            log(f"  * CRITICAL: VOCAB SIZE CHANGED BY {delta:+d} TOKENS                  *")
            log(f"  *                                                              *")
            log(f"  * Current model embedding layer expects exactly {vocab_v1} tokens    *")
            log(f"  * V2 config produces {vocab_v2} tokens -- shapes do NOT match      *")
            log(f"  *                                                              *")
            log(f"  * Consequence: V2 preprocessing is INCOMPATIBLE with the      *")
            log(f"  * current trained model (Run 3, epoch 48).                    *")
            log(f"  * You MUST train from scratch if you adopt the V2 config.     *")
            log(f"  {'*' * 64}")
        else:
            log(f"\n  SAFE: Vocab size unchanged. V2 config is compatible with the")
            log(f"  current model.")

    # ── [B4] Token type breakdown ─────────────────────────────────────────────
    log(f"\n[B4] VOCAB BREAKDOWN BY TOKEN TYPE")
    log("-" * 70)

    types_v1 = Counter(_tok_type(k) for k in tok_v1.vocab)
    header = f"\n  {'Type':<18}  {'V1':>5}"
    if v2_ok:
        header += f"  {'V2':>5}  {'Delta':>6}  Note"
    log(header)
    log("  " + "-" * (50 if v2_ok else 26))

    if v2_ok:
        types_v2 = Counter(_tok_type(k) for k in tok_v2.vocab)
        all_types = sorted(set(types_v1) | set(types_v2))
        for t in all_types:
            c1, c2 = types_v1.get(t, 0), types_v2.get(t, 0)
            d = c2 - c1
            flag = " *** CHANGED ***" if d != 0 else ""
            log(f"  {t:<18}  {c1:>5}  {c2:>5}  {d:>+6}{flag}")
    else:
        for t, c in sorted(types_v1.items()):
            log(f"  {t:<18}  {c:>5}")

    # ── [B5] Token-level diff ─────────────────────────────────────────────────
    if v2_ok and delta != 0:
        log(f"\n[B5] WHICH TOKENS WERE ADDED / REMOVED")
        log("-" * 70)

        keys_v1 = set(tok_v1.vocab)
        keys_v2 = set(tok_v2.vocab)
        added   = keys_v2 - keys_v1
        removed = keys_v1 - keys_v2

        by_type_added   = defaultdict(list)
        by_type_removed = defaultdict(list)
        for k in sorted(added):
            by_type_added[_tok_type(k)].append(k)
        for k in sorted(removed):
            by_type_removed[_tok_type(k)].append(k)

        if by_type_added:
            log(f"\n  ADDED ({len(added)} tokens):")
            for ttype, tokens in sorted(by_type_added.items()):
                sample = tokens[:6]
                ellipsis = f" ... ({len(tokens)} total)" if len(tokens) > 6 else ""
                log(f"    [{ttype}]  {', '.join(sample)}{ellipsis}")
        else:
            log(f"  ADDED: none")

        if by_type_removed:
            log(f"\n  REMOVED ({len(removed)} tokens):")
            for ttype, tokens in sorted(by_type_removed.items()):
                sample = tokens[:6]
                ellipsis = f" ... ({len(tokens)} total)" if len(tokens) > 6 else ""
                log(f"    [{ttype}]  {', '.join(sample)}{ellipsis}")
        else:
            log(f"  REMOVED: none")

        log(f"\n  Root cause of vocab change:")
        log(f"    beat_res {{(0,4):8}} -> {{(0,4):8, (4,12):4}}")
        log(f"    The (4,12):4 range adds Duration tokens for notes longer than")
        log(f"    4 beats (e.g. Duration_4.1.4 through Duration_11.3.4).")
        log(f"    These represent note durations from 4+ beats up to 12 beats.")
        log(f"    Most MIDI notes are shorter -- these tokens will be rare in practice.")

    elif v2_ok:
        log(f"\n[B5] TOKEN-LEVEL DIFF: no changes (vocab identical)")

    # ── [B6] File-level tokenization comparison ───────────────────────────────
    log(f"\n[B6] FILE-LEVEL TOKENIZATION COMPARISON  ({len(config_files)} files)")
    log("-" * 70)

    if v2_ok:
        log(f"\n  {'File':<42}  {'V1':>7}  {'V2':>7}  {'Delta':>7}  {'Delta%':>7}")
        log("  " + "-" * 70)
    else:
        log(f"\n  {'File':<42}  {'V1 tokens':>10}")
        log("  " + "-" * 55)

    deltas = []
    for midi_path in config_files:
        try:
            ids_v1 = _get_ids(tok_v1, midi_path)
            n_v1   = len(ids_v1)
            if v2_ok:
                ids_v2 = _get_ids(tok_v2, midi_path)
                n_v2   = len(ids_v2)
                d      = n_v2 - n_v1
                dpct   = _pct(d, n_v1)
                deltas.append(d)
                log(f"  {midi_path.name[:42]:<42}  {n_v1:>7,}  {n_v2:>7,}  {d:>+7,}  {dpct:>+6.1f}%")
            else:
                log(f"  {midi_path.name[:42]:<42}  {n_v1:>10,}")
        except Exception as exc:
            log(f"  {midi_path.name[:42]:<42}  ERROR: {str(exc)[:45]}")

    if deltas:
        log(f"\n  Token count delta across {len(deltas)} files:")
        log(f"    Mean   : {statistics.mean(deltas):>+.1f}")
        log(f"    Median : {statistics.median(deltas):>+.1f}")
        log(f"    Min    : {min(deltas):>+,}")
        log(f"    Max    : {max(deltas):>+,}")
        log(f"\n  Note: delta=0 for most files means their notes have durations")
        log(f"  <= 4 beats, so the new Duration tokens are not triggered.")
        log(f"  Files with long notes (held > 4 beats) will produce different")
        log(f"  Duration tokens under V2 (and these are NEW vocab IDs).")

    # ── [B7] Summary and decision guide ──────────────────────────────────────
    log(f"\n{'=' * 70}")
    log("ANALYSIS B -- SUMMARY & DECISION GUIDE")
    log(f"{'=' * 70}")
    log(f"\n  V1 vocab size              : {vocab_v1} tokens")
    if v2_ok:
        log(f"  V2 vocab size              : {vocab_v2} tokens")
        log(f"  Vocab delta                : {vocab_v2 - vocab_v1:+d} tokens")
    log(f"\n  Params applied from v2     : {sorted(v2_applicable)}")
    log(f"  Params skipped (unsupported): {sorted(v2_unsupported)}")

    log(f"\n  COMPATIBILITY:")
    if v2_ok and vocab_v2 != vocab_v1:
        log(f"    INCOMPATIBLE -- vocab changed.")
        log(f"    The extended beat_res adds {vocab_v2 - vocab_v1} Duration tokens.")
        log(f"    These tokens have NEW IDs that the current model has never seen.")
        log(f"")
        log(f"  OPTIONS:")
        log(f"    A) Keep V1 beat_res={{(0,4):8}} -- use only delete_equal_successive_*")
        log(f"       -> Vocab stays at {vocab_v1}. Fine-tune from Run 3 checkpoint.")
        log(f"    B) Use full V2 config -- vocab becomes {vocab_v2}.")
        log(f"       -> Must train from scratch. Cannot reuse Run 3 weights.")
        log(f"       -> V2 can represent longer durations more accurately.")
        log(f"")
        log(f"  max_bar_embedding=8 NOTE:")
        log(f"    This param does NOT exist in miditok {ver}.")
        log(f"    Bar tokens remain ['Bar_None'] (1 token) regardless.")
        log(f"    Positional bar info within a chunk is not encoded by the tokenizer.")
        log(f"    (Could be added via absolute position embedding in the model instead.)")
    elif v2_ok:
        log(f"    SAFE -- vocab unchanged. Fine-tuning from Run 3 is possible.")
    log("=" * 70)

    return lines


# ─── Analysis A ───────────────────────────────────────────────────────────────
def run_analysis_a(all_midi: list) -> list:
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("\n" + "=" * 70)
    log("ANALYSIS A -- PITCH RANGE VALIDATION FOR AUGMENTATION")
    log(f"Date          : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Pitch range   : [{PITCH_MIN}, {PITCH_MAX}]  (tokenizer pitch_range=(21,109))")
    log(f"Shifts tested : {SHIFTS}")
    log("=" * 70)

    # Sample
    random.seed(RANDOM_SEED)
    sample = random.sample(all_midi, min(PITCH_SAMPLE, len(all_midi)))
    log(f"\n  Sampling {len(sample)} files  (seed={RANDOM_SEED})")

    # ── [A1] Parallel pitch extraction ───────────────────────────────────────
    log(f"\n[A1] EXTRACTING RAW MIDI PITCHES  ({NUM_WORKERS} workers)")
    log("-" * 70)

    with Pool(NUM_WORKERS) as pool:
        results = pool.map(_extract_pitches, sample)

    good = [r for r in results if r["success"]]
    bad  = [r for r in results if not r["success"]]
    log(f"  Read OK  : {len(good):4d} / {len(sample)}")
    log(f"  Failed   : {len(bad):4d} / {len(sample)}")
    if bad:
        for r in bad[:5]:
            log(f"    FAIL {Path(r['path']).name}: {(r['error'] or '')[:60]}")

    if not good:
        log("  ERROR: no files processed.")
        return lines

    # ── [A2] Overall pitch distribution ───────────────────────────────────────
    log(f"\n[A2] PITCH DISTRIBUTION  (non-drum notes only)")
    log("-" * 70)

    all_pitches  = [p for r in good for p in r["pitches"]]
    total_notes  = len(all_pitches)
    files_no_mel = sum(1 for r in good if r["total_notes"] == 0)

    log(f"  Files with melodic notes     : {len(good) - files_no_mel}")
    log(f"  Files with NO melodic notes  : {files_no_mel}  (drums-only or empty)")
    log(f"  Total non-drum notes         : {total_notes:,}")

    if not all_pitches:
        log("  No melodic pitches found.")
        return lines

    s_pitches = sorted(all_pitches)
    log(f"\n  Mean   : {statistics.mean(all_pitches):.1f}")
    log(f"  Median : {statistics.median(all_pitches):.1f}")
    log(f"  Min    : {min(all_pitches)}")
    log(f"  Max    : {max(all_pitches)}")

    already_below = sum(1 for p in all_pitches if p < PITCH_MIN)
    already_above = sum(1 for p in all_pitches if p > PITCH_MAX)
    log(f"\n  Notes already < {PITCH_MIN} (out-of-range before any shift): "
        f"{already_below:,} ({_pct(already_below, total_notes):.3f}%)")
    log(f"  Notes already > {PITCH_MAX} (out-of-range before any shift): "
        f"{already_above:,} ({_pct(already_above, total_notes):.3f}%)")

    # Distribution by MIDI octave
    log(f"\n  Distribution by MIDI octave (octave -1 = MIDI 0-11, C4 = MIDI 60):")
    oct_counts = Counter(p // 12 - 1 for p in all_pitches)  # MIDI octave: oct = p//12 - 1
    for o in range(-1, 10):
        lo, hi  = (o + 1) * 12, (o + 1) * 12 + 11
        cnt     = oct_counts.get(o, 0)
        pct     = _pct(cnt, total_notes)
        marker  = " <- tokenizer boundary" if lo <= PITCH_MIN <= hi or lo <= PITCH_MAX <= hi else ""
        log(f"  Oct {o:2d}  MIDI {lo:3d}-{hi:3d}: {cnt:>8,}  ({pct:5.1f}%)  "
            f"{_bar_chart(cnt, total_notes, 18)}{marker}")

    # ── [A3] Shift-by-shift analysis ─────────────────────────────────────────
    log(f"\n[A3] SHIFT ANALYSIS -- NOTES OUTSIDE [{PITCH_MIN}, {PITCH_MAX}] AFTER SHIFT")
    log("-" * 70)
    log(f"  Notes outside this range are silently dropped by MidiTok during tokenization.")
    log(f"\n  {'Shift':>6}  {'Files affected':>18}  "
        f"{'Notes lost':>12}  {'% notes lost':>14}  {'Safety':>8}")
    log("  " + "-" * 66)

    melodic_files = [r["pitches"] for r in good if r["pitches"]]
    shift_data    = {}

    for shift in SHIFTS:
        files_hit  = 0
        notes_lost = 0
        total      = 0
        for pitches in melodic_files:
            lost = sum(1 for p in pitches
                       if (p + shift) < PITCH_MIN or (p + shift) > PITCH_MAX)
            if lost:
                files_hit += 1
            notes_lost += lost
            total      += len(pitches)

        pct_files  = _pct(files_hit, len(melodic_files))
        pct_notes  = _pct(notes_lost, total)
        safety     = ("SAFE" if pct_notes < 1.0
                      else "WARN" if pct_notes < 5.0 else "UNSAFE")
        shift_data[shift] = dict(
            files_hit=files_hit, notes_lost=notes_lost,
            total=total, pct_files=pct_files, pct_notes=pct_notes, safety=safety,
        )
        log(f"  {shift:>+6}  {files_hit:>5} / {len(melodic_files):<6} ({pct_files:5.1f}%)  "
            f"  {notes_lost:>8,} / {total:,}  ({pct_notes:5.2f}%)   {safety}")

    # ── [A4] Boundary analysis -- which pitches are at risk ────────────────────
    log(f"\n[A4] BOUNDARY ANALYSIS -- NOTES NEAR RANGE LIMITS")
    log("-" * 70)

    log(f"\n  UPPER BOUNDARY (notes at risk for upward shifts):")
    log(f"  {'Pitch range':<20}  {'Count':>8}  {'% of total':>12}  Risk for shift >=")
    for semitones in range(1, 6):
        threshold = PITCH_MAX - semitones + 1
        note_names = {109: "C#8", 108: "C8", 107: "B7", 106: "A#7/Bb7",
                      105: "A7", 104: "G#7/Ab7", 103: "G7", 102: "F#7"}
        name = note_names.get(threshold, f"MIDI {threshold}")
        cnt  = sum(1 for p in all_pitches if p >= threshold)
        pct  = _pct(cnt, total_notes)
        log(f"  >= {threshold} ({name}):           {cnt:>8,}  ({pct:>10.4f}%)  +{semitones}")

    log(f"\n  LOWER BOUNDARY (notes at risk for downward shifts):")
    log(f"  {'Pitch range':<20}  {'Count':>8}  {'% of total':>12}  Risk for shift <=")
    for semitones in range(1, 6):
        threshold = PITCH_MIN + semitones - 1
        note_names = {21: "A0", 22: "A#0/Bb0", 23: "B0", 24: "C1", 25: "C#1/Db1"}
        name = note_names.get(threshold, f"MIDI {threshold}")
        cnt  = sum(1 for p in all_pitches if p <= threshold)
        pct  = _pct(cnt, total_notes)
        log(f"  <= {threshold} ({name}):           {cnt:>8,}  ({pct:>10.4f}%)  -{semitones}")

    # ── [A5] Recommendation ───────────────────────────────────────────────────
    log(f"\n[A5] AUGMENTATION RECOMMENDATION")
    log("-" * 70)

    safe_shifts = [s for s, d in shift_data.items() if d["safety"] == "SAFE"]
    warn_shifts = [s for s, d in shift_data.items() if d["safety"] == "WARN"]
    bad_shifts  = [s for s, d in shift_data.items() if d["safety"] == "UNSAFE"]
    max_pct     = max(d["pct_notes"] for d in shift_data.values())

    log(f"\n  Thresholds: SAFE = <1% notes lost, WARN = 1-5%, UNSAFE = >5%")
    log(f"\n  SAFE    shifts : {safe_shifts}")
    log(f"  WARNING shifts : {warn_shifts}")
    log(f"  UNSAFE  shifts : {bad_shifts}")
    log(f"\n  Max note loss across ±5 range: {max_pct:.3f}%")

    if not bad_shifts:
        log(f"\n  CONCLUSION: Pitch shifting ±5 semitones is FEASIBLE.")
        log(f"  Even in the worst-case shift, < {max_pct:.2f}% of notes go out of range.")
        log(f"  MidiTok will silently drop these during tokenization.")
        log(f"  This is an acceptable loss for dataset augmentation.")
        log(f"\n  RECOMMENDATION:")
        log(f"    - Implement pitch shifting ±5 semitones as planned.")
        log(f"    - Optionally clamp out-of-range notes to PITCH_MIN/PITCH_MAX")
        log(f"      instead of dropping them (preserves note count, minor pitch error).")
        log(f"    - Expected augmentation factor: ×11 (original + 10 shifts)")
    else:
        log(f"\n  CAUTION: Some shifts cause significant note loss (>5%).")
        log(f"  Consider limiting augmentation to: {safe_shifts + warn_shifts}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log("ANALYSIS A -- SUMMARY")
    log(f"{'=' * 70}")
    log(f"  Files analyzed               : {len(good)}")
    log(f"  Total non-drum notes         : {total_notes:,}")
    log(f"  Pitch range (tokenizer)      : [{PITCH_MIN}, {PITCH_MAX}]")
    log(f"  Safe shifts                  : {safe_shifts}")
    if warn_shifts:
        log(f"  Warning shifts               : {warn_shifts}")
    if bad_shifts:
        log(f"  Unsafe shifts                : {bad_shifts}")
    log(f"  Max note loss (any shift)    : {max_pct:.3f}%")
    log(f"  Verdict                      : {'PITCH AUGMENTATION ±5 IS SAFE' if not bad_shifts else 'CHECK WARNING SHIFTS'}")
    log("=" * 70)

    return lines


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    if not MIDI_DIR.exists():
        print(f"ERROR: dataset not found at {MIDI_DIR.resolve()}")
        sys.exit(1)

    # Fast file scan (one glob per top-level subdir)
    top_subdirs = sorted(p for p in MIDI_DIR.iterdir() if p.is_dir())
    all_midi: list = []
    for sd in top_subdirs:
        all_midi.extend(sd.glob("*.mid"))
        all_midi.extend(sd.glob("*.midi"))

    print(f"\n{len(all_midi):,} MIDI files found in {MIDI_DIR}")
    LOGS_DIR.mkdir(exist_ok=True, parents=True)

    # Pick a fixed set of 10 files for config comparison
    random.seed(RANDOM_SEED)
    config_files = random.sample(all_midi, min(CONFIG_SAMPLE, len(all_midi)))

    # ── Analysis B ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RUNNING ANALYSIS B -- CONFIG V2 VOCAB IMPACT")
    print("=" * 70)
    lines_b = run_analysis_b(config_files)
    with open(REPORT_B, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines_b) + "\n")
    print(f"\nReport B written to: {REPORT_B.resolve()}")

    # ── Analysis A ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RUNNING ANALYSIS A -- PITCH SHIFT VALIDATION")
    print("=" * 70)
    lines_a = run_analysis_a(all_midi)
    with open(REPORT_A, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines_a) + "\n")
    print(f"\nReport A written to: {REPORT_A.resolve()}")

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import miditok   # noqa
    except ImportError:
        print("ERROR: miditok not found. Install with: pip install miditok")
        sys.exit(1)
    try:
        import miditoolkit  # noqa
    except ImportError:
        print("ERROR: miditoolkit not found. Install with: pip install miditoolkit")
        sys.exit(1)

    main()
