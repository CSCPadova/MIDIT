"""
Token Frequency Analysis -- MIDIT REMI+ Tokenizer
==================================================
Samples 300 random files from data/midi_files/lmd_full/ and tokenizes
each with the EXACT V1 config from preprocess_batch_improved.py.

For the full token stream of each file, counts occurrences of each
token TYPE (mapped to the 9 model heads + Special).

Reports:
  1. Total tokens analyzed across all files
  2. Count and % per token type
  3. Class imbalance factor (max/min frequency ratio)
  4. Current lambda weights (conf/config.yaml) vs frequency-balanced
     suggestion
  5. Flags any type appearing < 0.1% of total tokens

Uses multiprocessing with 4 workers.
Saves report to logs/token_frequency_analysis.txt

Run from MIDIT project root:
  python scripts/Analyze_dataset/analyze_token_frequency.py

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
REPORT_PATH = LOGS_DIR / "token_frequency_analysis.txt"

# ─── Constants ────────────────────────────────────────────────────────────────
SAMPLE_SIZE  = 300
NUM_WORKERS  = 4
RANDOM_SEED  = 42
RARE_THRESHOLD_PCT = 0.1   # flag types below this % of total tokens

# ─── Lambda weights from conf/config.yaml (Run 6, current) ───────────────────
# bar=0.0 and timesig=0.5 confirmed by supervisor Matteo Spanio (March 12 2026)
LAMBDA_WEIGHTS = {
    "pitch":    3.0,
    "velocity": 1.5,
    "duration": 1.0,
    "program":  1.5,
    "position": 0.5,
    "bar":      0.0,
    "tempo":    2.0,
    "timesig":  0.5,
    "rest":     1.0,
}

# Head types that contribute to loss (lambda > 0 or non-trivial)
MODEL_HEADS = list(LAMBDA_WEIGHTS.keys())

# ─── Token type -> model head mapping ────────────────────────────────────────
# Pitch and PitchDrum both map to the "pitch" head.
# PAD/BOS/EOS/MASK are "special" (not a model head).
def _head_of(token_name: str) -> str:
    if token_name.startswith("Pitch_"):     return "pitch"
    if token_name.startswith("PitchDrum_"): return "pitch"
    if token_name.startswith("Velocity_"):  return "velocity"
    if token_name.startswith("Duration_"):  return "duration"
    if token_name.startswith("Program_"):   return "program"
    if token_name.startswith("Position_"):  return "position"
    if token_name.startswith("Bar_"):       return "bar"
    if token_name.startswith("Tempo_"):     return "tempo"
    if token_name.startswith("TimeSig_"):   return "timesig"
    if token_name.startswith("Rest_"):      return "rest"
    return "special"   # PAD, BOS, EOS, MASK


# ─── Worker initializer -- builds tokenizer once per worker process ───────────
_tokenizer   = None
_id_to_head  = None   # pre-built id -> head_type mapping (O(1) lookup)


def _init_worker():
    """Initialize the REMI tokenizer and pre-build the id->head lookup."""
    global _tokenizer, _id_to_head
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
    _id_to_head = {tok_id: _head_of(tok_name)
                   for tok_name, tok_id in _tokenizer.vocab.items()}


# ─── Per-file analysis (runs in worker processes) ─────────────────────────────
def _analyze_file(midi_path: Path) -> dict:
    """Tokenize one file and return per-head token counts."""
    result = {
        "path":    str(midi_path),
        "success": False,
        "error":   None,
        "counts":  {},    # head_type -> int
        "total":   0,
    }
    try:
        global _tokenizer, _id_to_head

        tokens = _tokenizer(str(midi_path))
        if isinstance(tokens, list):
            ids = []
            for seq in tokens:
                ids.extend(seq.ids)
        else:
            ids = tokens.ids

        counts = Counter(_id_to_head[i] for i in ids if i in _id_to_head)
        result["counts"]  = dict(counts)
        result["total"]   = len(ids)
        result["success"] = True

    except Exception as exc:
        result["error"] = str(exc)
    return result


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _bar(count: int, total: int, width: int = 35) -> str:
    filled = int(width * count / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def _freq_balanced_lambda(freq: float, ref_freq: float, ref_lambda: float) -> float:
    """Lambda proportional to 1/freq, anchored so pitch -> ref_lambda."""
    if freq <= 0:
        return 0.0
    raw = (ref_freq / freq) * ref_lambda
    return round(min(raw, 99.0), 2)   # cap at 99 to avoid overflow on near-zero


# ─── Main analysis ─────────────────────────────────────────────────────────────
def run_analysis():
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(str(msg))

    t0 = time.time()

    log("=" * 72)
    log("TOKEN FREQUENCY ANALYSIS -- MIDIT REMI+ TOKENIZER (V1 CONFIG)")
    log(f"Dataset   : {MIDI_DIR.resolve()}")
    log(f"Date      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
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

    total_files = len(all_midi)
    log(f"\n  Total files in dataset : {total_files:,}")

    random.seed(RANDOM_SEED)
    sample = random.sample(all_midi, min(SAMPLE_SIZE, total_files))
    log(f"  Sample size            : {len(sample)}  (seed={RANDOM_SEED})")

    # ── Parallel tokenization ─────────────────────────────────────────────────
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
        log("ERROR: no files tokenized. Cannot continue.")
        return

    # ── Aggregate counts ──────────────────────────────────────────────────────
    total_tokens = sum(r["total"] for r in good)
    agg = Counter()
    for r in good:
        agg.update(r["counts"])

    # All known heads in display order
    DISPLAY_ORDER = [
        "pitch", "velocity", "duration", "program",
        "position", "bar", "tempo", "timesig", "rest", "special",
    ]

    # Make sure every head appears (even if 0)
    for h in DISPLAY_ORDER:
        if h not in agg:
            agg[h] = 0

    # ── [2] Token count and % per type ────────────────────────────────────────
    log(f"\n[2] TOKEN TYPE DISTRIBUTION")
    log("-" * 72)
    log(f"\n  Total tokens analyzed : {total_tokens:,}  across {len(good)} files")
    log(f"  Average per file      : {total_tokens / len(good):,.0f} tokens")
    log()

    # Header
    log(f"  {'Type':<12}  {'Count':>12}  {'%':>7}  {'Bar chart (% of total)'}")
    log("  " + "-" * 68)

    for h in DISPLAY_ORDER:
        cnt = agg[h]
        pct = _pct(cnt, total_tokens)
        lam = LAMBDA_WEIGHTS.get(h, "n/a")
        lam_str = f"  lambda={lam}" if h in LAMBDA_WEIGHTS else "  (no head)"
        bar = _bar(cnt, total_tokens)
        log(f"  {h:<12}  {cnt:>12,}  {pct:>6.2f}%  {bar}{lam_str}")

    # ── [3] Class imbalance ───────────────────────────────────────────────────
    log(f"\n[3] CLASS IMBALANCE")
    log("-" * 72)

    # Exclude 'special' from imbalance calc (not a model head)
    model_head_counts = {h: agg[h] for h in MODEL_HEADS}
    nonzero = {h: c for h, c in model_head_counts.items() if c > 0}
    most_frequent  = max(nonzero, key=nonzero.get)
    least_frequent = min(nonzero, key=nonzero.get)

    imbalance_ratio = nonzero[most_frequent] / nonzero[least_frequent]

    log(f"\n  Most  frequent head  : {most_frequent:<12}  "
        f"{agg[most_frequent]:>12,}  ({_pct(agg[most_frequent], total_tokens):.2f}%)")
    log(f"  Least frequent head  : {least_frequent:<12}  "
        f"{agg[least_frequent]:>12,}  ({_pct(agg[least_frequent], total_tokens):.2f}%)")
    log(f"\n  Imbalance ratio (most / least) : {imbalance_ratio:,.0f}x")
    log(f"  (excluding 'special' and zero-count heads)")

    # Sorted table for the 9 model heads
    log(f"\n  9 model heads ranked by frequency:")
    log(f"  {'Rank':<5}  {'Type':<12}  {'Count':>12}  {'%':>7}  "
        f"{'lambda':>8}  {'Training signal rank'}")
    log("  " + "-" * 68)
    ranked = sorted(model_head_counts.items(), key=lambda x: -x[1])
    for rank, (h, cnt) in enumerate(ranked, 1):
        pct  = _pct(cnt, total_tokens)
        lam  = LAMBDA_WEIGHTS[h]
        # effective signal weight = lambda * frequency
        eff  = lam * pct
        log(f"  {rank:<5}  {h:<12}  {cnt:>12,}  {pct:>6.2f}%  "
            f"{lam:>8.1f}  eff={eff:.3f}  {'(zero - excluded)' if lam==0 else ''}")

    # ── [4] Lambda comparison ─────────────────────────────────────────────────
    log(f"\n[4] CURRENT LAMBDA vs FREQUENCY-BALANCED SUGGESTION")
    log("-" * 72)
    log(f"""
  Frequency-balanced lambda: lambda_t proportional to 1/freq(t)
  Anchored so that pitch retains its current lambda={LAMBDA_WEIGHTS['pitch']}.
  Interpretation: if you wanted EQUAL total training signal per head,
  you would use these lambdas. Deviations from this are intentional
  (e.g., bar=0 because trivial, pitch=3.0 because hardest task).
""")

    # Reference point: pitch head
    pitch_freq   = _pct(agg["pitch"], total_tokens) / 100.0
    pitch_lambda = LAMBDA_WEIGHTS["pitch"]

    log(f"  {'Type':<12}  {'Freq%':>7}  {'Current L':>10}  "
        f"{'Freq-balanced L':>16}  {'Ratio C/FB':>11}  Note")
    log("  " + "-" * 80)

    for h in DISPLAY_ORDER:
        if h == "special":
            continue
        cnt  = agg[h]
        freq = cnt / total_tokens if total_tokens else 0
        cur  = LAMBDA_WEIGHTS.get(h, 0.0)
        fb   = _freq_balanced_lambda(freq, pitch_freq, pitch_lambda)
        pct  = _pct(cnt, total_tokens)

        if fb > 0:
            ratio = cur / fb
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "  --"

        # Flag interesting deviations
        note = ""
        if cur == 0.0:
            note = "<- zeroed (trivial)"
        elif freq > 0 and pct < RARE_THRESHOLD_PCT:
            note = "<- RARE (<0.1%) SEE SECTION 5"
        elif fb > 0 and cur / fb < 0.2:
            note = "<- under-weighted vs freq"
        elif fb > 0 and cur / fb > 5.0:
            note = "<- over-weighted vs freq (intentional?)"

        log(f"  {h:<12}  {pct:>6.2f}%  {cur:>10.1f}  {fb:>16.2f}  "
            f"{ratio_str:>11}  {note}")

    log(f"""
  Reading guide:
    Ratio > 1.0 = current lambda HIGHER than frequency-balanced (type gets
                  more signal than its share of tokens would justify)
    Ratio < 1.0 = current lambda LOWER  than frequency-balanced (type gets
                  less signal -- either intentional or worth reviewing)
    ratio=0/--  = lambda is 0 (head excluded from loss)
""")

    # ── [5] Rare type flags ───────────────────────────────────────────────────
    log(f"\n[5] RARE TOKEN TYPES  (< {RARE_THRESHOLD_PCT}% of total tokens)")
    log("-" * 72)

    rare_heads = [(h, agg[h]) for h in MODEL_HEADS
                  if _pct(agg[h], total_tokens) < RARE_THRESHOLD_PCT]

    if not rare_heads:
        log(f"  No model head is below {RARE_THRESHOLD_PCT}% of total tokens.")
    else:
        log(f"  {len(rare_heads)} head(s) below threshold:")
        for h, cnt in sorted(rare_heads, key=lambda x: x[1]):
            pct = _pct(cnt, total_tokens)
            lam = LAMBDA_WEIGHTS[h]
            log(f"\n  [{h}]  {cnt:,} tokens  ({pct:.4f}%)  lambda={lam}")

            # Context-specific notes
            if h == "bar":
                log(f"    Note: Bar=0 (zeroed) because Bar_None is always the")
                log(f"    same token. Low frequency is expected and correct.")
            elif h == "tempo":
                log(f"    Note: Tempo tokens appear only at tempo change events.")
                log(f"    Very few per file = model sees almost no Tempo training signal.")
                log(f"    Current lambda={lam} partially compensates, but this type")
                log(f"    may remain hard to learn regardless.")
            elif h == "timesig":
                log(f"    Note: TimeSig tokens appear only at time-sig change events.")
                log(f"    Most files have a single TimeSig that repeats every bar.")
                log(f"    97% top-1 accuracy confirmed -- lambda={lam} is appropriate.")
            elif h == "rest":
                log(f"    Note: Rest tokens appear only when there are explicit rests.")
                log(f"    Not all MIDI files encode rests explicitly.")
            else:
                log(f"    Review: lambda={lam} -- consider if this head needs")
                log(f"    higher weight to compensate for low token count.")

    # ── [6] Per-file variability ──────────────────────────────────────────────
    log(f"\n[6] PER-FILE VARIABILITY")
    log("-" * 72)
    log(f"  How consistent are token type fractions across files?\n")

    log(f"  {'Type':<12}  {'Mean %':>8}  {'StdDev %':>10}  "
        f"{'Min %':>8}  {'Max %':>8}  {'Files with 0':>13}")
    log("  " + "-" * 68)

    for h in DISPLAY_ORDER:
        if h == "special":
            continue
        per_file_pcts = [
            100.0 * r["counts"].get(h, 0) / r["total"]
            for r in good if r["total"] > 0
        ]
        zero_files = sum(1 for r in good if r["counts"].get(h, 0) == 0)

        if per_file_pcts:
            mean   = statistics.mean(per_file_pcts)
            std    = statistics.stdev(per_file_pcts) if len(per_file_pcts) > 1 else 0.0
            mn, mx = min(per_file_pcts), max(per_file_pcts)
            log(f"  {h:<12}  {mean:>7.2f}%  {std:>9.2f}%  "
                f"{mn:>7.2f}%  {mx:>7.2f}%  "
                f"{zero_files:>8} / {len(good)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0

    log(f"\n{'=' * 72}")
    log("SUMMARY")
    log(f"{'=' * 72}")
    log(f"  Files analyzed           : {len(good):>8,}")
    log(f"  Total tokens             : {total_tokens:>8,}")
    log(f"  Avg tokens/file          : {total_tokens/len(good):>8,.0f}")
    log(f"  Vocab size (V1 config)   : {'518':>8}")
    log()

    # Top-level per-type summary
    log(f"  {'Type':<12}  {'%':>7}  {'lambda':>8}  {'eff signal':>12}")
    log("  " + "-" * 45)
    for h in DISPLAY_ORDER:
        if h == "special":
            continue
        pct = _pct(agg[h], total_tokens)
        lam = LAMBDA_WEIGHTS.get(h, 0.0)
        eff = pct * lam
        flag = "  *** RARE ***" if pct < RARE_THRESHOLD_PCT else ""
        log(f"  {h:<12}  {pct:>6.2f}%  {lam:>8.1f}  {eff:>12.4f}{flag}")

    log()
    log(f"  Class imbalance (max/min frequency): {imbalance_ratio:,.0f}x")
    rare_nonzero_lambda = [(h, _pct(agg[h], total_tokens))
                           for h in MODEL_HEADS
                           if _pct(agg[h], total_tokens) < RARE_THRESHOLD_PCT
                           and LAMBDA_WEIGHTS[h] > 0]
    if rare_nonzero_lambda:
        log(f"  Rare heads with active lambda: "
            f"{[(h, f'{p:.4f}%') for h, p in rare_nonzero_lambda]}")
        log(f"  These heads may benefit from higher lambda to compensate.")
    else:
        log(f"  No active-lambda head is below {RARE_THRESHOLD_PCT}% threshold.")

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
