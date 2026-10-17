"""
validate_v2_preprocessing.py -- MIDIT V2 Preprocessing Validation
==================================================================
Validates .pt files produced by preprocess_v2.py against 5 criteria:

  Test 1: VOCAB RANGE       -- all token IDs in [0, 549]
  Test 2: BAR POSITIONS     -- structural integrity of bar_positions list
  Test 3: PITCH SHIFT       -- shift+1 raises Pitch tokens by exactly 1 semitone
  Test 4: CHUNKING          -- 8-bar chunk simulation from lakh_dataset.py
  Test 5: MASKING           -- TokenLevelMasking output properties

Run from MIDIT project root:
    python scripts/verification/validate_v2_preprocessing.py
"""

from __future__ import annotations

import datetime
import random
import sys
from pathlib import Path

import torch

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR     = Path("data/preprocessed_v2_test")
LOGS_DIR     = Path("logs")
OUTPUT_FILE  = LOGS_DIR / "v2_preprocessing_validation.txt"

VOCAB_SIZE   = 550          # V2 tokenizer
MASK_ID      = 3            # MASK_None=3 confirmed from V2 tokenizer
BARS_PER_CHUNK  = 8
MAX_TOKENS      = 8192
MASK_RATIO      = 0.15
SEED            = 42

# Test sample sizes
N_VOCAB_CHECK    = 100
N_SHIFT_PAIRS    = 10
N_CHUNK_SIM      = 50
N_MASK_SIM       = 20


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:.1f}%" if total > 0 else "N/A"


class TestResult:
    def __init__(self, name: str):
        self.name    = name
        self.passed  = 0
        self.failed  = 0
        self.details: list[str] = []

    def ok(self, msg: str = "") -> None:
        self.passed += 1
        if msg:
            self.details.append(f"  OK  {msg}")

    def fail(self, msg: str) -> None:
        self.failed += 1
        self.details.append(f"  FAIL  {msg}")

    @property
    def total(self) -> int:
        return self.passed + self.failed

    def summary(self) -> str:
        status = "PASS" if self.failed == 0 else "FAIL"
        return (
            f"[{status}]  {self.name}  "
            f"{self.passed}/{self.total} passed"
            + (f"  ({self.failed} failures)" if self.failed else "")
        )


# ── Tokenizer (initialized once) ──────────────────────────────────────────────

def _build_tokenizer():
    """Return (tokenizer, id_to_token, id_to_pitch_midi) for V2 config."""
    from miditok import REMI, TokenizerConfig
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_controls=False,
        delete_equal_successive_tempo_changes=True,
        delete_equal_successive_time_sig_changes=True,
    )
    tok = REMI(config)
    id_to_token = {v: k for k, v in tok.vocab.items()}

    # Map token_id -> MIDI pitch value for Pitch_* tokens (excludes drums)
    id_to_pitch: dict[int, int] = {}
    for name, tok_id in tok.vocab.items():
        if name.startswith("Pitch_"):
            try:
                id_to_pitch[tok_id] = int(name[6:])  # "Pitch_60" -> 60
            except ValueError:
                pass
    # Separate dict for PitchDrum (drums — should NOT shift)
    id_to_pitchdrum: dict[int, int] = {}
    for name, tok_id in tok.vocab.items():
        if name.startswith("PitchDrum_"):
            try:
                id_to_pitchdrum[tok_id] = int(name[10:])
            except ValueError:
                pass

    return tok, id_to_token, id_to_pitch, id_to_pitchdrum


# ── Test 1 -- Vocab range ─────────────────────────────────────────────────────

def test_vocab_range(files: list[Path], log) -> TestResult:
    result = TestResult("VOCAB RANGE")
    log()
    log("=" * 72)
    log("TEST 1 -- VOCAB RANGE  (all token IDs in [0, 549])")
    log("=" * 72)
    log(f"  Checking {len(files)} random .pt files...")
    log()

    for pt_file in files:
        try:
            data    = torch.load(pt_file, weights_only=False)
            tokens  = data["tokens"]
            min_id  = tokens.min().item()
            max_id  = tokens.max().item()

            if min_id < 0 or max_id >= VOCAB_SIZE:
                result.fail(
                    f"{pt_file.name}  min={min_id}  max={max_id}  "
                    f"(expected [0, {VOCAB_SIZE-1}])"
                )
            else:
                result.ok()
        except Exception as e:
            result.fail(f"{pt_file.name}  load error: {e}")

    if result.failed == 0:
        log(f"  All {result.total} files: token IDs strictly in [0, {VOCAB_SIZE-1}]")
    else:
        log(f"  Failures ({result.failed}):")
        for d in result.details:
            if "FAIL" in d:
                log(d)

    log()
    log(f"  {result.summary()}")
    return result


# ── Test 2 -- Bar positions integrity ─────────────────────────────────────────

def test_bar_positions(files: list[Path], log) -> TestResult:
    result = TestResult("BAR POSITIONS INTEGRITY")
    log()
    log("=" * 72)
    log("TEST 2 -- BAR POSITIONS INTEGRITY")
    log("=" * 72)
    log(f"  Checking {len(files)} .pt files...")
    log()

    stat_bars: list[int] = []

    for pt_file in files:
        try:
            data       = torch.load(pt_file, weights_only=False)
            tokens     = data["tokens"]
            bar_pos    = data["bar_positions"]
            n_toks     = len(tokens)
            file_ok    = True

            # Check [0] == 0
            if bar_pos[0] != 0:
                result.fail(f"{pt_file.name}  bar_positions[0]={bar_pos[0]}, expected 0")
                file_ok = False

            # Check [-1] == len(tokens)
            if bar_pos[-1] != n_toks:
                result.fail(
                    f"{pt_file.name}  bar_positions[-1]={bar_pos[-1]}, "
                    f"expected {n_toks} (len(tokens))"
                )
                file_ok = False

            # Check strictly ascending
            for i in range(1, len(bar_pos)):
                if bar_pos[i] <= bar_pos[i - 1]:
                    result.fail(
                        f"{pt_file.name}  not strictly ascending at index {i}: "
                        f"bar_pos[{i-1}]={bar_pos[i-1]}, bar_pos[{i}]={bar_pos[i]}"
                    )
                    file_ok = False
                    break

            # Check len >= 9 (at least 8 bars + sentinel)
            if len(bar_pos) < 9:
                result.fail(
                    f"{pt_file.name}  len(bar_positions)={len(bar_pos)}, "
                    f"expected >= 9 (8 bars + sentinel)"
                )
                file_ok = False

            if file_ok:
                n_bars = len(bar_pos) - 1
                stat_bars.append(n_bars)
                result.ok()

        except Exception as e:
            result.fail(f"{pt_file.name}  load error: {e}")

    if stat_bars:
        import statistics
        log(f"  Bar count stats (passing files):")
        log(f"    Mean   : {statistics.mean(stat_bars):.1f}")
        log(f"    Median : {statistics.median(stat_bars):.0f}")
        log(f"    Min    : {min(stat_bars)}")
        log(f"    Max    : {max(stat_bars)}")
    log()
    if result.failed:
        log(f"  Failures ({result.failed}):")
        for d in result.details:
            if "FAIL" in d:
                log(d)
    log(f"  {result.summary()}")
    return result


# ── Test 3 -- Pitch shift verification ────────────────────────────────────────

def test_pitch_shift(
    all_files: list[Path],
    id_to_pitch: dict[int, int],
    id_to_pitchdrum: dict[int, int],
    log,
) -> TestResult:
    result = TestResult("PITCH SHIFT VERIFICATION")
    log()
    log("=" * 72)
    log("TEST 3 -- PITCH SHIFT VERIFICATION  (shift=0 vs shift+1)")
    log("=" * 72)

    # Find pairs: base file (no shift marker) + _shift+1 version
    base_files = [f for f in all_files if "_shift" not in f.name]
    pairs: list[tuple[Path, Path]] = []
    for base in base_files:
        shift1 = base.parent / (base.stem + "_shift+1.pt")
        if shift1.exists():
            pairs.append((base, shift1))
        if len(pairs) >= N_SHIFT_PAIRS:
            break

    log(f"  Found {len(pairs)} base+shift+1 pairs to check")
    log()

    PITCH_MIN_MIDI = 21
    PITCH_MAX_MIDI = 108  # clip ceiling used in preprocess_v2.py

    for base_path, shift1_path in pairs:
        try:
            d0 = torch.load(base_path,   weights_only=False)
            d1 = torch.load(shift1_path, weights_only=False)
            t0 = d0["tokens"]
            t1 = d1["tokens"]

            if len(t0) != len(t1):
                result.fail(
                    f"{base_path.name}  len mismatch: "
                    f"shift=0 has {len(t0)} tokens, shift+1 has {len(t1)}"
                )
                continue

            melody_checked = 0
            melody_wrong   = 0
            drum_changed   = 0

            for pos in range(len(t0)):
                id0 = t0[pos].item()
                id1 = t1[pos].item()

                # Melodic pitch position
                if id0 in id_to_pitch:
                    p0 = id_to_pitch[id0]
                    if id1 not in id_to_pitch:
                        # token type changed unexpectedly
                        melody_wrong += 1
                    else:
                        p1 = id_to_pitch[id1]
                        expected_p1 = min(PITCH_MAX_MIDI, p0 + 1)
                        if p1 != expected_p1:
                            melody_wrong += 1
                    melody_checked += 1

                # Drum pitch -- should be UNCHANGED
                elif id0 in id_to_pitchdrum:
                    if id0 != id1:
                        drum_changed += 1

            if melody_checked == 0:
                result.fail(f"{base_path.name}  no Pitch tokens found")
                continue

            wrong_pct = 100.0 * melody_wrong / melody_checked
            if melody_wrong > 0:
                result.fail(
                    f"{base_path.name}  {melody_wrong}/{melody_checked} melody pitch "
                    f"tokens wrong ({wrong_pct:.1f}%)"
                )
            elif drum_changed > 0:
                result.fail(
                    f"{base_path.name}  {drum_changed} drum pitch tokens changed "
                    f"(should be unchanged)"
                )
            else:
                result.ok(
                    f"{base_path.name}  {melody_checked} melody pitches all +1  "
                    f"drums unchanged"
                )

        except Exception as e:
            result.fail(f"{base_path.name}  error: {e}")

    log()
    for d in result.details:
        log(d)
    log()
    log(f"  {result.summary()}")
    return result


# ── Test 4 -- Chunking simulation ─────────────────────────────────────────────

def test_chunking(files: list[Path], log) -> TestResult:
    result = TestResult("CHUNKING SIMULATION")
    log()
    log("=" * 72)
    log(f"TEST 4 -- CHUNKING SIMULATION  ({BARS_PER_CHUNK}-bar chunks, lakh_dataset style)")
    log("=" * 72)
    log(f"  Simulating __getitem__ chunking on {len(files)} files...")
    log()

    rng = random.Random(SEED + 4)
    chunk_lengths: list[int] = []

    for pt_file in files:
        try:
            data       = torch.load(pt_file, weights_only=False)
            tokens     = data["tokens"]
            bar_pos    = data["bar_positions"]
            num_bars   = len(bar_pos) - 1

            if num_bars < BARS_PER_CHUNK:
                result.fail(
                    f"{pt_file.name}  only {num_bars} bars, need {BARS_PER_CHUNK}"
                )
                continue

            # Replicate lakh_dataset._chunk_by_bars exactly
            max_start  = num_bars - BARS_PER_CHUNK
            start_bar  = rng.randint(0, max_start)
            end_bar    = start_bar + BARS_PER_CHUNK
            start_idx  = bar_pos[start_bar]
            end_idx    = bar_pos[end_bar]
            chunk      = tokens[start_idx:end_idx]
            chunk_len  = len(chunk)

            ok = True

            if chunk_len <= 0:
                result.fail(f"{pt_file.name}  empty chunk (len=0)")
                ok = False

            if chunk_len > MAX_TOKENS:
                result.fail(
                    f"{pt_file.name}  chunk too long: {chunk_len} > {MAX_TOKENS}"
                )
                ok = False

            # All token IDs in range
            if chunk_len > 0:
                if chunk.min().item() < 0 or chunk.max().item() >= VOCAB_SIZE:
                    result.fail(
                        f"{pt_file.name}  out-of-range token in chunk: "
                        f"min={chunk.min().item()} max={chunk.max().item()}"
                    )
                    ok = False

            if ok:
                chunk_lengths.append(chunk_len)
                result.ok()

        except Exception as e:
            result.fail(f"{pt_file.name}  error: {e}")

    if chunk_lengths:
        import statistics
        log(f"  Chunk length stats (passing files):")
        log(f"    Mean   : {statistics.mean(chunk_lengths):.0f} tokens")
        log(f"    Median : {statistics.median(chunk_lengths):.0f} tokens")
        log(f"    Min    : {min(chunk_lengths)} tokens")
        log(f"    Max    : {max(chunk_lengths)} tokens")
        long_chunks = sum(1 for c in chunk_lengths if c > MAX_TOKENS)
        log(f"    > {MAX_TOKENS} tokens (would truncate): {long_chunks}")
    log()
    if result.failed:
        log(f"  Failures ({result.failed}):")
        for d in result.details:
            if "FAIL" in d:
                log(d)
    log(f"  {result.summary()}")
    return result


# ── Test 5 -- Masking simulation ──────────────────────────────────────────────

def test_masking(files: list[Path], log) -> TestResult:
    result = TestResult("MASKING SIMULATION")
    log()
    log("=" * 72)
    log(f"TEST 5 -- MASKING SIMULATION  (TokenLevelMasking, ratio={MASK_RATIO})")
    log("=" * 72)
    log(f"  Applying masking to {len(files)} random chunks...")
    log()

    # Import from actual project module
    sys.path.insert(0, str(Path(".").resolve()))
    from midit.data.masking import TokenLevelMasking

    masker = TokenLevelMasking(
        mask_token_id=MASK_ID,
        mask_ratio=MASK_RATIO,
        mask_strategy="bert",
        vocab_size=VOCAB_SIZE,
    )

    rng = random.Random(SEED + 5)
    actual_ratios: list[float] = []
    TOLERANCE = 0.05  # allow mask ratio in [0.10, 0.20]

    for pt_file in files:
        try:
            data     = torch.load(pt_file, weights_only=False)
            tokens   = data["tokens"]
            bar_pos  = data["bar_positions"]
            num_bars = len(bar_pos) - 1

            if num_bars < BARS_PER_CHUNK:
                continue  # skip, not enough bars

            # Simulate chunk extraction
            max_start = num_bars - BARS_PER_CHUNK
            start_bar = rng.randint(0, max_start)
            start_idx = bar_pos[start_bar]
            end_idx   = bar_pos[start_bar + BARS_PER_CHUNK]
            chunk     = tokens[start_idx:end_idx]

            if len(chunk) == 0:
                continue

            masked_tokens, mask_labels = masker.apply_masking(
                tokens=chunk,
                bar_positions=None,
            )

            ok = True

            # Check 1: mask_labels is bool tensor
            if mask_labels.dtype != torch.bool:
                result.fail(
                    f"{pt_file.name}  mask_labels dtype={mask_labels.dtype}, "
                    f"expected torch.bool"
                )
                ok = False

            # Check 2: mask ratio ~15% (tolerance +-5%)
            ratio = mask_labels.float().mean().item()
            actual_ratios.append(ratio)
            if not (MASK_RATIO - TOLERANCE <= ratio <= MASK_RATIO + TOLERANCE):
                result.fail(
                    f"{pt_file.name}  mask ratio={ratio:.3f} "
                    f"outside [{MASK_RATIO-TOLERANCE:.2f}, {MASK_RATIO+TOLERANCE:.2f}]"
                )
                ok = False

            # Check 3: no token ID in masked input exceeds VOCAB_SIZE-1
            if masked_tokens.max().item() >= VOCAB_SIZE:
                result.fail(
                    f"{pt_file.name}  masked_tokens max={masked_tokens.max().item()} "
                    f">= {VOCAB_SIZE}"
                )
                ok = False

            # Check 4: MASK_ID appears only at masked positions
            mask_id_positions = (masked_tokens == MASK_ID).nonzero(as_tuple=True)[0]
            for pos in mask_id_positions:
                if not mask_labels[pos].item():
                    # MASK token at a non-masked position
                    # (only a problem if original token was not MASK -- which it won't be)
                    orig_was_mask = (chunk[pos].item() == MASK_ID)
                    if not orig_was_mask:
                        result.fail(
                            f"{pt_file.name}  MASK_ID at pos {pos.item()} "
                            f"but mask_labels[pos]=False"
                        )
                        ok = False
                        break

            if ok:
                result.ok()

        except Exception as e:
            result.fail(f"{pt_file.name}  error: {e}")

    if actual_ratios:
        import statistics
        log(f"  Actual mask ratio stats across {len(actual_ratios)} chunks:")
        log(f"    Mean   : {statistics.mean(actual_ratios):.4f}  "
            f"(target {MASK_RATIO:.2f}, tolerance +-{TOLERANCE:.2f})")
        log(f"    Min    : {min(actual_ratios):.4f}")
        log(f"    Max    : {max(actual_ratios):.4f}")
        log(f"    StdDev : {statistics.stdev(actual_ratios):.4f}")
    log()
    if result.failed:
        log(f"  Failures ({result.failed}):")
        for d in result.details:
            if "FAIL" in d:
                log(d)
    log(f"  {result.summary()}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    import time
    t0 = time.time()

    log("=" * 72)
    log("V2 PREPROCESSING VALIDATION -- MIDIT")
    log(f"Data dir : {DATA_DIR.resolve()}")
    log(f"Date     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Vocab    : {VOCAB_SIZE} tokens  (V2, IDs 0-{VOCAB_SIZE-1})")
    log(f"MASK ID  : {MASK_ID}")
    log("=" * 72)
    log()

    # ── Discover files ─────────────────────────────────────────────────────────
    if not DATA_DIR.exists():
        log(f"ERROR: {DATA_DIR} does not exist. Check path.")
        sys.exit(1)

    all_files = sorted(DATA_DIR.glob("*.pt"))
    log(f"  Total .pt files : {len(all_files):,}")

    rng = random.Random(SEED)
    shuffled = all_files.copy()
    rng.shuffle(shuffled)

    # Build file lists for each test
    vocab_files   = shuffled[:N_VOCAB_CHECK]
    bar_files     = shuffled[:N_VOCAB_CHECK]   # same 100 for bar check
    chunk_files   = shuffled[:N_CHUNK_SIM]
    mask_files    = shuffled[:N_MASK_SIM]

    log(f"  Files for T1 (vocab)   : {len(vocab_files)}")
    log(f"  Files for T2 (bar pos) : {len(bar_files)}")
    log(f"  Pairs for T3 (shift)   : up to {N_SHIFT_PAIRS}")
    log(f"  Files for T4 (chunk)   : {len(chunk_files)}")
    log(f"  Files for T5 (mask)    : {len(mask_files)}")
    log()

    # ── Build tokenizer ────────────────────────────────────────────────────────
    log("Building V2 tokenizer for pitch/vocab lookup...")
    _, id_to_token, id_to_pitch, id_to_pitchdrum = _build_tokenizer()
    log(f"  Pitch tokens    : {len(id_to_pitch)}")
    log(f"  PitchDrum tokens: {len(id_to_pitchdrum)}")
    log()

    # ── Run tests ─────────────────────────────────────────────────────────────
    results: list[TestResult] = []

    results.append(test_vocab_range(vocab_files, log))
    results.append(test_bar_positions(bar_files, log))
    results.append(test_pitch_shift(all_files, id_to_pitch, id_to_pitchdrum, log))
    results.append(test_chunking(chunk_files, log))
    results.append(test_masking(mask_files, log))

    # ── Final summary ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - t0
    all_passed = all(r.failed == 0 for r in results)
    overall    = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"

    log()
    log("=" * 72)
    log(f"FINAL SUMMARY  --  {overall}")
    log("=" * 72)
    log()
    for r in results:
        log(f"  {r.summary()}")
    log()
    total_checks  = sum(r.total  for r in results)
    total_passed  = sum(r.passed for r in results)
    total_failed  = sum(r.failed for r in results)
    log(f"  Total checks : {total_checks}")
    log(f"  Passed       : {total_passed}")
    log(f"  Failed       : {total_failed}")
    log()
    log(f"  Elapsed      : {total_elapsed:.1f}s")
    log(f"  Report saved : {OUTPUT_FILE.resolve()}")
    log("=" * 72)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nReport written to: {OUTPUT_FILE}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
