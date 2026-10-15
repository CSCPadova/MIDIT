# Cloze (Phase-1) accuracy — exact protocol (read-only findings)

The term "cloze" appears only in `.claude/CLAUDE.md`/`README.md`, not in code. The cloze
Top-1/Top-5 accuracy is computed by the Phase-1 accuracy scripts formerly in
`scripts/Evaluation_Accuracy/` (now `scripts/evaluation/evaluate_phase1_accuracy_DIRECT.py`
and `research/Evaluation_Accuracy/evaluate_phase1_EXTENDED.py`). There are **two** scripts:

- **`evaluate_phase1_accuracy_DIRECT.py` — authoritative / thesis.** V2 tokenizer
  (`beat_res={(0,4):8,(4,12):4}`, delete-equal flags → vocab 550), a proper
  `--split` (default `val`), and **sorted** token-name→class mapping that matches
  `LakhDataset` (docstring lines 17-23).
- **`evaluate_phase1_EXTENDED.py` — stale, do not cite.** V1 tokenizer
  (`beat_res={(0,4):8}` only → vocab 518, no delete-equal flags, line 234-245),
  **insertion-order** class mapping (the bug DIRECT fixed), and it samples from
  **all** `.pt` files with **no train/val split** (line 98-104).

All line references below are to **`evaluate_phase1_accuracy_DIRECT.py`** unless
noted. **FACT** = read in code; **INFERENCE** = deduction.

---

## Q1 — Masking protocol

**FACT — a single random 15% of tokens per file, masked once (not iterative).**
`apply_masking(tokens, targets, mask_ratio=0.15, mask_id=3)` (lines 93-108):
`num_mask = max(1, int(seq_len * 0.15))` (line 96), then
`mask_positions = random.sample(range(seq_len), num_mask)` (line 97) — a uniform
**without-replacement** sample of 15% of positions. It is **not** "all tokens
iteratively"; each file gets one random 15% mask and one forward pass
(lines 174-186).

Applied to **truncated** sequences: files with `< 50` tokens are skipped and
sequences are cut to the **first 512** tokens (lines 166-169), so accuracy is
measured on the first ≤512 tokens of each file, not the whole song.

## Q2 — Which token types are "target" (included)

**FACT — the script evaluates ALL 9 head types.** `create_targets_dict` builds
targets for `pitch` (Pitch + PitchDrum), `velocity, duration, program, position,
bar, tempo, timesig, rest` (lines 52-62), and the evaluation loop iterates over
every head returned by `multi_head_output` (line 189) and reports each
(lines 239-262). Since masking is random over all positions, every type receives
masked tokens in proportion to its frequency (rare types like `tempo` get very
few → noisy).

**INFERENCE / important caveat — the thesis *average* is a 7-type subset, done
manually.** The script's printed "Average Top-1" is a plain mean over **all**
heads with data (`np.mean` over `results.values()`, line 270). The thesis figure
(Avg Top-1 83.18%) **excludes Bar and TimeSig** ("trivially predictable", per
.claude/CLAUDE.md). That exclusion is a **post-hoc reporting choice**, NOT implemented in
this script — the script itself includes Bar (≈100%) and TimeSig (≈high), which
would inflate the raw printed average. So: **script = 9 types; cited average = 7
(Bar/TimeSig dropped).**

## Q3 — Mask replacement: 100% [MASK] or 80/10/10?

**FACT — always 100% [MASK], no 80/10/10.** `input_ids[mask_positions] = mask_id`
with `mask_id = 3` (lines 99-100). Every masked position is replaced by the
`[MASK]` token deterministically; there is no random-token or keep-unchanged
branch. (This matches inference, where masked slots are always `[MASK]`.)

## Q4 — How the 400 files are selected

**Two-stage, only the first stage is seeded (FACT):**

1. **Split (seeded):** `all_files = sorted(glob("*.pt"))`, then
   `np.random.RandomState(42)` shuffles the **file indices** and takes 80/10/10;
   `val = indices[n_train : n_train+n_val]` (lines 127-143). `--split` defaults to
   `val` (line 295).
2. **Sub-sample (NOT seeded):** `sample_files = random.sample(split_files,
   num_samples)` (line 150) uses Python's **global `random`** with **no seed set**
   anywhere in the script. `num_samples` defaults to 100; **"400" = the run passed
   `--num_samples 400`.**

So the 400 files are a random subsample of the val split; the **subsample itself
is not reproducibly fixed** (nor is the per-file masking, line 97 — also unseeded).

**⚠ Discrepancy to flag:** the split here is **file-level**, not the **song-level
(stem-grouped)** split used in training (`LakhDataset`, seed 42). The comment
"Replicate exact split logic from LakhDataset" is **inaccurate** — this shuffles
individual file indices, so pitch-shift augmentations of the same song can land in
different splits, and this eval's val set is **not** the training val set. This
mildly optimistic (possible augmentation leakage) and differs from the infilling
eval, which does use the song-level split.

## Q5 — Exact seed

- **Split seed = 42** (`np.random.RandomState(42)`, line 130). **FACT.**
- **Sub-sample seed = none** (unseeded `random.sample`, line 150). **FACT.**
- **Masking seed = none** (unseeded `random.sample`, line 97). **FACT.**

So the only fixed seed is **42 for the file-level split**; the 400-file selection
and the 15% masking are non-deterministic run-to-run. With 400 files the reported
average is statistically stable, but it is **not** exactly reproducible
file-for-file / mask-for-mask.

---

## Summary (authoritative script)

| Question | Answer (FACT) |
|---|---|
| Masking | random 15% of tokens, once per file, without replacement (line 96-97); on first ≤512 tokens (line 168-169) |
| Target types | all 9 heads evaluated (lines 52-62, 189); **cited average drops Bar+TimeSig** (post-hoc, not in code) |
| Mask replacement | 100% `[MASK]` (id 3), no 80/10/10 (lines 99-100) |
| File selection | val split via `RandomState(42)` file-level (lines 130-143) → `random.sample(400)` **unseeded** (line 150) |
| Seed | **42** for the split only; sampling + masking unseeded |

**Flags:** (1) split is file-level, not the training song-level split → not the
same val set as training, possible augmentation leakage; (2) the 83.18% average
excludes Bar/TimeSig by hand; (3) 400-file subsample and masking are unseeded, so
the number is reproducible only in distribution, not exactly. Use
`evaluate_phase1_accuracy_DIRECT.py` (V2), not `evaluate_phase1_EXTENDED.py` (V1,
buggy).
