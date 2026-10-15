# MIDIT — Decoding-strategy ablation: Phase-1 feasibility analysis

**Scope:** read-only analysis for a new `scripts/decoding_ablation.py` (argmax vs
sampling vs MaskGIT-style iterative decoding), analogous to MIDI-RWKV §E.3
Table 15. No existing file was modified.

Legend: **FACT** = read in code (with file:line) · **INFERENCE** = deduction ·
**HYPOTHESIS** = needs a runtime check.

---

## Q1 — Import compatibility (`midit/infer.py`)

All the utilities exist at module top level and are import-safe: `infer.py` guards
its entry point with `if __name__ == '__main__': main()` ([infer.py:493-494](../../../midit/infer.py#L493-L494)),
so importing it runs **no** inference. Two caveats: (a) a **module-level side
effect** — `sys.stdout` is *not* rewrapped in `infer.py` (that's in
`scripts/infer_from_midi.py`), so importing `midit.infer` is clean; (b) importing
it does trigger `from midit.model.masked_transformer import MaskedTransformerREMI`
and `from miditok import ...` at import time ([infer.py:21-24](../../../midit/infer.py#L21-L24)),
so the import cost includes loading torch + miditok (heavy, but no execution).

| Symbol | Exists? | Signature / value | Import-safe? |
|---|---|---|---|
| `build_tokenizer()` | ✅ | `build_tokenizer() -> REMI` ([infer.py:36](../../../midit/infer.py#L36)) | ✅ builds the V2 REMI+ tokenizer inline |
| `build_vocab_lookups()` | ✅ | `build_vocab_lookups(tokenizer: REMI) -> Dict` ([infer.py:57](../../../midit/infer.py#L57)); returns `{id_to_token, class_to_name, type_for_id, vocab}` | ✅ |
| `get_val_files()` | ✅ | `get_val_files(data_dir: str) -> List[Path]` ([infer.py:112](../../../midit/infer.py#L112)); song-level split, seed 42, one file per stem | ✅ (globs disk at call time, not import) |
| `extract_chunk()` | ✅ | `extract_chunk(file_path: Path, bars_per_chunk: int = 8) -> Optional[Dict]` ([infer.py:151](../../../midit/infer.py#L151)); returns `{tokens, bar_positions, source_file}` | ✅ |
| `apply_bar_mask()` | ✅ | `apply_bar_mask(tokens, bar_positions, mask_bars, lookups) -> (masked_tokens, mask_labels)` ([infer.py:178](../../../midit/infer.py#L178)) | ✅ |
| `MASK_ID / BOS_ID / EOS_ID / PAD_ID` | ✅ | `3 / 1 / 2 / 0` ([infer.py:27-30](../../../midit/infer.py#L27-L30)) | ✅ module constants |
| `SPECIAL_IDS` | ✅ | `{PAD_ID, BOS_ID, EOS_ID, MASK_ID}` = `{0,1,2,3}` ([infer.py:31](../../../midit/infer.py#L31)) | ✅ |

**Bonus (also importable, useful for the ablation):** `reconstruct()`
([infer.py:248](../../../midit/infer.py#L248)) — the current **argmax + oracle-type**
decoder — and `save_midi()` ([infer.py:365](../../../midit/infer.py#L365)),
`apply_random_mask()` ([infer.py:219](../../../midit/infer.py#L219)),
`compute_accuracy()` ([infer.py:326](../../../midit/infer.py#L326)),
`NOTE_PREFIXES` ([infer.py:173](../../../midit/infer.py#L173)).

**Recommendation:** import from `midit.infer` (package path), not by file path, and
ensure `PYTHONPATH=~/MIDIT` (the SLURM scripts already set this — Q4).

---

## Q2 — Multi-head output API (`masked_transformer.py`)

- **`model(x)` returns hidden states, not logits — CONFIRMED (FACT).** `forward()`
  returns `midi_encoded` of shape **(B, L, d_model=512)** ([masked_transformer.py:371-435](../../../midit/model/masked_transformer.py#L371-L435), return at [:435](../../../midit/model/masked_transformer.py#L435)). It applies token+position embeddings, LayerNorm, and the 12-layer `TransformerEncoder`; **no head is applied inside `forward`**.
- **`model.multi_head_output(hidden)` returns a dict — CONFIRMED (FACT).**
  `MultiHeadOutput.forward` returns `{head_name: (B, L, C_head)}` after a shared
  `LayerNorm` + `Dropout` ([multi_head_output.py:85-117](../../../midit/model/multi_head_output.py#L85-L117)). With `hidden` of shape `(1, L, 512)` each value is `(1, L, C_head)`.
- **Exact head keys & class counts (FACT, [multi_head_output.py:105-115](../../../midit/model/multi_head_output.py#L105-L115) + V2 vocab):**

  | key | C_head |
  |---|---|
  | `pitch` | 151 (Pitch 89 + PitchDrum 62) |
  | `velocity` | 32 |
  | `duration` | 64 |
  | `program` | 129 |
  | `position` | 96 |
  | `bar` | 1 |
  | `tempo` | 32 |
  | `timesig` | 9 |
  | `rest` | 32 |

- **Calling the two methods separately — no issue (FACT).** This is exactly what
  `reconstruct()` does ([infer.py:268-269](../../../midit/infer.py#L268-L269)). Two things to replicate in a custom decoder:
  1. Wrap in `torch.no_grad()` and call `model.eval()` (the pipeline does both — [infer.py:267](../../../midit/infer.py#L267), model set to eval by the caller).
  2. **The current decoder uses the *oracle* token type**: for each masked slot it picks which head to read from `type_for_id[original_tokens[pos]]` ([infer.py:286](../../../midit/infer.py#L286)), then `argmax` on that head ([infer.py:295](../../../midit/infer.py#L295)). **This is the key design point for the ablation** — sampling variants should replace only the read-out (argmax → temperature/top-k/top-p/iterative), keeping the same oracle-type + Position scaffolding, so results are comparable to the thesis's main numbers. A separate, stronger ablation could also drop the oracle (predict the type), but that's a bigger change — flag as a distinct condition.

**INFERENCE — where each decoding strategy plugs in:** everything except the
read-out is reusable. `argmax` (baseline), `temperature`, `top-k`, `top-p`, and
`MaskGIT` (iterative: fill high-confidence slots first, re-mask the rest, re-run
`model()` for K steps) all operate on the per-head logits at masked positions —
i.e. they only change lines analogous to [infer.py:295-305](../../../midit/model/masked_transformer.py) inside a copied `reconstruct`. MaskGIT needs the forward pass re-run per iteration (K× cost).

---

## Q3 — Metrics compatibility (`compute_muspy_metrics.py`)

**All three P-I metrics operate on `symusic.Score` objects loaded from MIDI files,
cropped by tick ranges — NOT on tokens (FACT).**

- **CP** — `compute_CP(orig, recon, bar_range)` ([compute_infilling_metrics.py:198](../../../scripts/Evaluation_Infilling/compute_infilling_metrics.py#L198)): duration-weighted 12-bin chroma cosine per masked bar, averaged. Inputs are `symusic` Scores + tick `bar_range`.
- **GS** — `compute_GS(orig, recon, bar_range, tpq, grid_res=16)` ([compute_infilling_metrics.py:224](../../../scripts/Evaluation_Infilling/compute_infilling_metrics.py#L224)): per-bar Jaccard of 16th-note onset grids.
- **PCHE** — `compute_PCHE(orig, recon, bar_range)` ([compute_infilling_metrics.py:255](../../../scripts/Evaluation_Infilling/compute_infilling_metrics.py#L255)): |ΔH| of pitch-class histograms.
- The batch driver `process_experiment` reads `original.mid / masked.mid /
  reconstructed.mid` from a seed dir, detects masked bars via `_detect_masked_bars`
  (original-vs-masked note presence), and computes P-I + muspy S-I
  ([compute_muspy_metrics.py:284-329](../../../scripts/Evaluation_Infilling/compute_muspy_metrics.py#L284-L329)). All file-based.

**Can I compute from raw token IDs without MIDI?** **Not with the existing
functions** — they take `symusic` Scores loaded from `.mid` via `_load`
([compute_infilling_metrics.py](../../../scripts/Evaluation_Infilling/compute_infilling_metrics.py)). Two viable approaches:

- **Recommended (lowest risk, comparable to thesis): go through MIDI files.**
  For each condition, `save_midi(tokens, tokenizer, path)` ([infer.py:365](../../../midit/infer.py#L365)) → then either call `_load` + `compute_CP/GS/PCHE` directly with the masked-bar tick range, or (simplest) write the `original.mid / masked.mid / reconstructed.mid` trio per condition and reuse `process_experiment`/`compute_muspy_metrics.py` unchanged. This guarantees the ablation numbers are computed by the **exact same code** as the thesis's main table — the safest choice for a defensible comparison.
  - **⚠ Reuse the fixed tpq-aware crop, not the raw one.** We already established the tick-based crop mis-aligns across differing tpq; for MIDIT-vs-MIDIT (same tpq on both sides) it's fine, but keep every condition at the **same tpq** (all are MIDIT REMI+ detok = tpq 16), so `compute_muspy_metrics.py` is safe here.
  - **I/O cost (INFERENCE):** 550 samples × 3 MIDI files (orig/masked/recon) ≈ **1,650 small MIDI writes** (+ reads at metric time). Each MIDI is a few KB and `save_midi` is sub-second (the 989-sample generation run wrote 3×989 files comfortably). Estimate **well under 5 minutes of I/O** total; negligible vs model inference. Note: `orig.mid`/`masked.mid` are **identical across the 11 conditions** for a given seed — write them **once per seed** (50×2 files) and only `reconstructed.mid` per condition (550), i.e. ~650 writes.

- **Optional (faster, more code): compute inline from tokens.** You would detokenize
  each token sequence to a `symusic.Score` in memory (`tokenizer(TokSequence(ids))`,
  as `save_midi` does at [infer.py:371](../../../midit/infer.py#L371)) — **without** writing to disk — then call `compute_CP/GS/PCHE(orig_score, recon_score, bar_range)`. Data needed from the token sequence: (a) the **masked-bar tick range** (derive from `bar_positions` → the tokenizer's tick grid, or reuse `_detect_masked_bars` on in-memory scores), (b) both `Score` objects. This avoids disk but re-implements the plumbing and risks diverging from the thesis pipeline. **Not recommended for a thesis ablation** — parity with the main table matters more than a few minutes of I/O.

**Verdict:** **go through MIDI files and reuse `compute_muspy_metrics.py`** (or its
CP/GS/PCHE functions) unchanged. I/O is negligible; parity with the thesis is the
priority.

---

## Q4 — SLURM template (extracted from existing scripts)

The canonical evaluation/generation job is `slurm/generate_samples.sh` (runs
`generate_samples_batch.py`). Extracted pattern (**FACT**):

- **GPU:** `#SBATCH --partition=allgroups`, `#SBATCH --gres=gpu:rtx:1` ([generate_samples.sh:8,10](../../../slurm/generate_samples.sh#L8)). (Other jobs use `gpu:a40:1` or `gpu:l40s:1`; `rtx` is what the generation job used.)
- **Resources:** `--mem=32G`, `--cpus-per-task=2`, `--ntasks-per-node=1`, `--time=3-00:00:00` ([:5-9](../../../slurm/generate_samples.sh#L5)).
- **Env activation:** **not conda** — a **uv/venv**: `cd ~/MIDIT` → `source .venv/bin/activate` → `export PYTHONPATH=/home/zanettisim/MIDIT:$PYTHONPATH` ([:12-14](../../../slurm/generate_samples.sh#L12)).
- **Paths (cluster):**
  - checkpoint: `/nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_infilling/infilling-epoch=05-val_loss=8.5836.ckpt`
  - data_dir: `/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k`
  - output: `/nfsd/tesi/zanettisim/MIDIT/outputs/...`
- **Logs:** `#SBATCH --output=logs/out/%j.txt`, `#SBATCH --error=logs/err/%j.txt` ([:3-4](../../../slurm/generate_samples.sh#L3)) — relative to `~/MIDIT`, so `logs/out` and `logs/err` must exist.

**Proposed job header for the ablation (a GPU node, ~550 recon passes; MaskGIT adds
K× per iterative condition):**
```
#SBATCH --job-name=midit-decoding-ablation
#SBATCH --output=logs/out/%j.txt
#SBATCH --error=logs/err/%j.txt
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:rtx:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
cd ~/MIDIT && source .venv/bin/activate
export PYTHONPATH=/home/zanettisim/MIDIT:$PYTHONPATH
```
(`.venv`, not conda; note the DEI login node forbids running Python — must be
`sbatch`, not the login shell.)

---

## Q5 — Flagged risks

1. **Oracle-type dependency (methodological, highest priority).** The current
   decoder reads the head chosen by the *ground-truth* token type
   ([infer.py:286](../../../midit/infer.py#L286)). Every decoding condition must keep this identical (only change the read-out), or the ablation confounds "decoding strategy" with "type selection". State this explicitly in the thesis; optionally add one non-oracle condition as a separate row.
2. **Determinism / seeding (INFERENCE).** argmax is deterministic; sampling
   conditions (temperature/top-k/top-p/MaskGIT) are **stochastic** → set and
   record a fixed `torch.manual_seed` per (seed, condition) so the table is
   reproducible. Without it the ablation isn't repeatable.
3. **`get_val_files` reproducibility.** It reselects the same 50 val files only if
   `--data_dir` points at the **same** `preprocessed_v2_50k` (song-level split,
   seed 42, [infer.py:133-139](../../../midit/infer.py#L133-L139)). Use the same data_dir as the thesis; don't re-preprocess.
4. **Sequence-length guard.** `generate_samples_batch` skips sequences `>= 8000`
   tokens ([generate_samples_batch.py:80](../../../scripts/inference/generate_samples_batch.py#L80)); the model's `max_seq_length` is 8192. For 50 fixed val files this is unlikely to bite, but a very dense 8-bar chunk could exceed it — add the same guard and log skips rather than crash on the position embedding.
5. **MaskGIT cost & correctness.** Iterative decoding re-runs `model()` K times per
   sample (K× the ~989-run inference cost per condition). Budget `--time`
   accordingly. Also: MaskGIT confidence must be taken from the **chosen (oracle)
   head's** softmax at each masked slot, and re-masking must only touch masked
   content slots (never the preserved Position/Bar/Tempo/TimeSig/Rest scaffold).
6. **CUDA / memory.** Batch size 1 per sample (as in `reconstruct`) → trivial VRAM
   (~42.6M params, one 8-bar chunk). No OOM risk on an rtx GPU. Load the checkpoint
   **once**, loop conditions/seeds in-process (as `generate_samples_batch` does),
   don't reload per condition.
7. **Device handling.** Follow the pipeline: `model.eval()`, `model.to(device)`,
   `torch.no_grad()`, and move each input with `.to(device)` ([infer.py:405-408](../../../midit/infer.py#L405), [:268](../../../midit/infer.py#L268)). CPU fallback works but is slow for MaskGIT.
8. **Metric-parity / crop.** Use the **same** `compute_muspy_metrics.py` crop path
   for every condition; all conditions are MIDIT REMI+ (tpq 16 on both orig and
   recon), so the tpq-mismatch bug does **not** apply here — but keep orig/masked
   fixed per seed and only vary reconstructed.
9. **Silent import cost.** Importing `midit.infer` pulls torch + miditok; on the
   broken-OpenSSL local machine `miditok` import crashes, but on the cluster `.venv`
   it's fine — run only via `sbatch`.
10. **CSV schema.** Write one row per (seed, condition) with `cp,gs,pche`
    (+ optionally muspy S-I) so it drops into the same aggregation as the thesis
    table; include a `condition` and `seed` column and a fixed decoding seed.

---

## Proposed conditions (11, matching the "50 seeds × 11 conditions" framing)

INFERENCE / proposal (for your review — not implemented):
`argmax` (baseline) · `temp=0.8` · `temp=1.0` · `temp=1.2` · `top-k=5` ·
`top-k=20` · `top-p=0.9` · `top-p=0.95` · `MaskGIT K=4` · `MaskGIT K=8` ·
`MaskGIT K=12`. All keep the oracle-type + Position scaffolding; only the
read-out differs. Adjust to taste before Phase 2.

---

**Bottom line:** the plan is low-risk. All needed utilities are importable and
side-effect-free; the model API is exactly as assumed (hidden → dict of 9 heads);
metrics are file-based but cheap to feed via `save_midi` + reuse of
`compute_muspy_metrics.py` (guaranteeing parity with the thesis table); the SLURM
pattern is `.venv` + `allgroups` + `gpu:rtx` with `/nfsd` paths. The two things to
get right are **keeping the oracle-type read-out identical across conditions** and
**seeding the stochastic decoders**. No file was modified.
