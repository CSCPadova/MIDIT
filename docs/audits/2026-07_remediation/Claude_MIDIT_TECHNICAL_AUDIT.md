# MIDIT Codebase Audit

**Legend:** [FACT] = directly read · [INFERENCE] = deduced from code · [HYPOTHESIS] = needs runtime confirmation (with the check named)

**Scope read:** the 7 module groups requested + `model_factory.py`, `config_infilling_v2.yaml`. Auxiliary scripts (`merge_metrics.py`, `run_extreme_metrics.py`, the `Analyze_dataset/*`, preprocessing) were **not** deep-read — flagged as *not verified* where relevant.

---

## 1. `midit/model/masked_transformer.py`

**Purpose & data flow** [FACT]
`MaskedTransformerREMI(L.LightningModule)`. Forward: `midi_tokens (B,S)` → `token_embedding` (nn.Embedding `total_vocab×512`) + `position_embedding` (learned, `max_seq_length×512`) → `LayerNorm` → `TransformerEncoder` (12 layers, `norm_first=True`, GELU, `batch_first=True`) → returns hidden `(B,S,512)`. Heads applied separately by `multi_head_output`.

**How it actually works** [FACT]
- Masking is **done in the dataset, not here**. `training_step`/`validation_step` read the already-masked `midi_tokens` and `mask_labels` from the batch; the comment at `masked_transformer.py:457` confirms "no second masking applied here." This is the fix for the historical double-masking bug.
- `apply_bar_masking()` (`masked_transformer.py:300-369`) is **dead code** — never called anywhere. [FACT] (grep: no call sites).
- Padding mask: `~midi_attention_mask.bool()` → `src_key_padding_mask` where `True`=ignore. attention_mask uses `True`=real token, so the inversion is correct. [FACT]
- Phase-2 text path (`masked_transformer.py:399-431`) exists but is unreachable: `build_model` hardcodes `phase=1` and `text_encoder=None`. [FACT]

**Hidden assumptions / magic numbers** [FACT]
- Default `vocab_sizes` (lines 116-127) and `special_token_ids` (`pad=515…mask=518`, lines 132-137) are **V1 legacy defaults**. Inert in normal runs because `build_model` always passes explicit V2 values, but they are baked into `save_hyperparameters()` and would silently be used if the model were ever instantiated without them.
- Default `lambda_weights` (lines 141-152) differ from `conf/config.yaml` — also inert (overridden by factory).
- `configure_optimizers` cosine schedule uses `total_steps = self.num_training_steps if >0 else trainer.estimated_stepping_batches` (`masked_transformer.py:548`). The comment notes `estimated_stepping_batches` returns `inf` under `DataLoaderWrapper` — hence the manual value. `lr_lambda` returns a 0-dim `torch.cos` tensor (line 555) — works, slightly unusual.

**Strengths**
- Clean separation: masking lives in the data layer; model is a plain bidirectional encoder. `norm_first=True` (pre-LN) is the stable/defensible choice.
- The `num_training_steps` injection is a correct workaround for the `inf` scheduler bug.

**Weaknesses / risks**
- **Paradigm:** non-autoregressive MLM — a single forward pass predicts all masked slots **independently** given fixed context. No joint modeling of masked-token interactions. [INFERENCE]
- Learned absolute position embeddings capped at `max_seq_length=8192`; sequences are truncated at `max_tokens` in the dataset, so no OOB, but no length extrapolation.
- Vestigial V1 defaults are a latent footgun if anyone constructs the model directly.

---

## 2. `midit/model/multi_head_output.py`

**Purpose & data flow** [FACT]
9 independent `Linear(512, vocab_t)` heads after a shared `LayerNorm`+`Dropout`. Input `(B,S,512)` → dict of logits `{type:(B,S,vocab_t)}`. Heads: pitch(151), velocity(32), duration(64), program(129), position(96), bar(1), tempo(32), timesig(9), rest(32) → sum+4 = **550** ✓ (matches vocab). [FACT/INFERENCE — arithmetic]

**How the loss actually works** [FACT] (`multi_head_output.py:177-226`)
Per type: `cross_entropy(logits, targets, reduction='none', ignore_index=-100)`, reshaped to `(B,S)`, then:
```
valid_mask = mask & (targets != -100)
loss_mean  = (loss * mask).sum() / valid_mask.sum().clamp(min=1)
weighted   = lambda_weights.get(type,1.0) * loss_mean
total     += weighted
```
- **Denominator counts only masked, valid tokens of THIS type** — the documented "denominator dilution" fix. Correct: because `ignore_index` zeroes ignored positions, `loss*mask` numerator equals `loss*valid_mask` numerically, so numerator/denominator are consistent. [FACT/INFERENCE]
- `total_loss` initialized as Python `float 0.0`; becomes a tensor after the first type. Safe because pitch is always present. [FACT]

**Hidden assumptions**
- `lambda_weights` with value `0.0` (bar/tempo/timesig, per config) makes `weighted_loss=0` → **those heads receive no gradient and are never trained.** [INFERENCE] Fine for infilling (they're preserved, not predicted), but the bar/tempo/timesig logits are effectively random.
- Head ordering in `class_to_name` must match `_create_targets_dict` ordering exactly; pitch head concatenates Pitch then PitchDrum — this ordering is duplicated in 3 files (dataset, infer, heads) and must stay in sync. [FACT] (fragility)

**Strengths**
- Per-type normalization is the right call for the extreme class imbalance of REMI+.
- `class_weights` machinery is present and Lightning-safe (registered buffers), though **not enabled by any config read** (see §7).

**Risks for validity**
- If `mask_labels` and `targets != -100` ever disagree, the denominator would silently diverge from the numerator's support. They're constructed consistently in the dataset (`lakh_dataset.py:400-402`), so currently safe.

---

## 3. `midit/data/lakh_dataset.py` + `midit/data/masking.py`

**Purpose & data flow** [FACT]
Loads preprocessed `.pt` (`{tokens, bar_positions, metadata}`), chunks by bars (`bars_per_chunk=8`), masks, and emits `{input_ids, original_tokens, mask_labels, targets, bar_positions, metadata}`. `collate_fn` pads with 0 (input/orig), `False` (mask), `-100` (targets), and builds `attention_mask`.

**Train/val/test split** [FACT] (`lakh_dataset.py:49-92`)
- **Song-level** split: strips `_shift±N` suffix, groups by stem, `np.random.seed(42)`, shuffles stems, 80/10/10. **All augmentations of a song land in the same split** — prevents augmentation leakage. Well done. [FACT]
- `infer.py get_val_files` reproduces the identical split with `RandomState(42)` and the same 80/10/10 — consistent selection between training and inference. [FACT]

**Masking strategies** [FACT]
- Routing (`lakh_dataset.py:214-248`): `masking.strategy` key wins; else legacy `bar_level` bool. `bar_fill`→`BarFillMasking(seed=1234 if val else None)`; `bar_level`→`BarLevelMasking`; else `TokenLevelMasking`.
- `BarFillMasking` (`masking.py:396-579`): picks a contiguous 1–`max_infill_bars` window containing ≥1 note, masks **only** note tokens (`Pitch/PitchDrum/Velocity/Duration/Program`), preserves structural tokens. Falls back to `TokenLevelMasking` if `bar_positions` is `None`/<2. If no note tokens found, returns **unmasked** (mask_labels all False) → that chunk contributes **zero loss**. [FACT]
- `TokenLevelMasking` uses Python `random` (80/10/10 BERT). `random.randint(0, vocab_size-1)` can sample special tokens (0-3) for the 10% random branch. [FACT] (minor)

**Hidden assumptions / determinism** [FACT/HYPOTHESIS]
- Val masker seed is **1234**, not 42 (the "seed=42" is the *split* seed; the *val-mask* seed is 1234). [FACT — discrepancy vs description, see §synthesis 4]
- **[HYPOTHESIS — validity-relevant]** The claim (comment `masking.py:224-226`) that the val task is "identical across epochs" is likely **not** true with `persistent_workers=True` (set in both train scripts). `BarFillMasking._rng = random.Random(1234)` is created once and **never re-seeded per epoch/sample**; persistent workers keep the dataset (and its advancing RNG) alive across epochs, so a given val sample gets a *different* mask each epoch. Run-to-run epoch-1 comparisons are still identical. **Check:** in a worker, log `mask_labels.nonzero()` for a fixed val index across two epochs; if they differ, the claim is violated. Impact is limited to val_loss noise, not test metrics.
- `_create_targets_dict` is an O(seq_len) Python loop per `__getitem__` (`lakh_dataset.py:535`) — CPU cost, mitigated by the O(1) `token_to_class` dict.
- `preload=True` loads **every** `.pt` into RAM in `__init__` (`lakh_dataset.py:259-295`) — explains the 120 GB SLURM memory need. With `num_workers>0` each worker forks a full copy (copy-on-write on Linux). [FACT/INFERENCE]

**Strengths**
- Song-level split is the single most important correctness decision here and it's done right, and mirrored at inference.
- Chunk index precomputed once; masking clean and interface-consistent across 4 maskers.

**Risks for validity**
- Chunks with no note tokens → silently zero-loss (BarFill) — harmless but slightly dilutes effective batch. [INFERENCE]
- `_chunk_by_tokens` uses `random.randint` for the start — nondeterministic even in val when token-based (not used in current bar_based config). [FACT]

---

## 4. `train_phase1.py` / `train_infilling.py` + `model_factory.py`

**LR / total-steps computation** [FACT] (`train_phase1.py:106-119`, `train_infilling.py:110-134`)
```
effective_batches = min(int(limit), len(loader))   # limit int
steps_per_epoch   = effective_batches // accumulate
total_steps       = steps_per_epoch * max_epochs
```
- This **correctly** divides by `accumulate_grad_batches`. Since the scheduler uses `interval:'step'`, Lightning steps it once per **optimizer** step (every `accumulate` batches). So `total_steps` matches the true optimizer-step count and cosine decay reaches 0 at end. The historical accumulate bug appears **fixed**. [FACT/INFERENCE] Phase-1 config: 25000//4×50 = 312,500 (matches the config comment). [FACT]
- Edge case [INFERENCE]: if `limit` is a fraction and the loader is shorter than expected, or training early-stops, `total_steps` overshoots and cosine won't fully anneal — not a bug, a sensitivity. A floor with a non-divisible `limit % accumulate` could also mis-count the final partial-accumulation step. **Runtime check:** run one smoke epoch with `limit_train_batches=5`, `accumulate_grad_batches=4`, then compare `trainer.global_step` to the printed `total_training_steps` — no real training job needed.

**Checkpoint loading behavior (which hparams are re-read vs inert)** [FACT] (`train_infilling.py:142-154`)
- `MaskedTransformerREMI.load_from_checkpoint(...)` restores **all** `save_hyperparameters()` values (vocab_sizes, d_model, lambda_weights, weight_decay, dropout, etc.) from the Phase-1 checkpoint.
- Only **three** are overridden post-load: `model.lr`, `model.warmup_steps`, `model.num_training_steps`. Everything else in `config_infilling.yaml`'s `model:`/`lambda_weights:`/`train.weight_decay` blocks is **inert during fine-tuning** — correctly documented in `config_infilling_v2.yaml` comments. [FACT — good]
- Consequence: `lambda_bar/tempo/timesig=0` etc. carry over from the Phase-1 checkpoint; you cannot change loss weights or `weight_decay` at fine-tune time via config, but changing `train.lr` does work. [INFERENCE]

**Trainer wiring** [FACT]
- `devices=1` hardcoded in both scripts — `cfg.train.devices`/DDP path is dead; the CLAUDE "DDP with 2 GPUs" task is **not implemented**. [FACT — discrepancy]
- `DataLoaderWrapper` remaps `input_ids→midi_tokens`, `attention_mask→midi_attention_mask` (`train_phase1.py:17-42`). This is why `collate_fn` emits `input_ids` while `training_step` reads `midi_tokens` — the mismatch is bridged here. [FACT — resolves an apparent key mismatch]
- `logs_dir` resolution (`train_phase1.py:155-160`): checks `cfg.data.logs_dir` then `cfg.logging.logs_dir` — **neither exists in `config.yaml`** (logs_dir lives under `paths:`), so Phase-1 silently defaults to `'logs'`. `config_infilling_v2.yaml` *does* set `logging.logs_dir`, so it's active only there. [FACT — `paths.logs_dir` is dead]

**`model_factory.build_model`** [FACT]
- Two loud invariant checks before allocation: (a) contiguous IDs (`max_id+1==len(vocab)`), (b) `sum(vocab_sizes)+4==len(vocab)`. Prevents the historical silent embedding-resize that broke `load_from_checkpoint`. Well-engineered. Only called by Phase-1 (`train_infilling` bypasses it via `load_from_checkpoint`). [FACT]

**Strengths:** explicit step math, loud vocab invariants, documented inert-hparam behavior.
**Weaknesses:** two near-identical training scripts (duplicated `DataLoaderWrapper`); `devices=1` hardcode contradicts stated scaling plan; batch_size/accumulate defaults in code (`2`/`4`) differ from the config that's actually loaded (`8`/`4`) — the config wins, but the defaults mislead.

---

## 5. `midit/infer.py` + `scripts/infer_from_midi.py`

**Pipeline** [FACT]
`load_from_checkpoint` → build V2 tokenizer (identical config to dataset) → `get_val_files` (same 80/10/10 split) → `extract_chunk` (first 8 bars) → `apply_bar_mask` (mask note tokens in `--mask_bars`, default `[3,4,5]`) → `reconstruct` → `save_midi` (original/masked/reconstructed).

**What is masked vs preserved** [FACT] (`infer.py:178-214`)
- Masked: `Pitch, PitchDrum, Velocity, Duration, Program` inside the selected bars.
- **Preserved in-place:** `Position, Bar, Tempo, TimeSig, Rest` — the entire rhythmic/structural scaffold.

**Token-type selection per masked slot — the critical mechanism** [FACT] (`infer.py:277-308`)
```
orig_id   = original_tokens[pos]           # ← ground truth
head      = type_for_id[orig_id]           # ← ORACLE: which head to use
class_idx = logits[head][0,pos].argmax()   # value within that type
```
- The reconstruction reads the **original token's type** to decide which of the 9 heads to query. This is **oracle token-type information**: the model is told "this masked slot is a Pitch / a Velocity / a Duration / a Program" and only predicts the *value*. [FACT]
- Out-of-range predictions and unrecognized types fall back to `orig_id` (`infer.py:298-300`) — the model **can never delete a note or emit a wrong-arity slot**; the slot count and slot types are fixed to the original. [FACT]
- Decoding is deterministic argmax, single forward pass, all masked slots filled independently. [FACT]

**Consequences for "infilling"** [INFERENCE]
- Output length = input length exactly; **fixed-length**, cannot generate variable numbers of notes.
- Because Position tokens are preserved and note count is fixed, **onset times are preserved by construction** → see §6 GS risk.
- Combined Pitch+PitchDrum head: a masked drum note (`PitchDrum_`) and a masked melodic note (`Pitch_`) both query the `pitch` head; argmax over 151 classes can cross the drum/melodic boundary, and `Program` is *also* masked/predicted separately, so an inconsistent (program=melodic, pitch=drum) pair is possible before MidiTok detokenization. [INFERENCE — edge case, worth a spot check]

**Risks** [FACT]
- `infer.py` does not guard the 8-bar chunk against `max_seq_length`; `generate_samples_batch.py` instead skips lengths `>=8000`, not `>8192`.
- `save_midi()` tries two APIs and can return `False`; the batch generator doesn't check the return value and may count a seed with a failed export as success (`scripts/generate_samples_batch.py:101`).
- `scripts/infer_from_midi.py --skip_filters` doesn't actually bypass F1/F2/F3: it calls `_process_file(..., False)` and the preprocessor still applies the filters; the option only suppresses the subsequent exception, but the `.pt` may not exist (`scripts/infer_from_midi.py:48`). [INFERENCE — needs a runtime check of preprocess_v2 signature; not deep-read]

**`infer_from_midi.py`** [FACT] runs `preprocess_v2._process_file` on a raw `.mid` (shift=0), then the same mask→reconstruct path.

**Strengths:** structural-scaffold infilling is a legitimate, well-defined task; tokenizer config is duplicated *identically* to the dataset (checked line-by-line, matches). Deterministic → reproducible.
**Weaknesses/risks:** the **oracle type selection** is the dominant validity caveat — reported reconstruction accuracy and downstream metrics assume knowledge a true generative infiller would not have. Must be stated explicitly in the thesis.

---

## 6. `generate_samples_batch.py` + `Evaluation_Infilling/*`

**`generate_samples_batch.py`** [FACT] loads the checkpoint once, loops seeds, reuses `infer.py` functions, skips seeds whose `reconstructed.mid` exists (without re-checking checkpoint/data/mask_bars), skips sequences ≥8000 tokens. Seeds index into the val split; `seed+offset` search for ≥8-bar files can converge different seeds onto the same song; no seed→source manifest is saved. Purely a batch driver.

**Metric definitions** [FACT]
- **CP** (`compute_infilling_metrics.py:198`): per-bar duration-weighted 12-chroma cosine, averaged over masked bars. Empty bar → 0 (penalizes). Genuine harmonic-content measure.
- **GS** (`compute_infilling_metrics.py:224`): per-bar Jaccard of 16th-note onset grids. **Both-empty → 1.0** (`_jaccard` line 173) — this contradicts the docstring which says empty bars score 0; code wins.
- **PCHE**: `|H_orig − H_recon|` of pooled pitch-class histograms over masked bars. Note: two very different distributions with equal entropy give PCHE=0.
- **JS similarity** (`js_similarity.py:86`): `100·exp(−2·mean_JS)` over Gaussians fit to `polyphony_rate, groove_consistency, scale_consistency` (model_agg vs ref_agg).

**Cropping** [FACT]
- CP/GS/PCHE and the JS `model_agg` crop to the **masked bar region**, auto-detected via `_detect_masked_bars` (bars with notes in original, zero in masked) (`compute_muspy_metrics.py:88`). Robust because masked.mid has no notes in masked bars.
- `_crop_music_to_range` filters by onset and re-times to tick 0 (`compute_muspy_metrics.py:196`); does **not** clip note durations (documented — the 5 metrics don't read note ends).

**Risks for validity** — concrete ones:

1. **[FACT → INFERENCE] GS is near-trivial under this masking.** Position tokens are preserved and `reconstruct` never drops/adds notes, so reconstructed onsets ≈ original onsets → Jaccard ≈ 1.0 by construction. GS mostly measures the masking scheme, not the model. Report GS with this caveat or drop it.

2. **[FACT] `groove_consistency` is called with inconsistent `measure_resolution`:**
   - Reference & model-fallback path: `groove_consistency(music, 4)` (`Matteo/muspy_metrics.py:86`, `compute_muspy_metrics.py:267`).
   - Model **cropped** path: `groove_consistency(music, resolution)` where `resolution=music.resolution` (`compute_model_agg.py:95`).
   So the JS `groove_consistency` compares model (measure_resolution=`resolution`, e.g. 480) against reference (measure_resolution=4) — **different grids, not comparable.** This directly corrupts one of the 3 JS metrics. Strong finding; verify by printing the two args at runtime.

3. **[INFERENCE] Population mismatch in JS similarity.** `ref_agg` is computed on **whole** LMD MIDI files (`compute_reference_agg.py`), while `model_agg` is computed on **3-bar cropped** masked regions. Comparing fragment statistics to full-song statistics biases the JS score.

4. **[FACT] Full-piece crop-fallback is untracked.** When bar detection fails, metrics fall back to the whole reconstructed file, which includes unmasked bars identical to the original → inflates CP/GS. The CSV has no column flagging fallback rows, so they can't be excluded post-hoc (`compute_muspy_metrics.py:284`).

5. **[FACT] `_js_divergence_gaussian` fits Gaussians** to metrics bounded in [0,1] — tails leak outside the domain; approximation inherited "verbatim from mxlGPT," acceptable but worth a footnote.

6. **[INFERENCE] Averaging inflation** (general): any average that includes preserved/trivial structural tokens is inflated. Infilling `compute_accuracy` correctly restricts to note tokens; the Phase-1 `evaluate_phase1_accuracy_DIRECT` average (per CLAUDE) includes bar≈100%/timesig≈97% — that ~49% figure is inflated and shouldn't be compared to infilling numbers.

---

## 7. Config audit — every key: ACTIVE or DEAD

**[FACT]** unless noted. "DEAD" = not read by any code path I traced; "conditional" = read only under a strategy not currently selected. Table refers to the operational entrypoints `train_phase1.py` / `train_infilling.py`.

| Key | Read at | Status |
|---|---|---|
| `tokenizer.*` (whole block) | — (tokenizer hardcoded in `lakh_dataset.py:98-111` & `infer.py:38-51`) | **DEAD** (self-documented) |
| `chunking.strategy` | lakh_dataset:256 | ACTIVE |
| `chunking.bars_per_chunk` | lakh_dataset:257 | ACTIVE |
| `chunking.tokens_per_chunk` | lakh_dataset:483 | **DEAD** (only if `token_based`; config=`bar_based`) |
| `chunking.max_tokens` | lakh_dataset:378, factory:114 | ACTIVE |
| `chunking.preprocessed_dir` | lakh_dataset:37 | ACTIVE |
| `chunking.raw_data_dir` | — | **DEAD** |
| `chunking.min_tokens_per_bar` / `filter_empty_bars` | — (grep: no read) | **DEAD** |
| `masking.mask_prob` | factory:115, TokenLevelMasking:245 | ACTIVE (phase1); **DEAD in infilling** (BarFill ignores it) |
| `masking.bar_level` | lakh_dataset:221, factory:116 | ACTIVE |
| `masking.mask_strategy` | lakh_dataset:246/231 | ACTIVE |
| `masking.strategy` | lakh_dataset:216 | ACTIVE (only defined in infilling configs) |
| `masking.max_infill_bars` | lakh_dataset:230 | ACTIVE (infilling) |
| `masking.bar_mask_prob` / `type_mask_ratio` | lakh_dataset:239-240 | **DEAD** (only if `bar_level:true`; config=false) |
| `lambda_weights.*` | factory:99 | ACTIVE (phase1); **inert in infilling** (restored from ckpt) |
| `class_weights.*` | factory:100 (`if 'class_weights' in cfg`) | **DEAD** — key absent from `config.yaml`; whole class-weights subsystem unreachable |
| `model.d_model/nhead/num_encoder_layers/dim_feedforward/dropout` | factory:110-113 | ACTIVE (phase1); **inert in infilling** (ckpt hparams) |
| `model.vocab_size` | — (computed dynamically) | **DEAD** |
| `model.max_seq_length` | — (factory uses `chunking.max_tokens`) | **DEAD** |
| `model.special_tokens` | — (factory hardcodes `{pad:0…mask:3}`) | **DEAD** |
| `train.batch_size/epochs/accelerator` | train scripts | ACTIVE |
| `train.devices` | — (Trainer hardcodes `devices=1`) | **DEAD** |
| `train.lr` | train scripts | ACTIVE |
| `train.weight_decay` | factory:120 | ACTIVE (phase1); **inert in infilling** |
| `train.warmup_steps` | factory:121 / train_infilling:128 | ACTIVE |
| `train.gradient_clip_val/precision/accumulate_grad_batches` | Trainer | ACTIVE |
| `train.limit_train_batches/limit_val_batches` | Trainer + step math | ACTIVE |
| `train.save_top_k` | ModelCheckpoint | ACTIVE |
| `train.monitor` | — (ModelCheckpoint hardcodes `monitor="val_loss"`) | **DEAD** |
| `train.phase` | — (factory hardcodes `phase=1`) | **DEAD** |
| `train.val_check_interval` / `log_every_n_steps` | Trainer | ACTIVE |
| `train.checkpoint_path` | train_infilling:63 | ACTIVE (infilling) |
| `logging.use_wandb` / `wandb_project` | train scripts | ACTIVE (wandb_project DEAD in infilling — name hardcoded) |
| `logging.logs_dir` | train_infilling:160 | ACTIVE only in `config_infilling_v2`; absent (→default 'logs') elsewhere |
| `paths.*` (all) | — (grep: no `cfg.paths` read) | **DEAD** — incl. `paths.logs_dir`, `paths.checkpoints_infilling` |

---

# Final Synthesis

## 1. End-to-end data flow (text diagram) [FACT unless noted]

```
raw .mid
  └─(preprocess_v2, offline; ±5 pitch-shift aug)→ {tokens, bar_positions, metadata}.pt

TRAINING
  .pt files ──song-level 80/10/10 split (seed 42, stem-grouped)──▶ LakhDataset
    └─ bar_based chunk (8 bars) ─▶ masking:
         Phase 1  : TokenLevelMasking 15% BERT 80/10/10
         Phase 1b : BarFillMasking (1–3 contiguous bars, note tokens only; val seed 1234)
    └─ targets dict (per-type class idx at masked positions, else −100)
  collate_fn (pad 0 / False / −100) ─▶ DataLoaderWrapper (input_ids→midi_tokens)
  ─▶ Embedding+PosEmb+LN ─▶ 12L TransformerEncoder ─▶ 9 Linear heads
  ─▶ per-type masked-denominator CE × λ ─▶ AdamW + warmup→cosine (manual total_steps)

INFERENCE (infer.py / generate_samples_batch)
  val .pt ─▶ first 8 bars ─▶ apply_bar_mask (mask note tokens in bars 3,4,5;
                              Position/Bar/Tempo/TimeSig/Rest preserved)
  ─▶ single forward ─▶ per masked slot: ORACLE original type → head → argmax value
      (note count & onsets fixed to original; OOR → keep original)
  ─▶ MidiTok detokenize ─▶ original.mid / masked.mid / reconstructed.mid

EVALUATION
  crop to masked bars (detected: notes in orig, none in masked)
   ├─ CP/GS/PCHE  (recon vs orig, per masked bar)
   └─ muspy model_agg (cropped) vs ref_agg (FULL LMD files) ─▶ JS = 100·exp(−2·mean_JS)
```

## 2. Ten most important limitations (ranked)

Classification: (a) inherent to paradigm · (b) implementation choice · (c) bug/fragility

1. **(c) Oracle token-type selection at inference** — `infer.py:286`. Model is given each masked slot's ground-truth *type*; only predicts the value. Inflates reported accuracy vs a true infiller. *Result-validity critical.*
2. **(b) GS metric near-trivial** — preserved Position tokens + fixed note count ⇒ onsets ≈ identical ⇒ GS≈1 regardless of model quality (`infer.py:277-300`, `compute_infilling_metrics.py:224`).
3. **(c) `groove_consistency` measure_resolution mismatch** (4 vs `resolution`) between reference/fallback and cropped-model paths (`compute_model_agg.py:95` vs `Matteo/muspy_metrics.py:86`). Corrupts 1 of 3 JS metrics.
4. **(b) JS population mismatch** — 3-bar cropped model regions compared to whole-song reference statistics (`compute_reference_agg.py`).
5. **(a) Non-autoregressive independence** — masked slots predicted independently in one pass; no joint coherence, deterministic argmax → low diversity (`multi_head_output.py:188`, `infer.py:267-269`).
6. **(a) Fixed-length infilling** — cannot add/remove notes; note count and slot types locked to the original (`infer.py:298-300`).
7. **(c) Val-mask non-determinism across epochs** [HYPOTHESIS] — shared `random.Random(1234)` + `persistent_workers=True` (`masking.py:445`, `train_infilling.py:99`); contradicts the "identical across epochs" claim.
8. **(b) `devices=1` hardcoded** — DDP/multi-GPU scaling (a stated project goal) not implemented (`train_phase1.py:203`).
9. **(b) Pitch+PitchDrum share one head** — drum/melodic values can cross under argmax while Program is separately predicted ⇒ possible inconsistent notes (`infer.py:82-93`).
10. **(c) Untracked metric fallbacks + no seed manifest** — full-piece crop fallback can inflate CP/GS unflagged; batch generation can duplicate songs without saving seed→source (`compute_muspy_metrics.py:119`, `generate_samples_batch.py:63`).

## 3. Discrepancies between code and the project description

- **"seed=42 validation masking":** split seed is 42, but val-**mask** seed is **1234** (`lakh_dataset.py:233`). [FACT]
- **"DDP with 2 GPUs / torchrun":** not implemented — `devices=1` hardcoded. [FACT]
- **"15% token-level MLM":** code masks exactly `floor(0.15·L)` (≥1), not an independent Bernoulli per token (`masking.py:311`). [FACT]
- **CLAUDE `mask_prob`/beat_res "V1 vocab 518":** current runtime is V2, vocab **550**, `duration=64` — consistent with `sum+4=550`. The `masked_transformer.py` in-code defaults (vocab≈518, pad=515) are stale V1 and inert. [FACT]
- **"9-head, bar/timesig removed from loss":** confirmed — but tempo is **also** λ=0 in `config.yaml` (not just bar/timesig). [FACT — `config.yaml:70-72`]
- **`class_weights` "NEW!" feature:** fully coded but **unreachable** (no config defines `class_weights`). [FACT]
- **"Deterministic":** correct for argmax inference, but NOT for Phase-1 val masking, and not identical across epochs even for BarFill val. [FACT]

## 4. Short-term hardening (no retraining) vs future work

**Short-term (no retraining):**
- Fix the `groove_consistency` `measure_resolution` to a single constant everywhere (#3) — recompute JS. *Cheapest high-impact fix.*
- Recompute `ref_agg` on **cropped** reference segments matched to the masked-bar geometry, so JS compares like-with-like (#4).
- In the thesis, either **drop GS** or report it strictly as a sanity check, stating it is bounded ≈1 by the scaffold (#2).
- Make val masking index-deterministic: derive the seed from `(validation_seed, sample_index)` instead of a stateful advancing RNG; log the mask seed/indices (#7). Confirm with a two-epoch mask-diff check.
- Compute optimizer steps with true Lightning semantics and add a test for `limit % accumulate != 0`.
- In Phase 1b, print/save an explicit diff between checkpoint hparams and applied hparams; fail if a required YAML key is inert.
- Flag or fail metric rows that used the full-piece crop fallback; don't average them into masked-region stats.
- Save an immutable per-sample manifest: checkpoint hash, source file, split, chunk, mask bars, token count, export status.
- Add a `len(tokens) <= model.max_seq_length` guard to every inference entrypoint; fix or remove `--skip_filters`.
- Delete/annotate dead config keys (esp. `train.monitor/phase/devices`, `model.*` inert keys, `paths.*`) to prevent silent-ignore bugs (#10, #1 config).
- Add an inference **ablation without oracle type** to quantify how much accuracy the oracle contributes (#1) — analysis only, not retraining.

**Future work (needs retraining / new architecture):**
- Replace single-pass argmax with **iterative/masked-parallel decoding** to address independence & diversity (#5).
- **Variable-length infilling** (predict note count / structural tokens) so the model generates Position/Program arity rather than inheriting it (#6).
- Split **Pitch vs PitchDrum** heads, or condition the pitch head on predicted Program (#9).
- Training/evaluation **without the oracle token-type**, for a measure closer to real infilling.
- Implement genuine **DDP** scaling (#8).
- If Phase 2 (text conditioning) is a thesis goal, it needs a text–MIDI dataset, config, checkpoint policy and a dedicated training loop; today the FLAN-T5 encoder alone is not a complete experimental path.

---

**Not verified (out of read scope):** `merge_metrics.py`, `run_extreme_metrics.py`, `preprocess_v2.py` internals (quality filters F1/F2/F3, pitch-shift correctness, `_process_file` signature), `text_encoder.py`, and the `Analyze_dataset/*` scripts. Claims about preprocessing behavior rest on the CLAUDE description, not a code read — treat as unverified.
