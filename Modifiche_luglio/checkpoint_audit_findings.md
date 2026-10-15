# Checkpoint audit — findings (authoritative training config)

**Source:** `hyper_parameters` read directly from the two thesis checkpoints
(`checkpoint_config_audit.md`, run on the cluster). The checkpoint is the
authoritative record of what was actually trained; `conf/config.yaml` has since
been edited and **no longer describes the thesis run**.

Legend: **FACT** = read from ckpt/log · **INFERENCE** = arithmetic deduction ·
**OPEN** = needs one more log check.

---

## 1. Lambda weights — Table 4.1 is CORRECT (config.yaml is stale)

**FACT (both checkpoints, identical):**
```
pitch 3.0 · velocity 1.5 · duration 1.0 · program 1.5 · position 0.5
bar 0.0 · tempo 1.0 · timesig 0.5 · rest 1.0
```
This **matches Table 4.1** (Pitch 3.0, Bar 0.0, Tempo 1.0, TimeSig 0.5, Rest 1.0).

The current `conf/config.yaml` (bar 0.0, **tempo 0.0, timesig 0.0, rest 0.5**) does
**NOT** match the trained model — it was modified after training.

> **Correction:** an earlier analysis (based on the stale `config.yaml`) said
> Table 4.1 should be 0/0/0.5. That was wrong. **Table 4.1 stands as-is; do not
> change it.** The checkpoint (authoritative) confirms tempo 1.0, timesig 0.5,
> rest 1.0. Phase-1b inherited the same lambda via `load_from_checkpoint`.

## 2. Effective batch size = 8 — the thesis claim is CORRECT

**FACT:** `batch_size`, `accumulate_grad_batches`, `limit_train_batches` are
**not** in the checkpoint (Trainer/DataLoader params). But:

**FACT:** `num_training_steps = 2,207,550` (both checkpoints); `warmup_steps = 1500`;
`lr = 0.0003` (3e-4); `weight_decay = 0.01`.

**INFERENCE (exact arithmetic):** `2,207,550 = 44,151 × 50`, and
`353,210 train files ÷ 8 ≈ 44,151`. This is an exact match for:
- **batch_size = 8**
- **accumulate_grad_batches = 1**  → **effective batch = 8 × 1 = 8**
- **limit_train_batches = 1.0** (full dataset per epoch — the err logs confirm
  runs with `Trainer(limit_train_batches=1.0)`)
- **one chunk per file** (dataset length ≈ file count → not stride-1 bar chunking)
- **50 planned epochs** (best checkpoint saved at epoch 14)

> **Correction:** the earlier `epoch_semantics` analysis used `config.yaml`
> (`accumulate=4`, `limit=25000`) and concluded **effective batch = 32** and
> **200,000 chunks/epoch, ~0.7% coverage**. Those describe the **stale config**,
> NOT the thesis checkpoint. The thesis run used **effective batch = 8** and
> **full-dataset epochs**.

## 3. Epoch semantics (REVISED for the thesis run)

**INFERENCE (from §2):**
- Optimizer steps/epoch ≈ **44,151** (not 6,250).
- Chunks/epoch ≈ **353,210** = one random chunk per training file = **~100% of
  files per epoch** (not 200,000 / 0.7%).
- Over the 14 epochs of the checkpoint: **each song seen ~14 times** (a different
  random window each epoch), i.e. **14 full passes**, not <1.
- LR schedule: linear warm-up over **1,500** steps, then cosine decay over the
  full horizon **2,207,550** steps. Best val at epoch 14 (~618k steps in).

So "an epoch = a full pass over the corpus" for the thesis run (because
`limit_train_batches = 1.0`), **not** a fixed 200k-chunk segment.

## 4. Other hparams (FACT, both checkpoints)

```
d_model 512 · nhead 8 · num_encoder_layers 12 · dim_feedforward 2048 · dropout 0.1
max_seq_length 8192 · mask_prob 0.15 · bar_level_masking True · phase 1
class_weights None · text_encoder_config None
special_token_ids {pad:0, bos:1, eos:2, mask:3}
vocab_sizes {pitch:151, velocity:32, duration:64, program:129, position:96,
             bar:1, tempo:32, timesig:9, rest:32}   → sum + 4 = 550 ✓
```

## 5. config.yaml has DRIFTED from the thesis checkpoint

The current `conf/config.yaml` disagrees with the checkpoint on **five** points:

| Parameter | Thesis checkpoint (authoritative) | current config.yaml |
|---|---|---|
| lr | **3e-4** (`lr=0.0003`) | 1.2e-3 |
| accumulate_grad_batches | **1** (inferred) | 4 |
| limit_train_batches | **1.0** (full) | 25000 |
| chunking strategy | **one chunk/file** (token-like) | bar_based |
| tempo / timesig / rest lambda | **1.0 / 0.5 / 1.0** | 0.0 / 0.0 / 0.5 |
| warmup_steps | 1500 | 1500 (matches) |

**Take-away for the thesis:** cite the **checkpoint** values, not `config.yaml`.
`config.yaml` appears to correspond to a **later** experiment (the log shows other
runs with batch 16, accumulate 4, limit 25000 → `num_training_steps 312,500`,
and `phase1_pretraining` runs with `220,760`).

## 6. Phase-1b (infilling) specifics

**FACT:** the infilling checkpoint hparams are **identical** to Phase-1's
(`num_training_steps 2,207,550`, `lr 3e-4`, same lambda). This is because
`train_infilling.py` uses `load_from_checkpoint`, so `save_hyperparameters()`
captured Phase-1's `__init__` values.

> **Caveat:** the infilling schedule actually used at runtime is set by
> `model.num_training_steps = ...` overridden **after** load
> (`train_infilling.py:149`); that runtime value is **not** re-saved into the
> ckpt hparams. So the infilling ckpt's `num_training_steps 2,207,550` is the
> **inherited Phase-1 value**, not the infilling horizon. The real Phase-1b
> horizon = `(limit_train_batches // accumulate) × epochs` from
> `config_infilling.yaml` (epochs 10) — confirm from its log
> (`logs/infilling_finetuning/version_*/hparams.yaml` shows `2,207,550` inherited;
> `version_8` shows `312,500`). lr for the infilling run = the overridden 3e-5
> (`config_infilling.yaml:14`), not the 3e-4 in hparams.

## 7. Open items to confirm (one log grep each)

- **OPEN:** find the Phase-1 job log that printed
  `total_training_steps: 2207550 (B batches // A accumulate = S steps/epoch x E epochs)`
  to read batch/accumulate/limit/epochs **verbatim** (confirms §2's inference).
  ```bash
  grep -rl "2207550" ~/MIDIT/logs/out/ ; grep -h "total_training_steps" ~/MIDIT/logs/out/*.txt | grep 2207550
  ```
- **OPEN:** identify **which** `conf/*.yaml` was the thesis Phase-1 config (13 files
  exist: config.yaml, config_BACKUP.yaml, config_fixed_OLD.yaml, config_original.yaml,
  phase1_optimized.yaml, phase1_test_lambda.yaml, …). Grep for the matching lambda +
  lr + limit:
  ```bash
  grep -rEl "tempo:\s*1\.0" ~/MIDIT/conf/ ; grep -rE "lr:|limit_train_batches:|accumulate_grad_batches:" ~/MIDIT/conf/*.yaml
  ```

## 8. Phase 2 question

**FACT:** both audited checkpoints have `phase = 1`. Phase 2 (text-conditioned,
FLAN-T5 + MidiCaps) is **deferred / out of thesis scope** — there is **no trained
Phase-2 checkpoint** (`text_encoder_config = None`, `phase = 1` in the hparams).
The two thesis checkpoints are **Phase 1** (pre-training) and **Phase 1b**
(infilling fine-tuning) — both non-text. If you ever have a Phase-2 `.ckpt`, the
same `srun`/`torch.load` snippet audits any checkpoint; just point it at the file.

---

## CONCLUSION FOR THE THESIS (corrected, drop-in)

> MIDIT Phase 1 was pre-trained with AdamW (learning rate 3×10⁻⁴, weight decay
> 0.01), a physical batch size of 8 with no gradient accumulation (**effective
> batch size 8**), and a linear warm-up of 1,500 steps followed by cosine decay
> over a horizon of 2,207,550 optimizer steps. Each epoch is a full pass over the
> ~353k training files (one 8-bar excerpt per file), i.e. ≈44,151 optimizer steps
> per epoch across 50 planned epochs; the reported checkpoint is the best-validation
> model at epoch 14. The multi-head loss uses per-type weights λ = (Pitch 3.0,
> Velocity 1.5, Duration 1.0, Program 1.5, Position 0.5, Bar 0.0, Tempo 1.0,
> TimeSig 0.5, Rest 1.0), as in Table 4.1. Phase 1b (infilling) fine-tunes this
> checkpoint with bar-fill masking, inheriting the same architecture and loss
> weights.

**All numbers above are the authoritative checkpoint values; `config.yaml` has
drifted and should not be cited.** The two OPEN items (§7) confirm the exact
batch/accumulate/limit verbatim if you want zero inference in the Methods section.

---

## Correction note (2026-10-15) — INFERRED, pending W&B confirmation

The "effective batch 8 / ~44,151 optimizer steps per epoch" conclusion in §2–§3
above is **superseded**. The epoch-14 checkpoint (HF release) stores
`global_step = 2,516,273` (Lightning counts optimizer steps), against a planned
`num_training_steps = 2,207,550`. This is consistent with **effective batch 2**:
batch_size 2 with gradient accumulation not passed to the Trainer (commit `be61f54`,
fixed in `b060a7c`), i.e. ≈14.25 epochs × 176,605 batches. The cosine schedule,
which had no clamp at the time (added in `5de167b`), therefore passed its planned
horizon around **epoch 12.5**. Status: INFERRED from checkpoint metadata and commit
history; confirm from the W&B run or the Phase-1 job log.
