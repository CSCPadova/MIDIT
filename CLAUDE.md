# CLAUDE.md — MIDIT Ground Truth

**Purpose of this file:** prevent a future Claude instance from reintroducing
known errors. It is ground truth, **not** a project manual and **not** a task
tracker. For history, run `git log`. If a fact here conflicts with something you
recall or with an old doc, this file wins — but verify against the repo before
acting, and never propagate a claim you cannot cite.

Last audited: **2026-08-10** (branch `audit-fixes`).

---

## People & Scope

- **Candidate:** Simone Zanetti — MSc Data Science, University of Padova (DEI)
- **Supervisor:** Prof. Antonio Rodà · **Co-supervisor:** Matteo Spanio (PhD)
- **Thesis:** *"MIDIT: Bidirectional Masked Language Modeling for Symbolic Music
  Infilling"* — submission **2026-09-10**, defense **2026-09-25**
- **Working branch:** `audit-fixes` (contains `masked-transformer`, already merged)
- **What MIDIT is:** a BERT-style Masked Language Model over REMI+ MIDI tokens.
  Phase 1 = unconditional MLM pre-training (Lakh); Phase 1b = infilling fine-tuning.

---

## Canonical Checkpoints — DO NOT CONFUSE

| Role | File | Notes |
|---|---|---|
| **Phase 1 — thesis, zero-shot** | `phase1-epoch=14-val_loss=2.7254.ckpt` | V2, vocab 550. **The only correct Phase-1 checkpoint to cite.** Published to HuggingFace as `midit-phase1-epoch14.ckpt`. |
| **Phase 1b — thesis, fine-tuned** | `infilling-epoch=05-val_loss=8.5836.ckpt` | Created 2026-05-28, fine-tuned **FROM epoch=14 above**. |

**Forensic confirmation (audit 2026-08-10):** the infilling checkpoint was fine-tuned
from `epoch=14`, established by three converging tests — file timestamp (2026-05-28,
*before* the config was switched to epoch=49 on 2026-06-01), launcher (`train_infilling.sh`
passes no `checkpoint_path` / `--config-name` override, so the base is read verbatim
from `config_infilling.yaml`, which pointed at epoch=14 on that date), and checkpoint
metadata (same-lineage base hparams). The thesis provenance is sound.

### ⚠️ KNOWN DIVERGENCE — not the thesis base
`conf/config_infilling.yaml` and `conf/config_infilling_v2.yaml` currently point at
`phase1-epoch=49-val_loss=2.3866.ckpt`. This is a **POST-THESIS experiment**
(introduced 2026-06-01, after the thesis checkpoint already existed on 2026-05-28;
v2 config added 2026-06-10). It is a *later epoch of the same pre-training run*, adopted
for a subsequent Phase-1b re-run attempt. **It is NOT the thesis base.** Do not treat it
as ground truth, and do **not** "fix" the thesis numbers to match it.

### ❌ DEAD — do not use or cite
- `phase1-epoch=48-val_loss=2.3922.ckpt` — V1, vocab 518, stale ~49%-accuracy regime.
  Superseded by the V2 run. (Not present on this disk; referenced only by the old CLAUDE.md.)
- `phase1-epoch=44-val_loss=0.0982.ckpt` (local, in `logs/checkpoints_phase1/`) — residue
  of the December val_loss≈0 bug. Invalid; ignore or delete. Some debug scripts still
  reference it — do not trust their output.

---

## Verified Thesis Numbers — cite these, nothing else

**Cloze accuracy (epoch=14; Bar/TimeSig excluded as trivially predictable):**
- Avg Top-1 **83.18%** · Avg Top-5 **94.00%** · Pitch Top-1 **84.29%**

**50-song infilling eval (3 masked bars), Zero-Shot → Fine-Tuned:**
- CP **0.297 → 0.681** · GS **0.861 → 0.966** · PCHE **0.487 → 0.320**

**SOTA comparison vs Anticipatory MT** — 878 matched samples, time-aligned crop
(corrected 2026-08-09, see bug note below):
- **CP:** MIDIT 0.728 ± 0.19 vs Anticipatory 0.649 ± 0.23 — p < 0.001, dz = 0.33,
  MIDIT wins 61% of samples (**moderate advantage, not dominance**)
- **PCHE:** 0.323 ± 0.31 vs 0.312 ± 0.41 — t-test p = 0.47 (n.s.); Wilcoxon p = 0.001,
  effect negligible dz = 0.02 → read as **statistical parity**
- **Density (n_pitches_used):** MIDIT 13.85 ± 6.40 vs Anticipatory 14.84 ± 7.60 —
  comparable. **NOT ~55%** as an earlier (buggy) analysis claimed.
- **S-I:** groove_consistency MIDIT 0.840 vs Antic. 0.911; scale_consistency 0.976 vs 0.969

**Baseline random-in-key (CP):** B1 (uniform) 0.468 · B2 (context-weighted) 0.654 ·
MIDIT FT 0.729 — gap vs B2 is only **+0.075**.

---

## Known-Fixed Bugs — do NOT reintroduce or re-flag as open

- **tpq crop mismatch** (fixed 2026-08-09). Anticipatory MT reconstructions use
  tpq = 50; MIDIT/REMI+ uses tpq = 16. An earlier eval script applied MIDIT's
  tick-based crop window verbatim to Anticipatory files without unit conversion,
  misaligning the evaluated region. Fixed in
  [scripts/audit_fixes/recompute_anticipatory_metrics_fixed.py](scripts/audit_fixes/recompute_anticipatory_metrics_fixed.py)
  with a time-based (seconds-aligned) crop. MIDIT's own scores were unaffected
  (same tick grid on both sides).
  **NEVER cite the superseded numbers:** Anticipatory CP = 0.435, PCHE = 0.790,
  d = 1.16; "892 samples" (now 878).
- Prior Phase-1 bugs (double-masking, eval class-mapping, LR-scheduler freeze, loss
  denominator dilution, mask_labels=None, val_loss=0) are all fixed. See `git log`;
  do not re-open.

---

## Working Principles

- **Never invent results.** When reasoning about results, separate **FACT /
  INFERENCE / HYPOTHESIS** and give `file:line` citations.
- **No re-training, no re-inference** of the 989-sample generation set during the
  remaining thesis work — **analysis only, on existing outputs.**
- **Verification-first:** audit and report before modifying code or claims.
- **Discussion framing (Matteo's directive):** the two paradigms — MIDIT
  (bidirectional, parallel) vs Anticipatory MT (autoregressive) — show **different
  quality profiles, not a ranking.** MIDIT favors contextual fidelity; Anticipatory
  is competitive on distributional diversity. **Never frame as "MIDIT wins."**

---

## Repository Structure (verified 2026-08-10)

```
midit/
├── train_phase1.py            # Phase 1 MLM pre-training
├── train_infilling.py         # Phase 1b infilling fine-tune (hydra config_name=config_infilling)
├── infer.py                   # inference / generation entry point
├── __main__.py
├── model/
│   ├── masked_transformer.py  # main model (+ Phase 2 path)
│   ├── multi_head_output.py   # 9 per-token-type heads
│   ├── model_factory.py
│   └── text_encoder.py        # FLAN-T5 (Phase 2, unused)
├── data/
│   ├── lakh_dataset.py        # dataset + HARDCODED TokenizerConfig (source of truth for vocab)
│   └── masking.py             # masking strategies (token-level, bar-fill)
├── loss/                      # loss modules
└── logs/

scripts/
├── audit_fixes/               # 2026-08 audit + correction scripts (tpq fix, baselines, stats, probes)
├── sota_baselines/anticipatory/   # Anticipatory MT baseline
├── Evaluation_Infilling/      # CP/GS/PCHE + MusPy metrics, merge
├── Evaluation_Accuracy/       # cloze Top-1/Top-5
├── Generation/                # sample generation from checkpoints
├── Pre_Processing/            # preprocess_v2.py (current), validators
├── Tokenization/  Masking/  Tests/  Analyze_dataset/  Smoke_test_training/  debug/  figures/
```

**Tokenizer is hardcoded** in [midit/data/lakh_dataset.py](midit/data/lakh_dataset.py)
(REMI, `pitch_range=(21,109)`, `beat_res={(0,4):8,(4,12):4}`, `num_velocities=32`,
`delete_equal_successive_{tempo,time_sig}_changes=True`) → **vocab 550**. The
`tokenizer:` block in `conf/config.yaml` is **informational only** — not read by code.
Current `lambda_weights` (config.yaml): pitch 3.0, velocity 1.5, duration 1.0, program 1.5,
position 0.5, rest 0.5, **bar/tempo/timesig = 0.0** (zeroed).

---

## Status (as of 2026-08-10)

Thesis is in the **write-up + audit** stage (submission 2026-09-10). All experiments
are frozen: work is **analysis and correction on existing outputs only** — no new
training or inference of the generation set. The audit that produced this file's
verified numbers lives under [scripts/audit_fixes/](scripts/audit_fixes/); remaining
work follows the current revision plan (results-integrity and discussion framing).
Phase 2 (text-conditioned, MidiCaps) remains **deferred** and out of thesis scope.

---

## Correction note (2026-10-15) — Phase-1 batch / schedule — INFERRED, pending W&B confirmation

The claim "effective batch 8 / ~44,151 optimizer steps per epoch" for the thesis
Phase-1 run (from `Modifiche_luglio/checkpoint_audit_findings.md`) is **superseded**.

- **FACT:** the epoch-14 checkpoint (`midit-phase1-epoch14.ckpt`, HF release) stores
  `global_step = 2,516,273` and `num_training_steps = 2,207,550`. Lightning's
  `global_step` counts optimizer steps (checked with a minimal CPU run).
- **FACT:** commit `be61f54` (2026-03-26) divides `total_training_steps` by
  `accumulate_grad_batches` but does not pass it to `L.Trainer` (added in `b060a7c`);
  its `lr_lambda` has no `min(1.0, …)` clamp (added in `5de167b`, whose message
  reports a "cosine restart observed at epoch ~14"). Config then: batch_size 2,
  accumulate 4, lr 3e-4, val_check_interval 0.25.
- **INFERRED:** `global_step` ≈ 14.25 epochs × 176,605 batches → one optimizer step per
  batch of 2, i.e. **effective batch 2** (accumulation not applied), and the cosine
  schedule passed its planned horizon around **epoch 12.5** (LR then rising again).
- Not stated in the public README/ERRATA until confirmed from the W&B run / job log.
- Also verified 2026-10-15: both released checkpoints store λ Tempo/TimeSig/Rest =
  1.0/0.5/1.0; thesis Table 4.1 (0.0/0.0/0.5) is wrong → `docs/ERRATA.md`.
