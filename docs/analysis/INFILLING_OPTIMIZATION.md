# Phase 1b Infilling Fine-tuning — Optimization Analysis

**Scope:** stabilize and (marginally) improve the BarFillMasking infilling fine-tuning of MIDIT.
**Status:** analysis + proposals. Implementation: only **Proposal A + deterministic val masking** was applied (see §6). Proposals B and C are documented but NOT implemented.
**Author:** generated from a full audit of `train_infilling.py`, `lakh_dataset.py`, `masking.py`, `config_infilling.yaml`, `config.yaml`.

> **Post-thesis experiment.** Proposal A is not a thesis model. Its launcher
> `slurm/train_infilling_v2.sh` is no longer in the tree; it is preserved at the
> `archive-full-tree` tag.

---

## 0. Honest expectations (read first)

- The **~8.5 val_loss floor is mostly irreducible.** In infilling, the loss is computed *only* on masked content tokens (Pitch/PitchDrum/Velocity/Duration/Program), and it is dominated by Pitch (λ=3.0, highest entropy, ~151 classes). Pitch is musically *underdetermined* given the preserved scaffold — many pitches are equally valid — so its cross-entropy cannot be driven near zero. **A 10% val_loss reduction (8.5 → 7.7) is optimistic; expect 0–5% from the configs below.** The honest way to make the *number* smaller is lambda rescaling, which is arithmetic, not a better model (flagged explicitly below).
- **Stability (Goal 1) is the achievable win.** The oscillation is well-explained and directly addressable with larger effective batch + deterministic val masking + smoothing. Expect clearly cleaner curves.
- Downstream musical metrics (CP, PCHE, JS) are already good and are **loosely coupled** to CE loss; do not expect these proposals to move them much, in either direction.

---

## 1. Audit — current effective configuration

What actually runs today (after `config_infilling.yaml` overrides `config.yaml`, and after `load_from_checkpoint` restores baked-in hparams):

| Parameter | Value | Source | Note |
|---|---|---|---|
| Masking strategy | `bar_fill`, `max_infill_bars=3`, `mask_strategy=always` | `config_infilling.yaml:21-25` | 100% MASK on note tokens only |
| Masked token types | Pitch, PitchDrum, Velocity, Duration, Program | `masking.py:411` | structural tokens preserved |
| `lambda_weights` | pitch 3.0, vel 1.5, dur 1.0, prog 1.5, pos 0.5, rest 0.5, bar/tempo/timesig 0.0 | **checkpoint hparams**, *not* config | see Finding A |
| Effective loss heads | pitch, velocity, duration, program | derived | pos/rest λ>0 but never masked → contribute 0 |
| `batch_size` | 8 | `config.yaml:100` (not overridden) | |
| `accumulate_grad_batches` | 4 | `config.yaml:109` | **effective batch = 32** |
| `lr` | 3e-5 | `config_infilling.yaml:13` | applied post-load |
| `warmup_steps` | 200 | `config_infilling.yaml:14` | applied post-load |
| LR schedule | linear warmup → cosine to 0 | `masked_transformer.py:551-557` | |
| `gradient_clip_val` | 1.0 | `config.yaml:107` | |
| `precision` | bf16 | `config.yaml:108` | |
| `epochs` | 10 | `config_infilling.yaml:15` | |
| `limit_train_batches` | 50000 | `config_infilling.yaml:16` | ~12.5k optimizer steps/epoch |
| `limit_val_batches` | 500 | `config_infilling.yaml:17` | |
| `val_check_interval` | 1.0 | `config_infilling.yaml:18` | once per epoch |
| Val masking | `BarFillMasking`, **seed=None** | `lakh_dataset.py:224-230` | **non-deterministic** — see Finding B |

---

## 2. Diagnosis — root causes, ranked

### Cause #1 (primary) — per-batch variance of the masked content, amplified by small effective batch

`BarFillMasking._select_bars` picks `num_infill = randint(1, min(3, num_bars))` and a random start bar (`masking.py:496-507`). Consequences:

- The **number** of masked note tokens per sample swings wildly (a dense bar ≈ 200 tokens vs a sparse bar ≈ 5).
- The **predictability** of the masked region swings too (repeated single-program riff vs polyphonic multi-instrument passage).
- The total loss is, by construction:
  `L ≈ 3·CE_pitch + 1.5·CE_vel + 1·CE_dur + 1.5·CE_prog`
  Worst-case (max entropy) ≈ `3·ln151 + 1.5·ln32 + 1·ln64 + 1.5·ln129 ≈ 32`. Highly-predictable batches drop toward ~0. **The observed 0–25 range is exactly this content-predictability swing, not a bug.**
- With **effective batch = 32**, each step averages too few samples to wash out that swing. This is the single biggest, cheapest lever.

### Cause #2 — the loss has no "ballast" of easy tokens

In pre-training (token-level random masking) trivial structural tokens (Position, Bar, TimeSig, near-0 CE) are also masked and pull the average down to ~2.4. In infilling **only the four hardest heads ever enter the loss**, so there is nothing to stabilize/lower the running mean. This is why the number is both *high* and *flat*, and why it is not directly comparable to the Phase 1 val_loss.

### Cause #3 — non-deterministic validation masking (measurement noise)

The val masker has `seed=None` (`lakh_dataset.py:224-230`). With `persistent_workers=True` the per-worker RNG advances, so **the validation set is masked differently every epoch and every run.** Part of the epoch-to-epoch val_loss wobble — and part of why 6 runs land in a *range* rather than a point — is the evaluation task itself moving. Fixing this does not lower the floor but makes the val curve smoother and runs comparable.

### Cause #4 — degenerate / near-empty masked samples spike toward 0

When a chunk has no maskable note tokens, `BarFillMasking` returns an unmasked clone with an all-False `mask_labels` (`masking.py:547-549`). In `compute_loss`, `valid_mask.sum()` is 0 → `loss_mean = (loss*mask).sum() / clamp(min=1) = 0` (`multi_head_output.py:200-204`). Such samples (and very sparse ones) contribute ~0, deepening the downward spikes of the 0–25 oscillation.

### Cause #5 (minor) — short warmup on a converged checkpoint

200 warmup steps then an immediate jump to peak lr=3e-5 on already-converged weights can add early-epoch noise. Not the dominant cause (instability persists mid-training), but cheap to soften.

### Cause #6 (cosmetic) — per-micro-batch logging, no smoothing

`self.log("train_loss", ..., prog_bar=True)` logs raw per-step loss (`masked_transformer.py:480`). Much of the *visual* chaos is just unsmoothed logging granularity. The W&B smoothing slider already fixes figures for free; an EMA log is optional.

---

## 3. Key audit findings (gotchas)

- **Finding A — `lambda_weights` overrides in config are inert during fine-tuning.** `load_from_checkpoint` restores the saved hparams; only `lr`/`warmup_steps`/`num_training_steps` are overwritten afterward (`train_infilling.py:142-149`). **Any λ change requires a one-line code patch**, not a config edit. (Relevant to Proposal C, not implemented.)
- **Finding B — validation masking is non-deterministic** (Cause #3). **Fixed** as part of this work (see §6).
- **Finding C — `masking.mask_prob: 0.15` in `config_infilling.yaml` is inert** for `bar_fill` (only the `TokenLevelMasking` fallback uses it). Harmless, but don't expect tuning it to do anything.
- **Finding D — `batch_size` is not overridden in `config_infilling.yaml`,** so it inherits 8 from base. Easy to forget.
- **Finding E — `paths.checkpoints_infilling` is NOT read; the checkpoint subdir name is hardcoded.** `train_infilling.py:165` does `checkpoints_dir = os.path.join(logs_dir, "checkpoints_infilling")`, where `logs_dir = cfg.logging.logs_dir or cfg.data.logs_dir or 'logs'`. The original runs landed on `/nfsd/.../logs/checkpoints_infilling/` because `~/MIDIT/logs` is a symlink to `/nfsd/tesi/zanettisim/MIDIT/logs` (confirm with `ls -la ~/MIDIT/logs`). Consequence: a `paths.checkpoints_infilling: ..._v2/` override would be **inert**, and v2 would write into the *same* `checkpoints_infilling/` dir as the original → collision (`save_last=True` overwrites `last.ckpt`). **Fixed** by setting an absolute `logging.logs_dir: /nfsd/.../logs_v2` in `config_infilling_v2.yaml` (see §6/§7).

---

## 4. Proposals — comparison

| | **A — Stability (config-only)** | **B — Curriculum (code)** | **C — Rebalance (code)** |
|---|---|---|---|
| Effective batch | 8 × **16** = 128 | 8 × **8** = 64 | 8 × 8 = 64 |
| `max_infill_bars` | 3 (fixed) | **1 → 2 → 3** ramped by epoch | 3 |
| `lr` / `warmup` | 3e-5 / **1000** | 3e-5 / 1000 | **2e-5** / 1000 |
| `gradient_clip_val` | **0.5** | 0.5 | 0.5 |
| λ change | none | none | **pitch 3.0→2.0, prog 1.5→1.0** |
| Deterministic val mask | **yes** | yes | yes |
| Code change needed? | **No** (+ 1-line val seed) | Yes (callback + val seed) | Yes (λ override + val seed) |
| Targets | Goal 1 (strong), Goal 2 (weak) | Goal 1 (medium), Goal 2 (small) | Goal 2-number (arithmetic), Goal 1 (medium) |
| Expected val_loss Δ | ~0 to −3% (real) | ~0 to −5% (real) | −10–20% **on the number** (mostly rescaling) |
| Risk | Low | Medium (complexity) | Medium (breaks comparability) |
| Decision | **IMPLEMENTED** | Not implemented | Not implemented |

---

## 5. Proposal details

### Proposal A — Stability (IMPLEMENTED, config-only)

**Changes** (new file `conf/config_infilling_v2.yaml`):
- `train.accumulate_grad_batches: 16` (effective batch 32 → 128).
- `train.warmup_steps: 1000`.
- `train.gradient_clip_val: 0.5`.
- (keep `lr: 3e-5`, `batch_size: 8`, `epochs: 10`.)

**Why:** averaging 128 samples/step instead of 32 directly attacks Cause #1 — the dominant variance source. Each optimizer step sees a far more representative mix of masked content, so both the gradient and the logged loss are smoother. Longer warmup + tighter clip soften Causes #5/#1 spikes.

**Expected impact:** clearly smoother train curve (Goal 1 ✓). Val_loss: small real improvement (0–3%) from steadier optimization. **Honest:** unlikely to break the 8.5 floor meaningfully.

**Risk:** Low. 4× fewer optimizer steps/epoch for the same `limit_train_batches`; if you want the same number of *updates*, raise `limit_train_batches` or `epochs` (watch the 48h budget). bf16 + clip 0.5 is safe.

**Note on lr:** with 4× larger effective batch, linear scaling would *raise* lr. We intentionally keep 3e-5 for stability. If A converges cleanly but slowly, a follow-up with `lr: 4e-5` is reasonable.

### Proposal B — Curriculum on `max_infill_bars` (NOT implemented)

Start with `max_infill_bars=1`, ramp 1→2→3 across epochs via a Lightning callback that mutates the live masker (keep val fixed at 3). Reduces early variance and may buy a small real val_loss improvement. Medium complexity: the `DataLoaderWrapper` (`train_infilling.py:24-47`) hides the dataset, so the callback must capture `train_dataset` by closure. Documented for a possible future run.

### Proposal C — Per-head rebalance (NOT implemented)

`pitch 3.0 → 2.0`, `program 1.5 → 1.0`. Requires a code override after `load_from_checkpoint` (Finding A) because YAML λ is inert. Lowers the *number* 10–20% but this is **arithmetic rescaling, not a better model**, and makes val_loss non-comparable to the existing runs. Reporting-only option.

---

## 6. What was implemented in this change

1. **`conf/config_infilling_v2.yaml`** — Proposal A (new file; original `config_infilling.yaml` untouched).
2. **Deterministic validation masking** in `midit/data/lakh_dataset.py` — `BarFillMasking` now receives `seed=(1234 if split == 'val' else None)`. Train stays stochastic; only validation is pinned, removing Cause #3.
3. **`slurm/train_infilling_v2.sh`** — SLURM launcher for Proposal A (L40S, 48h).
4. **Output isolation (Finding E)** — `config_infilling_v2.yaml` sets `logging.logs_dir: /nfsd/tesi/zanettisim/MIDIT/logs_v2` so v2 checkpoints/tb/wandb go to `/nfsd/.../logs_v2/` (absolute, on NFSD), with **no collision** with the original `/nfsd/.../logs/checkpoints_infilling/` run. A `paths.checkpoints_infilling` key would have been inert.

Nothing else in the existing pipeline was modified (the `lakh_dataset.py` val-seed patch is backward-compatible: train behavior unchanged).

**Figures:** use the W&B smoothing slider (0.9–0.99) for thesis plots — no code needed.

---

## 7. Config file (Proposal A) — `conf/config_infilling_v2.yaml`

```yaml
defaults:
  - config
  - _self_

train:
  lr: 3e-5
  warmup_steps: 1000
  epochs: 10
  accumulate_grad_batches: 16     # effective batch 8*16 = 128
  gradient_clip_val: 0.5
  limit_train_batches: 50000
  limit_val_batches: 500
  val_check_interval: 1.0
  checkpoint_path: /nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_phase1/phase1-epoch=49-val_loss=2.3866.ckpt

masking:
  strategy: bar_fill
  max_infill_bars: 3
  mask_strategy: always
  mask_prob: 0.15

# Absolute NFSD logs_dir → isolates v2 outputs and avoids the ~/ NAS (Finding E).
logging:
  logs_dir: /nfsd/tesi/zanettisim/MIDIT/logs_v2
```

> Reminder: do **not** put `lambda_weights` here expecting an effect (Finding A), and do **not** rely on `paths.checkpoints_infilling` (Finding E) — neither is read.

---

## 8. SLURM template — `slurm/train_infilling_v2.sh`

```bash
#!/bin/bash
#SBATCH --job-name=midit_infill_v2a
#SBATCH --output=logs/slurm_infill_v2a_%j.out
#SBATCH --error=logs/slurm_infill_v2a_%j.err
#SBATCH --partition=allgroups
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:l40s:1

cd /home/zanettisim/MIDIT
source .venv/bin/activate
export PYTHONPATH=/home/zanettisim/MIDIT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python midit/train_infilling.py \
    --config-name=config_infilling_v2 \
    logging.use_wandb=true \
    chunking.preprocessed_dir=/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k
```

> `--config-name` overrides the `config_name="config_infilling"` baked into `@hydra.main` in `train_infilling.py`.

---

## 9. What NOT to expect

- A dramatic val_loss drop. The floor is content-entropy dominated; 0–5% real improvement is the realistic band.
- Better musical metrics as a guarantee — CE and JS/CP/PCHE are loosely coupled; they may not move.
- Any effect from `mask_prob` under `bar_fill` (Finding C), or from `lambda_weights` set in YAML (Finding A).

**Bottom line:** the confident win is *stability and figure quality* (Proposal A + deterministic val masking). Treat val_loss reduction as a bonus.
