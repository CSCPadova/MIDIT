# Configuration, drift and checkpoints

This directory holds the [Hydra](https://hydra.cc/) configuration for MIDIT. The
YAML files are kept **unchanged** as records of the experiments that used them,
which means the current defaults do **not** describe the thesis runs. This page
documents the differences; the checkpoints' stored hyperparameters are the
authoritative record of what was trained.

## Files

| File | Role |
|---|---|
| `config.yaml` | Base config. Default for `midit/train_phase1.py` (Phase 1 pre-training). |
| `config_infilling.yaml` | Phase 1b fine-tuning. Inherits `config.yaml`; default for `midit/train_infilling.py`. |
| `config_infilling_v2.yaml` | Post-thesis Phase 1b variant ("Proposal A", stability: larger effective batch, longer warm-up). Select with `--config-name=config_infilling_v2`. Writes logs to an absolute cluster path. |
| `configs_archive/` | Phase-1 configs from December 2025 (pre-V2 tokenizer). Historical only — see its README. |
| `hydra/` | Launcher and callback configs from the project template; not referenced by any `defaults` list. |
| `config.py` | Dataclass schema from the project template; not imported anywhere and out of sync with the YAML. |

The `tokenizer:` block in `config.yaml` is documentation only: the tokenizer is
hard-coded in `midit/data/lakh_dataset.py` (V2, vocabulary 550).

## Checkpoints

| Checkpoint (cluster file name) | Released as | Status |
|---|---|---|
| `phase1-epoch=14-val_loss=2.7254.ckpt` | `midit-phase1-epoch14.ckpt` | **Thesis** Phase-1 model (zero-shot base) |
| `infilling-epoch=05-val_loss=8.5836.ckpt` | `midit-infilling-epoch05.ckpt` | **Thesis** Phase-1b model, fine-tuned from epoch 14 |
| `phase1-epoch=49-val_loss=2.3866.ckpt` | — | Later epoch of the same Phase-1 run. Used by the current `config_infilling*.yaml` and by `slurm/generate_samples_zs.sh`. **Not a thesis model.** |
| `phase1-epoch=44-val_loss=0.0982.ckpt` | — | **Invalid** (December 2025 val_loss≈0 bug). Still referenced by `research/debug/debug_per_class_accuracy.py` and `research/Tokenization/verify_token_head_mapping.py`; do not trust their output. |

Evidence that the thesis Phase-1b model was fine-tuned from epoch 14:

- `config_infilling.yaml` pointed at the epoch-14 checkpoint from its creation
  (commit `12eb476`, 2026-04-14) until commit `f313df4` (2026-06-01), which
  changed only that line to epoch 49.
- The thesis infilling checkpoint dates from 2026-05-28, before that change.
- `slurm/train_infilling.sh` passes no checkpoint override, so the config value was used.

## Configuration drift

Thesis values are read from the `hyper_parameters` stored in the released
checkpoints (the Phase-1b checkpoint inherits them from Phase 1 through
`load_from_checkpoint`):

| Setting | Thesis (checkpoint) | Current `config.yaml` |
|---|---|---|
| `train.lr` | 3e-4 | 1.2e-3 |
| `lambda_weights.tempo` | 1.0 | 0.0 |
| `lambda_weights.timesig` | 0.5 | 0.0 |
| `lambda_weights.rest` | 1.0 | 0.5 |
| other `lambda_weights` | pitch 3.0 · velocity 1.5 · duration 1.0 · program 1.5 · position 0.5 · bar 0.0 | same |
| `train.warmup_steps`, `train.weight_decay` | 1500, 0.01 | same |
| model size, `masking.mask_prob` | 512 / 8 heads / 12 layers / FF 2048, 0.15 | same |

Notes:

- **Batch size and gradient accumulation are not stored in checkpoints.** The
  Phase-1 checkpoint stores the *planned* scheduler horizon
  `num_training_steps = 2,207,550`; the batch and schedule configuration of the
  thesis run is pending verification and not documented here.
- **`bar_level_masking=True`** in the stored hyperparameters is an unused
  constructor argument (it was hard-coded before commit `d666a1f`). Training
  masking is applied by the dataset; `config.yaml` has selected token-level
  masking (`masking.bar_level: false`) since commit `98cbace` (2026-02-26).
- In Phase 1b, the loss weights come from the loaded checkpoint, so
  `lambda_weights` in the config have no effect there.
- Phase 1b settings at the time of the thesis run (`config_infilling.yaml` before
  `f313df4`): lr 3e-5, 200 warm-up steps, BarFillMasking with up to 3 bars.

To run Phase 1 with the thesis optimizer and loss settings, override the drifted
values explicitly:

```bash
uv run python midit/train_phase1.py \
    chunking.preprocessed_dir=data/preprocessed_v2_50k \
    train.lr=3e-4 \
    lambda_weights.tempo=1.0 lambda_weights.timesig=0.5 lambda_weights.rest=1.0
```

## Cluster paths

`config_infilling.yaml` and `config_infilling_v2.yaml` contain absolute paths on
the original cluster (`/nfsd/...`). Override them on the command line
(`train.checkpoint_path=...`, `logging.logs_dir=...`) rather than editing the files.
