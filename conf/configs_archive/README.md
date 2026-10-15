# Config Archive - MIDIT Phase 1

This directory stores historical configurations for reproducibility.

> These are **December 2025** Phase-1 configs (before the V2 tokenizer, vocabulary
> 550). They do **not** describe the thesis models. For the thesis checkpoints and
> how the current configs differ from them, see [`../README.md`](../README.md).

## Files

| Config | Date | Checkpoint | Accuracy | Notes |
|--------|------|-----------|----------|-------|
| `config_baseline_epoch44.yaml` | 2025-12-11 | epoch=44 | 22.58% | ⚠️ Lambda hardcoded bug |
| `config_lambda-fix_2025-12-17.yaml` | 2025-12-13 | epoch=47 | 24.93% | ✅ Lambda fix applied |

## Usage

To reproduce a specific experiment (⚠️ the `cp` below **overwrites** the live
`conf/config.yaml` — commit or back it up first):
```bash
# Copy archived config to main config
cp conf/configs_archive/config_lambda-fix_2025-12-17.yaml conf/config.yaml

# Run training
python midit/train_phase1.py
```

## Naming Convention

Format: `config_{experiment-name}_{date}.yaml`

Examples:
- `config_baseline_epoch44.yaml` - Initial baseline
- `config_lambda-fix_2025-12-17.yaml` - Lambda weights fix
- `config_token-mask_2025-12-18.yaml` - Token-level masking (future)