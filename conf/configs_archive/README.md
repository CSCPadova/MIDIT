# Config Archive - MIDIT Phase 1

This directory stores historical configurations for reproducibility.

## Files

| Config | Date | Checkpoint | Accuracy | Notes |
|--------|------|-----------|----------|-------|
| `config_baseline_epoch44.yaml` | 2025-12-11 | epoch=44 | 22.58% | ⚠️ Lambda hardcoded bug |
| `config_lambda-fix_2025-12-17.yaml` | 2025-12-13 | epoch=47 | 24.93% | ✅ Lambda fix applied |

## Usage

To reproduce a specific experiment:
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