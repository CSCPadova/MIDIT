# Checkpoint Mapping - MIDIT Phase 1

## Naming Convention

Format: `{experiment-name}_epoch{N}_acc{XX.XX}.ckpt`

## Current Checkpoints

### On Cluster (`logs/checkpoints_phase1/`)

| Original Name | New Name (Logical) | Epoch | Val Loss | Avg Acc | Status |
|--------------|-------------------|-------|----------|---------|--------|
| phase1-epoch=44-val_loss=0.0982.ckpt | baseline_epoch44_acc22.58.ckpt | 44 | 0.0982 | 22.58% | ⚠️ Baseline (lambda hardcoded) |
| phase1-epoch=47-val_loss=0.4393.ckpt | lambda-fix_epoch47_acc24.93.ckpt | 47 | 0.4393 | 24.93% | ✅ Best current (lambda fixed) |

## Commands to Rename (Execute on Cluster)
```bash
# Backup originals first
cd ~/MIDIT/logs/checkpoints_phase1/
mkdir -p archive_originals

# Copy originals to archive
cp phase1-epoch=44-val_loss=0.0982.ckpt archive_originals/
cp phase1-epoch=47-val_loss=0.4393.ckpt archive_originals/

# Create symbolic links with logical names
ln -s phase1-epoch=44-val_loss=0.0982.ckpt baseline_epoch44_acc22.58.ckpt
ln -s phase1-epoch=47-val_loss=0.4393.ckpt lambda-fix_epoch47_acc24.93.ckpt

# List to verify
ls -lh *.ckpt

## Future Checkpoints

When you train new models, name them:
- `token-mask_epoch{N}_acc{XX.XX}.ckpt` - Token-level masking experiments
- `class-weights_epoch{N}_acc{XX.XX}.ckpt` - Class weights experiments
- `hybrid_epoch{N}_acc{XX.XX}.ckpt` - Hybrid approaches