#Resaults 16/12/2025 primi test
# MIDIT Phase 1 - Training Results Log

## Baseline
| Checkpoint | Epoch | Pitch | Velocity | Duration | Avg | Notes |
|-----------|-------|-------|----------|----------|-----|-------|
| epoch=44 | 44 | 7.34% | 2.81% | 36.94% | 22.58% | Pre-fix |
| epoch=47 | 47 | 13.02% | 1.36% | 49.69% | 24.93% | Lambda fix |

## Baseline (Prima dei Fix)

| Date | Checkpoint | Epoch | Config | Val Loss | Pitch Acc | Velocity Acc | Duration Acc | Avg Acc | Notes |
|------|-----------|-------|--------|----------|-----------|--------------|--------------|---------|-------|
| 2025-12-04 | epoch=44 | 44 | lambda_old, bar_mask=0.15 | 0.0982 | 7.34% | 2.81% | 36.94% | 22.58% | Baseline pre-fix |

## Fix Iteration 1: Lambda Weights Corretti

| Date | Checkpoint | Epoch | Config | Val Loss | Pitch Acc | Velocity Acc | Duration Acc | Avg Acc | Notes |
|------|-----------|-------|--------|----------|-----------|--------------|--------------|---------|-------|
| 2025-12-13 | epoch=47 | 47 | lambda_fixed, bar_mask=0.4 | 0.4393 | **13.02%** | 1.36% | **49.69%** | **24.93%** | Lambda fix works! Pitch +77% |

## Fix Iteration 2: Token-Level Masking (TBD)

| Date | Checkpoint | Epoch | Config | Val Loss | Pitch Acc | Velocity Acc | Duration Acc | Avg Acc | Notes |
|------|-----------|-------|--------|----------|-----------|--------------|--------------|---------|-------|
| TBD | TBD | TBD | token_mask=0.15 | TBD | TBD | TBD | TBD | TBD | Testing token-level |

---

## Configuration Details

### Baseline (epoch 44)
```yaml
lambda_weights:
  pitch: 1.0
  position: 4.5  # Too high!
  bar: 10.0      # Too high!
  
masking:
  bar_mask_prob: 0.15
  type_mask_ratio: 0.3
```

### Fix 1 (epoch 47)
```yaml
lambda_weights:
  pitch: 1.0
  velocity: 0.5
  position: 0.2  # Fixed!
  bar: 0.1       # Fixed!
  
masking:
  bar_mask_prob: 0.4
  type_mask_ratio: 0.8
```

---

## Key Insights

### What Works:
- ✅ Lambda weights correction: pitch +77% improvement
- ✅ Duration improved significantly (36% → 49%)
- ✅ Loss stable (no collapse)

### What Doesn't:
- ❌ Velocity/program worse (mode collapse?)
- ❌ Bar-level masking instability (0-54% variance)
- ❌ Overall accuracy still low (<25%)

### Next Steps:
1. Try token-level masking (15% fixed)
2. Increase training epochs (60-80)
3. Add class weights for velocity/program