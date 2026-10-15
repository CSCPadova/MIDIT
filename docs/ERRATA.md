# Errata — thesis text

Corrections to the printed MSc thesis *MIDIT: Multi-Track Symbolic Music Infilling
via BERT-Style Pre-Training and Distribution-Aware Fine-Tuning* (University of
Padova, A.Y. 2025/2026).

**All reported results are computed on the released checkpoints
(`midit-phase1-epoch14.ckpt`, `midit-infilling-epoch05.ckpt`) and are unaffected
by these corrections.** The errata concern only how the training configuration is
described.

## Loss weights

- **Table 4.1 (p. 44).** The per-type loss weights for Tempo, TimeSig and Rest
  were **1.0 / 0.5 / 1.0** in the released checkpoints (Phase 1 epoch 14 and
  Phase 1b epoch 5), not 0.0 / 0.0 / 0.5 as printed. The printed values match a
  later version of `conf/config.yaml` (commit `631d40b`, 2026-05-20). All other
  weights in the table are correct: Pitch 3.0, Velocity 1.5, Duration 1.0,
  Program 1.5, Position 0.5, Bar 0.0.
- **pp. 62–63.** The sentences that attribute the low Tempo accuracy to
  λ_Tempo = 0, and that state λ_TimeSig = 0, follow from the printed Table 4.1 and
  do not hold: both weights were non-zero during training (1.0 and 0.5).

The values above are the `lambda_weights` stored in the `hyper_parameters` of both
released checkpoints.

## Pending verification

- **Hardware:** pending verification.
- **Training schedule:** pending verification.
