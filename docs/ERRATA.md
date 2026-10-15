# Errata — thesis text

Corrections to the printed MSc thesis *MIDIT: Multi-Track Symbolic Music Infilling
via BERT-Style Pre-Training and Distribution-Aware Fine-Tuning* (University of
Padova, A.Y. 2025/2026).

**All reported results are computed on the released checkpoints
(`midit-phase1-epoch14.ckpt`, `midit-infilling-epoch05.ckpt`).** The loss-weight
errata concern only how the training configuration is described and do not affect
any result. The only reported values that change are the GC column of Table 5.11
and the reference line of Figure 5.5 (seed-set averaging, see below); no
conclusion changes. Figure 2.9, an illustration of the token stream, has incorrect
token labels; no result depends on it.

## Figure 2.9 — REMI+ token-stream example (p. 15)

- **Figure 2.9** (caption: "REMI+ token stream from a real multi-track file in the
  Lakh MIDI Dataset, tokenized with the MIDIT V2 configuration") does not depict a
  V2 stream. It shows a file tokenized with the earlier **V1** configuration
  (518-token vocabulary) whose token ids were decoded with the **V2** vocabulary
  (550 tokens), so 14 of the 28 labels are wrong:

  | Position | Printed | Actual token |
  |---|---|---|
  | 2 | Prog. 105 | TimeSig 4/4 |
  | 3 | Dur. 4.1 | Position 0 |
  | 4 | Rest 2.1 | Tempo 121.29 |
  | 5 | Drum 72 | Rest 4.0 |
  | 6 | Dur. 4.1 | Position 0 |
  | 7, 11, 25 | Prog. 55 | Program 87 |
  | 15, 19 | Prog. 96 | Program −1 (drums) |
  | 16 | Pos. 73 | PitchDrum 36 |
  | 20 | Pos. 79 | PitchDrum 42 |
  | 23 | Drum 57 | Rest 0.1 |
  | 24 | Dur. 5.1 | Position 4 |

  The remaining labels (Bar, Pitch, Velocity and Duration tokens) are correct. The
  figure is illustrative only; no result depends on it.
- **Provenance.** Re-rendering the first 28 tokens of the V1 file
  `lakh_00000_Caught Up In You.pt` with the V2 vocabulary using
  `scripts/preprocessing/visualize_tokenization_v2.py` reproduces the printed
  figure byte for byte; the same ids decoded with the V1 tokenizer give the
  actual tokens above.

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

## Table 5.11 — extreme infilling GC (p. 80) and text (p. 81)

- **Table 5.11 (p. 80).** The caption states seeds 1–5 with seed 10 excluded, and
  the CP column follows it. The GC (groove consistency) column, however, averaged
  seeds 1–5 **and** seed 10. Corrected GC values (seeds 1–5):

  | Configuration | GC printed (seeds 1–5 + 10) | GC corrected (seeds 1–5) |
  |---|---|---|
  | first1 | 0.942 ± 0.044 | 0.932 ± 0.042 |
  | first2 | 0.924 ± 0.046 | 0.912 ± 0.041 |
  | first3 | 0.899 ± 0.034 | 0.896 ± 0.036 |
  | last1 | 0.819 ± 0.043 | 0.826 ± 0.044 |
  | last2 | 0.820 ± 0.042 | 0.820 ± 0.046 |
  | last3 | 0.814 ± 0.044 | 0.813 ± 0.048 |
  | central | 0.848 ± 0.069 | 0.856 ± 0.073 |

  ± is the population standard deviation (ddof = 0) across seeds, the convention
  that reproduces all 14 printed ± values in Table 5.11 (CP and GC).

- **p. 81.** The sentence quoting first1 = 0.942, first3 = 0.899, last1 = 0.819
  and last3 = 0.814 should read first1 = 0.932, first3 = 0.896, last1 = 0.826 and
  last3 = 0.813.
- The CP values in Table 5.11 are correct as printed. The qualitative finding is
  unchanged: GC is higher when the first bars are masked (0.896–0.932) than when
  the last bars are masked (0.813–0.826).
- **Provenance.** Recomputed with `scripts/evaluation/compute_muspy_metrics.py` (formerly in `scripts/Evaluation_Infilling/`)
  (metric code unchanged since commit `ec9ea11`; muspy 0.5.0) on the original MIDI outputs of
  the extreme-infilling runs (first/last configurations) and of the 3-bar
  fine-tuned run (central). Averaging the same per-seed values over seeds 1–5 and
  10 reproduces every printed GC value exactly.

## Figure 5.5 — extreme infilling CP (p. 80)

- **Figure 5.5 (p. 80).** The bars are means over seeds 1–5, as captioned. The
  dashed reference line for central 3-bar infilling, however, is drawn at
  CP = 0.622, the mean over seeds 1–5 **and** seed 10. The caption's value,
  0.602 ± 0.265 (seeds 1–5, as in Table 5.11), is correct; only the line's
  position is affected. This is the same seed-set issue as the Table 5.11 GC
  column; no conclusion changes.

## Pending verification

- **Hardware:** pending verification.
- **Training schedule:** pending verification.
