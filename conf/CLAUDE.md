# CLAUDE.md — MIDIT Project Context

This file provides full context for Claude Code when working on the MIDIT repository.
Read this before making any changes to the codebase.

---

## Project Overview

**MIDIT** is a BERT-style Masked Language Model (MLM) for symbolic music generation.
It operates on REMI+ tokenized MIDI sequences and is trained in two phases:

- **Phase 1:** Unconditional MLM pre-training on the Lakh MIDI Dataset
- **Phase 2:** Text-conditioned fine-tuning on MidiCaps (not yet started)

**Thesis student:** Simone Zanetti
**Supervisor:** Matteo Spanio (DEI, University of Padova)
**Repository:** github.com/SimoneZanetti00/MIDIT (branch: simone-dev)

---

## Repository Structure

```
MIDIT/
├── midit/
│   ├── model/
│   │   ├── masked_transformer.py     # Main model — TransformerEncoder + Phase 2 path
│   │   ├── multi_head_output.py      # 9 separate prediction heads (one per token type)
│   │   └── text_encoder.py           # FlanT5TextEncoder (Phase 2, not yet used)
│   ├── data/
│   │   └── lakh_dataset.py           # Dataset class — loads .pt files, chunking, masking
│   └── train_phase1.py               # Phase 1 training script
├── scripts/
│   ├── Evaluation_Accuracy/
│   │   └── evaluate_phase1_accuracy_DIRECT.py  # Evaluation script (Top-1, Top-5)
│   ├── Pre_Processing/
│   │   └── preprocess_batch_improved.py        # Preprocessing script
│   └── Tokenization/
│       └── visualize_tokenization.py           # Tokenization visualization tool
├── conf/
│   └── config.yaml                   # Hydra config — all hyperparameters
├── slurm/
│   └── train_phase1.sh               # SLURM job script (gpu:a40:1)
└── logs/
    └── checkpoints_phase1/
        └── phase1-epoch=48-val_loss=2.3922.ckpt  # BEST MODEL — do not delete
```

---

## Model Architecture

Standard PyTorch TransformerEncoder trained from scratch (no pretrained weights).

```
TransformerEncoder:
  num_layers:      12
  d_model:         512
  nhead:           8
  dim_feedforward: 2048
  dropout:         0.1

Input embedding:   518 × 512
Output:            MultiHeadOutput (9 separate linear heads)
Total parameters:  ~42.5M (all trainable)
```

This is NOT a pretrained model. All weights are randomly initialized and trained on Lakh MIDI.

---

## Tokenization — REMI+

**Tokenizer:** MidiTok REMI
**Vocabulary size:** 518 tokens

### TokenizerConfig (current — preprocessing already done):
```python
TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8},
    num_velocities=32,
    use_programs=True,
    one_token_stream_for_programs=True,
    use_time_signatures=True,
    use_tempos=True,
    use_rests=True,
    use_chords=False,
    use_pitchdrum_tokens=True,
    special_tokens=["PAD", "BOS", "EOS", "MASK"],
)
```

### Token types and counts:
| Type | Count | IDs |
|---|---|---|
| PAD/BOS/EOS/MASK | 4 | 0,1,2,3 |
| Pitch | 89 | melodic notes |
| PitchDrum | 62 | percussion (Program=-1) |
| Velocity | 32 | 32 quantized bins |
| Duration | 32 | beat.subdivision.resolution format |
| Program | 129 | -1=drums, 0-127=GM instruments |
| Position | 96 | position within bar |
| Bar | 1 | Bar_None (max_bar_embedding not set) |
| Tempo | 32 | 32 bins 40-250 BPM |
| TimeSig | 9 | time signature combinations |
| Rest | 32 | rest durations |

### Note structure in stream:
```
[Position] [Program] [Pitch/PitchDrum] [Velocity] [Duration]
```

### Multi-track interleaving:
All instruments are merged into a single token stream, ordered by time position.
Program=-1 indicates drums (uses PitchDrum tokens instead of Pitch tokens).

---

## Dataset

**Source:** Lakh MIDI Dataset (lmd_full)
**Current subset used:** 9,987 files
**Full dataset available:** ~160,000 files at `data/midi_files/lmd_full/`

### Preprocessing output:
- Format: `.pt` files (pre-tokenized PyTorch tensors)
- Location: `data/preprocessed_phase1/`
- Size: ~2.2GB (current subset)
- Strategy: offline preprocessing — tokenize once, load tensors during training

### Chunking strategy:
- Method: `bar_based` — respects musical bar boundaries
- Bars per chunk: 8
- Max tokens: 8192
- Result: one MIDI file → multiple training examples (~10 per file on average)

### Storage (cluster):
- Raw MIDI: `~/MIDIT/data/midi_files/` (~7.9GB)
- Preprocessed: `~/MIDIT/data/preprocessed_phase1/` (~2.2GB)
- New preprocessing target: `/nfsd/tesi/zanettisim/MIDIT/data/` (1.9TB available)

---

## Multi-Head Loss

9 separate prediction heads, one per token type. Weighted loss:

```
L = Σ_t λ_t · CrossEntropy(y_t, ŷ_t)
```

### Current lambda weights (Run 6 config):
```yaml
lambda_weights:
  pitch:    3.0   # tripled — pitch is hardest and most important
  velocity: 1.5
  duration: 1.0
  program:  1.5
  position: 0.5
  bar:      0.0   # zeroed — Bar_None always same token, no signal
  tempo:    2.0
  timesig:  0.5   # reduced — 97% trivial accuracy (same token repeated)
  rest:     1.0
```

**Important:** Bar and TimeSig are confirmed to be removed from loss (lambda=0).
Confirmed by supervisor Matteo Spanio in call on March 12, 2026.

---

## Training History — Phase 1

| Run | Job ID | Config | Pitch Top-1 | Avg Top-1 | Avg Top-5 | val_loss | Outcome |
|---|---|---|---|---|---|---|---|
| Run 1 | 3869532 | batch=2, lr=3e-4, masking+eval fixes | 17.22% | 45.67% | 76.02% | 3.01 | First reliable result |
| Run 2 | 3875354 | + TokenLevelMasking | 18.77% | 48.59% | 77.86% | 2.43 | Improvement |
| **Run 3 ★** | **3876912** | **+ LR fix + denominator fix** | **19.18%** | **49.02%** | **77.43%** | **2.3922** | **BEST MODEL** |
| Run 4 | 3887951 | batch=16, lr=3e-4, A40 | 17.98% | 47.57% | 75.16% | 2.4517 | Worse — LR not scaled |
| Run 5 | 3891904 | batch=16, lr=8e-4 | — | — | — | diverged ~4.1 | Cancelled at epoch 46 |
| Run 6 | 3915569 | batch=2, lr=3e-4, lambda rebalanced | TBD | TBD | TBD | TBD | In progress / completed |

**Best checkpoint:** `logs/checkpoints_phase1/phase1-epoch=48-val_loss=2.3922.ckpt`
Do not delete this file.

### Plateau analysis:
~49% Avg Top-1 is a structural ceiling — not a bug. All 5 runs converge to the same value.
Root causes:
1. Dataset too small (9,987 files, no augmentation)
2. Pitch prediction is inherently hard (151 pitch classes, depends on global harmony)
3. No BPE compression — long sequences without compression

---

## Critical Bugs Fixed (do not reintroduce)

### 1. Double masking bug
**File:** `midit/train_phase1.py` + `midit/model/masked_transformer.py`
**Issue:** Masking was applied twice — effective mask rate was ~2% instead of 15%
**Fix:** Apply masking only once, in the dataset collate function

### 2. Evaluation class mapping bug
**File:** `scripts/Evaluation_Accuracy/evaluate_phase1_accuracy_DIRECT.py`
**Issue:** Token type class mapping was misaligned — accuracy was underestimated
**Fix:** Align class indices with actual token type ordering from tokenizer

### 3. LR scheduler frozen after epoch 2
**File:** `conf/config.yaml` + `midit/train_phase1.py`
**Issue:** Learning rate scheduler stopped updating after epoch 2
**Fix:** Correct scheduler step() call placement in training loop

### 4. Loss denominator dilution
**File:** `midit/model/multi_head_output.py`
**Issue:** Loss denominator included all tokens including non-masked ones — diluted signal for rare tokens (Tempo, TimeSig)
**Fix:** Denominator counts only masked tokens per type

### 5. mask_labels=None bug (December 2025 — historical)
**Issue:** mask_labels was None — loss was computed on all tokens, not just masked ones
**Effect:** Loss artificially deflated ~33× (val_loss=0.09 was fake — real performance was ~40% Top-1)
**Fix:** Correctly pass mask_labels to loss computation
**Note:** All December 2025 checkpoints are invalid due to this bug

### 6. Bar-level masking (replaced)
**Old:** BarLevelMasking — masked token types independently, not complete musical notes
**New:** TokenLevelMasking (BERT-style) — masks complete tokens at 15%, consistent with BERT

---

## Evaluation Methodology

**Script:** `scripts/Evaluation_Accuracy/evaluate_phase1_accuracy_DIRECT.py`
**Samples:** 500 files from preprocessed dataset
**Masking:** 100% mask rate (no random/unchanged substitution) for clean evaluation
**Metrics:** Top-1 and Top-5 accuracy, computed separately per token type

### Current best results (Run 3):
| Token Type | Top-1 | Top-5 |
|---|---|---|
| Pitch | ~19% | ~77% |
| Velocity | ~58% | — |
| Duration | high | — |
| Program | moderate | — |
| Position | moderate | — |
| Bar | ~100% | — | ← trivial (always Bar_None) |
| Tempo | ~21% | — | ← hard (appears 1 token per file) |
| TimeSig | ~97% | — | ← trivial (same token repeated every bar) |
| **Average** | **~49%** | **~77%** |

---

## Cluster Configuration (DEI SLURM)

**Login:** `ssh zanettisim@login.dei.unipd.it`
**Current SLURM script:** `slurm/train_phase1.sh`

### Storage:
| Path | Size | Available | Use |
|---|---|---|---|
| `~/MIDIT/` | 18GB | ~2GB | Code + current data (quota 20GB) |
| `/nfsd/tesi/zanettisim/` | 2.0TB | 1.9TB | **New preprocessing target** |

### GPU nodes available:
- `gpu1`: 8× A40 (48GB) — 8 CPU cores per GPU limit
- `gpu2/3`: 8× RTX 3090 (24GB) — 6 CPU cores per GPU limit
- `gpu7`: 8× L40S (48GB) — 8 CPU cores per GPU limit

### CPU nodes for preprocessing (no GPU needed):
- `runner-07/10`: 96 cores, 3TB RAM
- `runner-11`: 192 cores, 6TB RAM

---

## Current Status (March 2026)

### Phase 1: substantially complete
- Best model Run 3 saved and validated
- Plateau ~49% confirmed as structural (dataset size, no augmentation)
- All critical bugs fixed

### Immediate next tasks:

**TASK 1 — New preprocessing on full Lakh dataset**
- Target: ~160,000 files (vs current 9,987)
- Data augmentation: pitch shifting ±5 semitones (~11× more data)
- Output format: sharded .pt files (few large files instead of many small ones)
- Parallelization: 32-48 CPU cores on cluster
- Output path: `/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2/`
- Config improvements needed:
  - `beat_res={(0,4):8, (4,12):4}` — complete range
  - `max_bar_embedding=8` — model knows position in chunk
  - `delete_equal_successive_tempo_changes=True`
  - `delete_equal_successive_time_sig_changes=True`
  - Filter files with constant Velocity_111 (low quality)

**TASK 2 — Training on full dataset**
- Same architecture as Run 3
- `lambda_bar=0.0`, `lambda_timesig=0.0` (confirmed by Matteo)
- Linear LR scaling if batch size changes: `new_lr = base_lr × (new_bs / base_bs)`
- DDP (Distributed Data Parallel) with 2 GPUs if available — use `torchrun`

**TASK 3 — Qualitative evaluation (Infilling)**
- Mask sequences from real MIDI files
- Let model complete them
- Convert output tokens back to MIDI and listen

**TASK 4 — BPE comparison (separate branch)**
- Incompatible with current 9-head architecture
- Requires separate model branch with standard cross-entropy (single head)
- Must be done AFTER final preprocessing is defined (BPE vocab depends on token frequencies)

### Phase 2: deferred
- Text conditioning on MidiCaps (168,385 MIDI + caption pairs)
- Text encoder: FLAN-T5 (frozen) — already implemented in `midit/model/text_encoder.py`
- Phase 2 path already exists in `masked_transformer.py`
- Missing: `midit/data/midicaps_dataset.py`, `midit/train_phase2.py`, `conf/config_phase2.yaml`

---

## Key Decisions (confirmed with Matteo Spanio, March 12 2026)

- Remove Bar and TimeSig from loss (lambda=0) — confirmed
- Scale to full Lakh dataset with pitch shifting augmentation — confirmed
- Use sharding for preprocessing output — confirmed
- Keep current model architecture for now — confirmed
- BPE as ablation/comparison for thesis — confirmed
- Phase 2 deferred until Phase 1 is solid — confirmed
- Matteo unavailable until mid-May — work autonomously, update via Slack

---

## Tech Stack

| Tool | Version/Details |
|---|---|
| Python | 3.12 |
| PyTorch | latest |
| PyTorch Lightning | used for training loop |
| Hydra | config management (conf/config.yaml) |
| Weights & Biases | experiment tracking |
| MidiTok | REMI tokenization |
| MusPy | musical quality metrics (future) |
| uv | virtual environment management |

---

## Key References for Thesis

- **MusicBERT** (Zeng et al., ACL 2021) — primary baseline, same MLM paradigm
- **MidiBERT-Piano** (Hsiao et al., 2021) — BERT-style on piano MIDI
- **FIGARO** (Rütte et al., 2022) — uses REMI+, same tokenizer
- **MuseTok** — closest existing work (REMI+, same datasets) — mandatory comparison
- **BPE for Symbolic Music** (Fradet et al.) — BPE ablation reference
- **MuseCoco** (Lu et al., 2023) — text-to-symbolic-music, Phase 2 competitor
- **MusicAgent** (Microsoft Muzic) — music AI toolkit reference