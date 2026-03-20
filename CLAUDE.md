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
│   ├── Analyze_dataset/              # 8 dataset analysis scripts
│   ├── Evaluation_Accuracy/
│   │   └── evaluate_phase1_accuracy_DIRECT.py  # Evaluation script (Top-1, Top-5)
│   ├── Pre_Processing/
│   │   ├── preprocess_batch_improved.py        # V1 preprocessing script (deprecated)
│   │   └── preprocess_v2.py                    # V2 preprocessing script (current)
│   ├── Tests/
│   │   └── validate_v2_preprocessing.py        # V2 preprocessing validation (5 tests)
│   └── Tokenization/
│       ├── visualize_tokenization.py           # V1 tokenization visualization
│       └── visualize_tokenization_v2.py        # V2 tokenization visualization (current)
├── conf/
│   └── config.yaml                   # Hydra config — all hyperparameters
├── slurm/
│   └── train_phase1.sh               # SLURM job script (gpu:a40:1)
└── logs/
    └── checkpoints_phase1/
        └── phase1-epoch=48-val_loss=2.3922.ckpt  # BEST V1 MODEL — do not delete
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

Input embedding:   550 × 512   (V2 vocab; V1 runs used 518 × 512)
Output:            MultiHeadOutput (9 separate linear heads)
Total parameters:  ~42.5M (all trainable)
```

This is NOT a pretrained model. All weights are randomly initialized and trained on Lakh MIDI.

---

## Tokenization — REMI+

**Tokenizer:** MidiTok REMI
**Vocabulary size:** 550 tokens (V2)

### TokenizerConfig (V2 — current):
```python
TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8, (4, 12): 4},   # extended range vs V1
    num_velocities=32,
    use_programs=True,
    one_token_stream_for_programs=True,
    use_time_signatures=True,
    use_tempos=True,
    use_rests=True,
    use_chords=False,
    use_pitchdrum_tokens=True,
    special_tokens=["PAD", "BOS", "EOS", "MASK"],
    delete_equal_successive_tempo_changes=True,    # new in V2
    delete_equal_successive_time_sig_changes=True, # new in V2
)
```

**V1 config (deprecated):** `beat_res={(0, 4): 8}`, no `delete_equal_successive_*` flags, vocab=518.

### Token types and counts:
| Type | Count | IDs |
|---|---|---|
| PAD/BOS/EOS/MASK | 4 | 0,1,2,3 |
| Pitch | 89 | melodic notes |
| PitchDrum | 62 | percussion (Program=-1) |
| Velocity | 32 | 32 quantized bins |
| Duration | 64 | beat.subdivision.resolution format (32 in V1) |
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
**Full dataset available:** ~160,000 files at `data/midi_files/lmd_full/`

### V2 Preprocessing pipeline (`scripts/Pre_Processing/preprocess_v2.py`):
- **Quality filters (applied before tokenization):**
  - F1: skip files with fewer than 50 notes (too sparse)
  - F2: skip files shorter than 15 seconds (too short)
  - F3: skip files where top-1 velocity bin > 90% of all notes (constant velocity — low quality)
- **Pitch shifting augmentation:** 11 shifts per file (−5 to +5 semitones), drum tracks unchanged, pitches clipped to [21, 108]
- **Output format:** individual `.pt` files, same structure as V1, with added metadata fields:
  - `pitch_shift`: integer (0 for original, ±N for augmented)
  - `tokenizer_ver`: `"v2"`
- **File naming:** `{stem}.pt` (shift=0), `{stem}_shift+N.pt` (positive), `{stem}_shift-N.pt` (negative)

### Preprocessing output:
- Format: `.pt` files (pre-tokenized PyTorch tensors)
- Structure: `{'tokens': Tensor, 'bar_positions': List[int], 'metadata': dict}`
- Strategy: offline preprocessing — tokenize once, load tensors during training

### Chunking strategy:
- Method: `bar_based` — respects musical bar boundaries
- Bars per chunk: 8
- Max tokens: 8192
- Result: one MIDI file → multiple training examples (~10 per file on average)

### Current preprocessed datasets (cluster):
| Path | Source files | Status |
|---|---|---|
| `~/MIDIT/data/preprocessed_phase1/` | 9,987 files, V1 | Done — V1 training only |
| `/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_1k/` | ~800 files | Test set |
| `/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_5k/` | 3,000 files | Validated ✓ |
| `/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k/` | ~50,000 files | In progress (SLURM) |

### Validation:
`scripts/Tests/validate_v2_preprocessing.py` — 5 automated tests (vocab range, bar positions, pitch shift correctness, chunking, masking). All pass on preprocessed_v2_5k.

### Storage (cluster):
- Raw MIDI: `~/MIDIT/data/midi_files/` (~7.9GB)
- V1 preprocessed: `~/MIDIT/data/preprocessed_phase1/` (~2.2GB)
- V2 preprocessed: `/nfsd/tesi/zanettisim/MIDIT/data/` (1.9TB available)

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
| **Run 3 ★** | **3876912** | **+ LR fix + denominator fix** | **19.18%** | **49.02%** | **77.43%** | **2.3922** | **BEST V1 MODEL** |
| Run 4 | 3887951 | batch=16, lr=3e-4, A40 | 17.98% | 47.57% | 75.16% | 2.4517 | Worse — LR not scaled |
| Run 5 | 3891904 | batch=16, lr=8e-4 | — | — | — | diverged ~4.1 | Cancelled at epoch 46 |
| Run 6 | 3915569 | batch=2, lr=3e-4, lambda rebalanced | TBD | TBD | TBD | TBD | Completed — metrics TBD |
| Run 7 | 3979604 | V2 data (3k files), batch=2, lr=3e-4 | — | — | — | ~17 @ ep20 | V2 pipeline test — small dataset |

**Best V1 checkpoint:** `logs/checkpoints_phase1/phase1-epoch=48-val_loss=2.3922.ckpt`
Do not delete this file.

### Plateau analysis (V1, 9,987 files):
~49% Avg Top-1 is a structural ceiling — not a bug. All 5 V1 runs converge to the same value.
Root causes:
1. Dataset too small (9,987 files, no augmentation)
2. Pitch prediction is inherently hard (151 pitch classes, depends on global harmony)
3. No BPE compression — long sequences without compression

### V2 val_loss note:
Run 7 val_loss ~17 on 3k files is expected to be much higher than V1 (~2.39): larger vocab (550 vs 518), more Duration tokens (64 vs 32), and a much smaller training set than V1. Not a regression — it is a pipeline validation run only.

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

### 7. val_loss=0 bug (March 2026)
**File:** `midit/train_phase1.py`
**Issue:** Validation dataset was instantiated with `apply_masking=False` — all mask_labels were False, all targets were -100, loss denominator was 1, val_loss was always 0
**Fix:** `val_dataset = LakhDataset(cfg, split='val', apply_masking=True)` — MLM validation requires masking

---

## Evaluation Methodology

**Script:** `scripts/Evaluation_Accuracy/evaluate_phase1_accuracy_DIRECT.py`
**Samples:** 500 files from preprocessed dataset
**Masking:** 100% mask rate (no random/unchanged substitution) for clean evaluation
**Metrics:** Top-1 and Top-5 accuracy, computed separately per token type

### Current best results (Run 3, V1 data):
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
| `~/MIDIT/` | 18GB | ~2GB | Code + V1 data (quota 20GB) |
| `/nfsd/tesi/zanettisim/` | 2.0TB | 1.9TB | V2 preprocessed data |

### GPU nodes available:
- `gpu1`: 8× A40 (48GB) — 8 CPU cores per GPU limit
- `gpu2/3`: 8× RTX 3090 (24GB) — 6 CPU cores per GPU limit
- `gpu7`: 8× L40S (48GB) — 8 CPU cores per GPU limit

### CPU nodes for preprocessing (no GPU needed):
- `runner-07/10`: 96 cores, 3TB RAM
- `runner-11`: 192 cores, 6TB RAM

---

## Current Status (March 2026)

### Phase 1 — V2 pipeline: in progress
- V2 preprocessing script complete and validated (5/5 tests pass on preprocessed_v2_5k)
- V2 pipeline test (Run 7, 3k files): val_loss ~17 — confirmed pipeline works end-to-end
- preprocessing_v2_50k: SLURM job running on cluster
- W&B logging active alongside TensorBoard: project `MIDIT-Phase1`

### Immediate next tasks:

**TASK 1 — DONE: V2 preprocessing pipeline**
- Script: `scripts/Pre_Processing/preprocess_v2.py`
- Validation: `scripts/Tests/validate_v2_preprocessing.py` — all 5 tests pass
- Currently preprocessing ~50k files on cluster

**TASK 2 — Training on V2 preprocessed data**
- Wait for preprocessed_v2_50k to complete, then launch training
- Same architecture as Run 3
- `lambda_bar=0.0`, `lambda_timesig=0.0` (confirmed by Matteo)
- Linear LR scaling if batch size changes: `new_lr = base_lr × (new_bs / base_bs)`
- DDP (Distributed Data Parallel) with 2 GPUs if available — use `torchrun`
- Eventually scale to full ~168k files (11× augmented)

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
| Weights & Biases | experiment tracking (project: MIDIT-Phase1) |
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
