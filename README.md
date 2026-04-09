# MIDIT — A BERT-style Masked Language Model for Symbolic MIDI Music Generation

Master's thesis project — University of Padova, DEI Department  
**Student:** Simone Zanetti | **Supervisor:** Prof. Matteo Spanio

---

## Overview

MIDIT is a BERT-style Masked Language Model (MLM) for symbolic MIDI music generation. It operates on REMI+ tokenized MIDI sequences and is trained in two phases:

- **Phase 1 (complete):** Unconditional MLM pre-training on the Lakh MIDI Dataset (~50k files with pitch-shifting augmentation)
- **Phase 2 (planned):** Text-conditioned fine-tuning on MidiCaps with FLAN-T5

The model uses a standard PyTorch TransformerEncoder (12 layers, 8 heads, d_model=512, ~42.5M parameters) with a **multi-head output layer** — 9 separate linear prediction heads, one per REMI+ token type (Pitch, Velocity, Duration, Program, Position, Bar, Tempo, TimeSig, Rest).

---

## Key Results (Phase 1)

| Metric | Value |
|--------|-------|
| Avg Top-1 Cloze Accuracy | 81.88% |
| Avg Top-5 Cloze Accuracy | 93.85% |
| Pitch Top-1 | 84.43% |
| Best checkpoint | epoch=14, val_loss=2.7254 |

Evaluated on the validation split (44,151 files, seed=42, 400 samples).

---

## Architecture
TransformerEncoder:
num_layers:      12
d_model:         512
nhead:           8
dim_feedforward: 2048
dropout:         0.1
norm_first:      True  (pre-norm)
Input embedding:   550 × 512  (REMI+ V2 vocabulary)
Output:            MultiHeadOutput — 9 separate linear heads
Total parameters:  ~42.5M (all trainable, no pretrained weights)

---

## Tokenization — REMI+ V2

Tokenizer: MidiTok REMI with the following configuration:

```python
TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8, (4, 12): 4},
    num_velocities=32,
    use_programs=True,
    one_token_stream_for_programs=True,
    use_time_signatures=True,
    use_tempos=True,
    use_rests=True,
    delete_equal_successive_tempo_changes=True,
    delete_equal_successive_time_sig_changes=True,
)
```

Vocabulary size: **550 tokens**. All instruments are merged into a single interleaved stream ordered by time position.

---

## Dataset

**Source:** Lakh MIDI Dataset (lmd_full, ~168k files available)  
**Current training subset:** 50,000 files  
**Preprocessing:** Offline tokenization to `.pt` files  
**Augmentation:** Pitch shifting ±5 semitones (×11 per file)  
**Quality filters:** min 50 notes, min 15s duration, velocity diversity  
**Split:** 80/10/10 (train/val/test), seed=42, deterministic  
**Output:** 441k `.pt` files, 66GB


```
## Repository Structure
MIDIT/
├── midit/
│   ├── model/
│   │   ├── masked_transformer.py     # Main model
│   │   ├── multi_head_output.py      # 9 prediction heads
│   │   └── text_encoder.py           # FLAN-T5 (Phase 2, not yet used)
│   ├── data/
│   │   ├── lakh_dataset.py           # Dataset class with chunking
│   │   └── masking.py                # TokenLevelMasking, BarFillMasking
│   ├── train_phase1.py               # Phase 1 training script
│   └── infer.py                      # Infilling inference script
├── scripts/
│   ├── Evaluation_Accuracy/          # Top-1/Top-5 cloze evaluation
│   ├── Evaluation_Infilling/         # CP/GS/PCHE infilling metrics
│   └── Pre_Processing/               # Preprocessing pipeline
├── conf/
│   └── config.yaml                   # Hydra configuration
└── slurm/
    └── train_phase1_v2_50k_safe.sh   # SLURM job script
```


## Installation

```bash
# Clone the repository
git clone https://github.com/SimoneZanetti00/MIDIT.git
cd MIDIT
git checkout masked-transformer

# Create virtual environment (requires Python 3.12)
uv sync
source .venv/bin/activate
```

---

## Training

```bash
# Phase 1 — MLM pre-training
python midit/train_phase1.py

# Or on SLURM cluster
sbatch slurm/train_phase1_v2_50k_safe.sh
```

Key training configuration (see `conf/config.yaml`):

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Warmup steps | 1500 |
| Effective batch size | 8 (batch=2, accumulate=4) |
| Precision | bf16 |
| Epochs | 50 |
| Limit train batches | 200k/epoch |

---

## Evaluation

```bash
# Cloze accuracy (Top-1 / Top-5)
python scripts/Evaluation_Accuracy/evaluate_phase1_accuracy_DIRECT.py \
    --checkpoint logs/checkpoints_phase1/<checkpoint>.ckpt \
    --data_dir /path/to/preprocessed_v2_50k \
    --split val \
    --num_samples 400

# Infilling — generate masked MIDI and compare with original
python midit/infer.py \
    --checkpoint logs/checkpoints_phase1/<checkpoint>.ckpt \
    --data_dir /path/to/preprocessed_v2_50k \
    --output_dir outputs/infilling/ \
    --seed 10 \
    --mask_bars 4

# CP / GS / PCHE metrics on infilling output
python scripts/Evaluation_Infilling/compute_infilling_metrics.py \
    --original  outputs/infilling/original.mid \
    --infilled   outputs/infilling/reconstructed.mid \
    --bar_start 4 --bar_end 4
```

---

## Tech Stack

| Tool | Role |
|------|------|
| PyTorch | Model implementation |
| PyTorch Lightning | Training loop |
| MidiTok | REMI+ tokenization |
| Hydra | Configuration management |
| Weights & Biases | Experiment tracking |
| SLURM | Cluster job scheduling |
| symusic / pretty_midi | MIDI processing for metrics |

---

## Cluster (DEI UniPD)

- **Login:** `user@login.dei.unipd.it`
- **GPU:** NVIDIA A40 48GB (SLURM partition: allgroups)
- **Data:** `/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_50k/`
- **Checkpoints:** `/nfsd/tesi/zanettisim/MIDIT/logs/checkpoints_phase1/`

---

## Acknowledgements

Built on top of the [DEI Blade Template](https://github.com/CSCPadova/dei-blade-template) by Prof. Matteo Spanio (University of Padova).

References: MusicBERT (Zeng et al., ACL 2021), MidiBERT-Piano (Hsiao et al., 2021), BERT (Devlin et al., 2019), MidiTok (Fradet et al., 2021), REMI (Huang & Yang, 2020).

---

## License

MIT License
