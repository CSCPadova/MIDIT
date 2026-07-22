<div align="center">

# 🎹 MIDIT

### Bidirectional Masked Language Modeling for Symbolic Music Infilling

*A BERT-style Transformer that reconstructs missing bars of multi-track MIDI —<br>using the musical context **before and after** the gap.*

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.x-792EE5.svg)](https://lightning.ai/)
[![MidiTok](https://img.shields.io/badge/MidiTok-REMI%2B-green.svg)](https://github.com/Natooz/MidiTok)
[![Checkpoints](https://img.shields.io/badge/🤗%20Hub-checkpoints-yellow.svg)](https://huggingface.co/SimoneZanetti00/MIDIT-checkpoints)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Master's thesis in Data Science — University of Padova**<br>
*Candidate:* Simone Zanetti · *Supervisor:* Prof. Antonio Rodà · *Co-supervisor:* PhD Matteo Spanio

</div>

---

## 🎯 What is MIDIT?

Autoregressive music models (GPT-style) generate strictly left-to-right: when filling a gap in the middle of a piece, the music *after* the gap is available and musically informative — yet ignored by design. A musician completing a phrase knows both what comes before **and** what comes after.

**MIDIT** applies bidirectional Masked Language Modeling to this problem:

- 🎼 **REMI+ tokenization** (MidiTok): multi-track MIDI as a single interleaved token stream with explicit metric structure
- 🧠 **12-layer bidirectional TransformerEncoder** (~42.5M parameters, trained from scratch on the Lakh MIDI Dataset)
- 🎯 **9 type-specific prediction heads** with a weighted multi-head loss — one head per REMI+ token type
- 🔧 **Two-phase training**: BERT-style pre-training, then **BarFillMasking** fine-tuning that aligns the training distribution with the real infilling task
- ⚡ **Single-pass parallel inference**: all masked positions predicted at once (a fraction of a second per sample vs. ~6.6 s for an autoregressive baseline)

> **Task definition.** MIDIT performs *content infilling on a given rhythmic scaffold*: structural tokens (Bar, Position, Tempo, TimeSig, Rest) are preserved, and the model predicts the musical content (Pitch, Velocity, Duration, Program) of the masked bars. It predicts **what** to play, not **when** — the "rewrite these bars" use case of a DAW workflow.

---

## 📊 Key Results

### Pre-training (Phase 1 — MLM cloze accuracy, best checkpoint `epoch=14, val_loss=2.7254`)

| Metric | Value |
|---|---|
| Avg Top-1 cloze accuracy | **83.18%** |
| Avg Top-5 cloze accuracy | **94.00%** |
| Pitch Top-1 | **84.29%** |

*Averages exclude the trivially predictable Bar (100%) and TimeSig (98.4%) types.*

### The core finding: zero-shot collapse → diagnosed → fixed

Pre-trained with 15% scattered masking, the model **collapses** on contiguous-bar infilling (pitch accuracy: 26.2% at 1 bar → **0.8%** at 5 bars). An ablation isolating masking *structure* from masking *rate* shows the cause is structural: at the **same ~42% mask rate**, Content Preservation is **0.966** (scattered) vs **0.128** (contiguous bars).

**BarFillMasking fine-tuning** (5 epochs, `infilling-epoch=05`) closes the gap:

| Masked bars | Zero-shot pitch acc. | Fine-tuned | Δ |
|---|---|---|---|
| 1 | 26.2% | 38.5% | +12.3 pp |
| 3 | 10.9% | **40.1%** | +29.2 pp |
| 5 | 0.8% | **33.9%** | **+33.1 pp** |

### Large-scale validation (50 songs, 3 masked bars)

| Model | CP ↑ | GS ↑ | PCHE ↓ |
|---|---|---|---|
| Zero-shot | 0.297 | 0.861 | 0.487 |
| **Fine-tuned** | **0.681** | **0.966** | **0.320** |

### Against the state of the art (paired, 892 matched samples, same protocol)

| Metric | **MIDIT FT** | Anticipatory MT | p | Cohen's d |
|---|---|---|---|---|
| CP ↑ | **0.729 ± 0.19** | 0.435 ± 0.25 | < 0.001 | **1.16** |
| PCHE ↓ | **0.323 ± 0.31** | 0.790 ± 0.80 | < 0.001 | 0.57 |

*GS is not comparable across paradigms (MIDIT preserves Position tokens; Anticipatory regenerates onsets) and is therefore reported separately.*

### Verified genuinely generative 🔬

Inference integrity was audited mechanically: the model input contains `[MASK]` at every masked position (runtime probe: 20/20 slots), predictions come from head-wise argmax over logits only, **46–73% of Pitch tokens differ** from the ground truth with plausible in-register substitutions, and shuffling only the *context* bars changes 12/20 predictions — predictions are conditioned on musical context, mechanically excluding any copy path. See `scripts/audit_fixes/`.

---

## 🏗️ Architecture

```
Input: REMI+ token sequence (vocab 550, up to 8,192 positions), content tokens → [MASK]
  │
  ├─ Token Embedding (550 × 512) + learned Positional Embedding
  │
  ├─ 12 × TransformerEncoder layer (pre-norm)
  │     · Multi-Head Self-Attention (h = 8, d_model = 512)
  │     · Feed-Forward (dim = 2048, GELU) · dropout 0.1
  │     · no causal mask — full bidirectional attention
  │
  └─ Multi-Head Output: 9 type-specific linear heads
        Pitch · Velocity · Duration · Program · Position · Bar · Tempo · TimeSig · Rest

Loss:  L = Σₜ λₜ · (1/|Mₜ|) · Σᵢ∈Mₜ CE(yₜ⁽ⁱ⁾, ŷₜ⁽ⁱ⁾)     (per-type masked-token normalization)
```

**Tokenizer** — MidiTok REMI+ (V2 config, vocab = 550):

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

---

## 🗃️ Dataset

| | |
|---|---|
| **Source** | [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) (`lmd_full`, ~168k multi-track files) |
| **Training subset** | 50,000 files |
| **Quality filters** | ≥ 50 notes · ≥ 15 s duration · velocity diversity (80.3% survival) |
| **Augmentation** | pitch shifting ±5 semitones (×11 per file, drums excluded) |
| **Split** | 80/10/10 **song-level** (all transposed variants share the split — zero leakage), seed = 42 |
| **Output** | 441,518 pre-tokenized `.pt` sequences (66 GB), ~14.5M 8-bar training chunks |

---

## 📁 Repository Structure

```
MIDIT/
├── midit/
│   ├── model/
│   │   ├── masked_transformer.py      # Main model (Lightning module)
│   │   ├── multi_head_output.py       # 9 prediction heads + weighted loss
│   │   └── text_encoder.py            # FLAN-T5 (Phase 2, deferred — not used)
│   ├── data/
│   │   ├── lakh_dataset.py            # Dataset, chunking, song-level split
│   │   └── masking.py                 # TokenLevelMasking, BarFillMasking
│   ├── train_phase1.py                # Phase 1: MLM pre-training
│   ├── train_infilling.py             # Phase 1b: BarFillMasking fine-tuning
│   └── infer.py                       # Infilling inference (validation set)
├── scripts/
│   ├── infer_from_midi.py             # Infilling on an external MIDI file
│   ├── generate_samples_batch.py      # Batch generation (N seeds, one model load)
│   ├── Evaluation_Accuracy/           # Top-1/Top-5 cloze evaluation
│   ├── Evaluation_Infilling/          # CP / GS / PCHE + JS similarity pipeline
│   ├── Pre_Processing/                # Tokenization & preprocessing pipeline
│   └── audit_fixes/                   # Evaluation-integrity audit & verification probes
├── conf/                              # Hydra configs (phase 1 / infilling)
├── slurm/                             # SLURM job scripts (DEI cluster)
└── thesis_materials/
```

---

## ⚙️ Installation

Requires **Python 3.12** and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/SimoneZanetti00/MIDIT.git
cd MIDIT
git checkout masked-transformer

uv sync
source .venv/bin/activate
```

---

## 📦 Pretrained Checkpoints

Pretrained weights are hosted on the [Hugging Face Hub](https://huggingface.co/SimoneZanetti00/MIDIT-checkpoints):

| Checkpoint | Phase | Description |
|---|---|---|
| [`midit-phase1-epoch14.ckpt`](https://huggingface.co/SimoneZanetti00/MIDIT-checkpoints/resolve/main/midit-phase1-epoch14.ckpt) | Phase 1 (MLM pre-training) | base / zero-shot model, val_loss 2.7254 |
| [`midit-infilling-epoch05.ckpt`](https://huggingface.co/SimoneZanetti00/MIDIT-checkpoints/resolve/main/midit-infilling-epoch05.ckpt) | Phase 1b (BarFillMasking FT) | fine-tuned infilling model — **used for all thesis results** |

Download both into `logs/checkpoints/`:

```bash
uv pip install huggingface_hub    # or: pip install huggingface_hub

hf download SimoneZanetti00/MIDIT-checkpoints \
    midit-infilling-epoch05.ckpt midit-phase1-epoch14.ckpt \
    --local-dir logs/checkpoints/
```

Weights contain the Lightning `state_dict` + hyperparameters only (optimizer state removed). Trained on the Lakh MIDI Dataset; released for research purposes.

---

## 🚀 Quickstart — Infilling on your own MIDI

This is the **out-of-the-box** path: it needs only the installed package and a
checkpoint. The input MIDI is tokenized on the fly — **no dataset required.**

```bash
python scripts/infer_from_midi.py \
    --midi_path  path/to/your_song.mid \
    --checkpoint logs/checkpoints/midit-infilling-epoch05.ckpt \
    --mask_bars 3 4 5 \
    --start_bar 12 \
    --output_dir outputs/my_song/
```

Produces `original.mid`, `masked.mid` and `reconstructed.mid` in `--output_dir`.

> ⚠️ Unless `--skip_filters` is passed, the input MIDI must pass the same quality
> filters used in training (≥ 50 notes, ≥ 15 s, velocity diversity), otherwise the
> script exits. Inference is **deterministic** (argmax decoding): identical inputs
> produce byte-identical reconstructions across runs.

<details>
<summary><b>Commands that require the preprocessed dataset</b></summary>

The following reproduce the thesis evaluation on held-out validation songs and
therefore need the preprocessed `.pt` dataset (`--data_dir`). See
[Reproducing the dataset](#-reproducing-the-dataset) to build it — it is **not**
distributed (66 GB).

```bash
# Infilling on a validation-set sample (masks bars 3-5, reconstructs them)
python midit/infer.py \
    --checkpoint logs/checkpoints/midit-infilling-epoch05.ckpt \
    --data_dir   data/preprocessed_v2_50k \
    --seed 5 --mask_bars 3 4 5 --output_dir outputs/demo_seed5/

# Batch generation (loads the model once, generates N samples)
python scripts/generate_samples_batch.py \
    --checkpoint logs/checkpoints/midit-infilling-epoch05.ckpt \
    --data_dir   data/preprocessed_v2_50k \
    --output_dir outputs/samples_1000 \
    --n_seeds 1000 --start_seed 1 --mask_bars 3 4 5
```

</details>

---

## 🔧 Reproducing the dataset

The preprocessed dataset (441k `.pt` files, 66 GB) is **not distributed**. To
rebuild it from scratch:

1. Download the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)
   (`lmd_full`) and place the raw `.mid` files at **`data/midi_files/lmd_full/`**
   (this path is hard-coded in the preprocessing script; edit the `DATASET_DIR`
   constant to use another location).

2. Run the single-stage preprocessing pipeline (filter → tokenize → augment →
   save):

   ```bash
   python scripts/Pre_Processing/preprocess_v2.py \
       --output_dir data/preprocessed_v2_50k \
       --num_workers 32
   ```

   Optional flags: `--max_files N` (0 = all), `--dry_run`, `--seed 42`.

> **Tokenizer note.** The MidiTok REMI+ config is rebuilt in code (see the
> `TokenizerConfig` above), never shipped as a file. The 550-token vocabulary is
> deterministic from that config, so reproducing the exact vocabulary requires
> the same config **and** a matching MidiTok version.

---

## 🏋️ Training

**Phase 1 — MLM pre-training** (BERT-style 15% token-level masking):

```bash
python midit/train_phase1.py                      # local / single GPU

sbatch slurm/train_phase1_v3.sh \                 # SLURM cluster
    train.batch_size=2 train.accumulate_grad_batches=4 train.lr=3e-4
```

| Parameter | Value |
|---|---|
| Optimizer | AdamW (weight decay 0.01) |
| LR schedule | 1,500-step warmup → peak 3×10⁻⁴ → cosine decay |
| Effective batch | 8 (batch 2 × accumulation 4) |
| Precision | bf16 · gradient clipping 1.0 |
| Epoch | 200k batches (~7% of available chunks) |
| Hardware | 1 × NVIDIA A40 48 GB |

**Phase 1b — infilling fine-tuning** (BarFillMasking: 1–3 contiguous bars, content tokens only, structure preserved):

```bash
python midit/train_infilling.py \
    model.checkpoint_path=logs/checkpoints/midit-phase1-epoch14.ckpt \
    train.lr=3e-5
```

> ℹ️ The infilling validation loss (~8.6) is **not comparable** to Phase 1 (~2.7): Phase 1b masks only high-entropy content tokens, while Phase 1 averages over trivially predictable structural tokens.

---

## 📏 Evaluation

> All evaluation commands below operate on the preprocessed dataset or on batch-generated samples — see [Reproducing the dataset](#-reproducing-the-dataset).

**Cloze accuracy (Top-1 / Top-5, per token type):**

```bash
python scripts/Evaluation_Accuracy/evaluate_phase1_accuracy_DIRECT.py \
    --checkpoint logs/checkpoints/midit-phase1-epoch14.ckpt \
    --data_dir   data/preprocessed_v2_50k \
    --split val --num_samples 400
```

**Infilling metrics (CP / GS / PCHE) on a results directory:**

```bash
python scripts/Evaluation_Infilling/compute_muspy_metrics.py \
    --results_dir outputs/samples_1000 \
    --output_csv  outputs/metrics/samples_1000.csv
```

**JS Similarity pipeline** (distributional comparison vs. the LMD corpus):

```bash
# 1. reference aggregate (LMD corpus statistics)
python scripts/Evaluation_Infilling/Matteo/compute_reference_agg.py \
    --midi_dir data/midi_files/lmd_full/ --output_json outputs/ref_agg.json \
    --max_files 1000 --seed 42

# 2. model aggregate (masked-bar crops of the reconstructions)
python scripts/Evaluation_Infilling/Matteo/compute_model_agg.py \
    --samples_dir outputs/samples_1000 --output_json outputs/model_agg.json

# 3. JS score
python scripts/Evaluation_Infilling/Matteo/compute_js_score.py \
    --ref_agg outputs/ref_agg.json --model_agg outputs/model_agg.json
```

**Integrity & verification probes** (audit that inference cannot copy the input):

```bash
python scripts/audit_fixes/check_sample_integrity.py --samples_dir outputs/samples_1000
python scripts/audit_fixes/probe_no_cheat.py \
    --checkpoint logs/checkpoints/midit-infilling-epoch05.ckpt \
    --data_dir data/preprocessed_v2_50k --seed 79 --n_slots 20 --mask_bars 3 4 5
```

---

## 🧰 Tech Stack

| Tool | Role |
|---|---|
| [PyTorch](https://pytorch.org/) + [Lightning](https://lightning.ai/) | Model & training loop |
| [MidiTok](https://github.com/Natooz/MidiTok) | REMI+ tokenization |
| [Hydra](https://hydra.cc/) | Configuration management |
| [Weights & Biases](https://wandb.ai/) | Experiment tracking |
| [MusPy](https://github.com/salu133445/muspy) / symusic | Evaluation metrics |
| [uv](https://github.com/astral-sh/uv) | Environment management |
| SLURM | Cluster scheduling (DEI, University of Padova) |

---

## ⚠️ Known Limitations

Disclosed and quantified in the thesis:

- **Position scaffolding** — the rhythmic grid is given; MIDIT predicts content, not onsets (Option B protocol, consistent with MusicBERT)
- **Oracle token-type selection** at inference (mitigated: the type is always observable from REMI+ grammar)
- **Deterministic argmax decoding** — one completion per input; stochastic sampling is future work
- **Local 8-bar context** — no long-range song structure
- **50k / 168k files** — Pitch and Tempo learning curves still descending at the best checkpoint

**Future work:** full-dataset training with shard-based I/O · stochastic decoding · BPE ablation · Phase 2 text-conditioned infilling (MidiCaps + FLAN-T5).

---

## 📚 References

- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers*, NAACL 2019
- Zeng et al., *MusicBERT*, ACL 2021 · Hsiao et al., *MidiBERT-Piano*, 2021
- Huang & Yang, *REMI (Pop Music Transformer)*, ACM MM 2020 · von Rütte et al., *FIGARO (REMI+)*, ICLR 2023
- Fradet et al., *MidiTok*, ISMIR 2021
- Thickstun et al., *Anticipatory Music Transformer*, 2023
- Raffel, *The Lakh MIDI Dataset*, 2016

## 🙏 Acknowledgements

Developed at the University of Padova. Cluster infrastructure based on the [DEI Blade Template](https://github.com/CSCPadova/dei-blade-template). Evaluation metric utilities (`Matteo/`) courtesy of PhD Matteo Spanio.

## 📄 License

[MIT](LICENSE)
