# Migration Guide: From MNIST VAE to Masked Diffusion Transformer

This document explains the changes made to transition from the MNIST VAE example to the production Masked Diffusion Transformer for text-to-MIDI generation.

## What Changed

### Removed Components

The following MNIST VAE components were removed (backed up to .bak files):

1. **midit/model.py** (135 lines)
   - `Encoder` class: Simple VAE encoder for 28x28 images
   - `Decoder` class: Simple VAE decoder for 28x28 images  
   - `VAE` class: Lightning module for MNIST generation
   - Reparameterization trick for VAE
   - Image logging to TensorBoard

2. **midit/dataloader.py** (93 lines)
   - `MNISTDataModule`: Lightning data module for MNIST
   - MNIST dataset downloading and preprocessing
   - SSL workarounds for MNIST download

### New Components

#### 1. Model Architecture (`midit/model/`)

**diffusion_transformer.py** (392 lines)
- `SinusoidalPositionEmbeddings`: Timestep encoding for diffusion
- `MaskedDiffusionTransformer`: Main model class
  - Text encoder for conditioning
  - MIDI token embeddings
  - Transformer encoder/decoder architecture
  - Diffusion noise scheduling (linear/cosine)
  - Forward pass with text conditioning
  - Training step with diffusion loss
  - Generation method for inference

**text_encoder.py** (79 lines)
- `TextEncoder`: Transformer-based text encoder
  - Token and position embeddings
  - Multi-layer transformer encoder
  - Attention masking support
  - Pooled output for conditioning

#### 2. Loss Functions (`midit/loss/`)

**diffusion_loss.py** (164 lines)
- `DiffusionLoss`: Flexible loss for diffusion training
  - Cross-entropy loss for discrete tokens
  - MSE loss for continuous noise
  - Hybrid loss combining both
  - Masking support for variable-length sequences
- `MaskedDiffusionLoss`: Extended loss with masking utilities

#### 3. Data Loading (`midit/midicaps_dataloader.py`)

**midicaps_dataloader.py** (211 lines)
- `MidiCapsDataset`: Dataset for text-MIDI pairs
  - Currently uses synthetic data for testing
  - Proper padding and attention masks
  - Ready for HuggingFace MidiCaps integration
- `MidiCapsDataModule`: Lightning data module
  - Train/val/test splits
  - Configurable sequence lengths
  - Fast run mode for testing

#### 4. Configuration Updates

**conf/config.py** (expanded from 28 to 57 lines)
- Added `Model` dataclass: Architecture parameters
- Added `Diffusion` dataclass: Diffusion parameters
- Added `Logging` dataclass: Logger configuration
- Expanded `Data` dataclass: MIDI/text parameters
- Expanded `Train` dataclass: Added fast_run flag

**conf/config.yaml** (expanded from 20 to 41 lines)
- Model configuration (d_model, nhead, layers, etc.)
- Diffusion configuration (timesteps, schedule)
- Logging configuration (wandb, tensorboard)
- Training configuration (fast_run flag)
- Data configuration (vocab sizes, sequence lengths)

#### 5. Training Script Updates

**midit/train.py** (updated, 125 lines)
- Replaced VAE with MaskedDiffusionTransformer
- Replaced MNISTDataModule with MidiCapsDataModule
- Added WandB logger support (configurable)
- Added model checkpointing
- Added fast_run mode
- Improved hyperparameter logging

## Key Differences

### Architecture

| MNIST VAE | Masked Diffusion Transformer |
|-----------|------------------------------|
| Simple encoder-decoder | Transformer encoder-decoder |
| 28×28 images | Variable-length MIDI sequences |
| Latent space sampling | Iterative denoising |
| No conditioning | Text conditioning |
| Single output | Multiple timesteps |

### Training

| MNIST VAE | Masked Diffusion Transformer |
|-----------|------------------------------|
| Reconstruction + KL loss | Cross-entropy loss |
| Single forward pass | Random timestep per batch |
| Fixed image size | Variable sequence lengths |
| No text input | Text conditioning required |
| Image generation | MIDI generation |

### Configuration

| MNIST VAE | Masked Diffusion Transformer |
|-----------|------------------------------|
| latent_dim: 3 | d_model: 512 |
| Simple MLP | Multi-head attention |
| No diffusion params | timesteps, beta_schedule |
| TensorBoard only | TensorBoard + WandB |
| No fast_run | fast_run flag |

## Migration Steps

If you have custom code based on the old MNIST VAE:

### 1. Update Imports

**Old:**
```python
from midit.model import VAE
from midit.dataloader import MNISTDataModule
```

**New:**
```python
from midit.model import MaskedDiffusionTransformer
from midit.midicaps_dataloader import MidiCapsDataModule
```

### 2. Update Model Initialization

**Old:**
```python
model = VAE(
    latent_dim=cfg.data.latent_dim,
    lr=cfg.optim.lr
)
```

**New:**
```python
model = MaskedDiffusionTransformer(
    midi_vocab_size=cfg.data.midi_vocab_size,
    text_vocab_size=cfg.data.text_vocab_size,
    d_model=cfg.model.d_model,
    nhead=cfg.model.nhead,
    num_encoder_layers=cfg.model.num_encoder_layers,
    num_decoder_layers=cfg.model.num_decoder_layers,
    dim_feedforward=cfg.model.dim_feedforward,
    dropout=cfg.model.dropout,
    max_seq_length=cfg.model.max_seq_length,
    timesteps=cfg.diffusion.timesteps,
    lr=cfg.optim.lr,
    beta_schedule=cfg.diffusion.beta_schedule,
)
```

### 3. Update Data Module

**Old:**
```python
data_module = MNISTDataModule(
    data_dir=cfg.data.data_dir,
    batch_size=cfg.train.batch_size,
    num_workers=cfg.data.num_workers,
)
```

**New:**
```python
data_module = MidiCapsDataModule(
    data_dir=cfg.data.data_dir,
    batch_size=cfg.train.batch_size,
    num_workers=cfg.data.num_workers,
    max_midi_length=cfg.data.max_midi_length,
    max_text_length=cfg.data.max_text_length,
    midi_vocab_size=cfg.data.midi_vocab_size,
    text_vocab_size=cfg.data.text_vocab_size,
    fast_run=cfg.train.fast_run,
)
```

### 4. Update Configuration Files

Remove MNIST-specific parameters:
- `data.latent_dim`

Add new parameters:
- `model.*` (all model architecture params)
- `diffusion.*` (diffusion-specific params)
- `logging.*` (logger configuration)
- `data.max_midi_length`, `data.max_text_length`
- `data.midi_vocab_size`, `data.text_vocab_size`
- `train.fast_run`

### 5. Update Training Loop

The training loop is now handled by Lightning, but if you have custom training code:

**Old batch format:**
```python
x, _ = batch  # Images and labels
```

**New batch format:**
```python
midi_tokens = batch["midi_tokens"]
text_tokens = batch["text_tokens"]
text_attention_mask = batch["text_attention_mask"]
midi_attention_mask = batch["midi_attention_mask"]
```

## Backwards Compatibility

The old MNIST VAE code is still available in backup files:
- `midit/model_vae_old.py.bak`
- `midit/dataloader_mnist_old.py.bak`

These files are excluded from git via `.gitignore` but remain in your working directory for reference.

## Testing the New Implementation

### Quick Test
```bash
uv run train train.fast_run=true logging.use_wandb=false train.accelerator=cpu
```

### Full Training
```bash
uv run train
```

### With Custom Config
```bash
uv run train model.d_model=256 diffusion.timesteps=500
```

## Common Issues

### Issue: Module not found errors
**Solution:** Make sure imports are updated to new package structure

### Issue: Config errors
**Solution:** Update config.yaml with new parameters or use command-line overrides

### Issue: Data loading errors
**Solution:** The new dataloader uses synthetic data by default. To use real MidiCaps:
1. Install: `uv pip install datasets`
2. Update `midicaps_dataloader.py`

### Issue: Model too large for memory
**Solution:** Reduce model size:
```bash
uv run train model.d_model=128 model.num_encoder_layers=2
```

## Need Help?

- Check `README.md` for general usage
- Check `USAGE.md` for detailed examples
- Look at the code docstrings
- The old VAE code is in `.bak` files for reference
