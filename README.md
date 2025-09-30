# MIDIT

A Masked Diffusion Transformer for text-to-MIDI generation

## Overview

MIDIT is a deep learning model that generates MIDI music from text descriptions. It uses a Masked Diffusion Transformer architecture, combining the power of diffusion models with transformer-based sequence modeling and text conditioning.

### Key Features

- **Masked Diffusion Transformer**: State-of-the-art architecture for sequence generation
- **Text-to-MIDI Generation**: Generate music from natural language descriptions
- **Flexible Configuration**: All hyperparameters configurable via Hydra
- **Multiple Logging Options**: Support for both WandB and TensorBoard
- **Fast Run Mode**: Quick testing with reduced dataset and epochs

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- GPU recommended for training

## Installation

To install the required dependencies, run the following command:

```bash
make install
```

Or you can install the dependencies manually using uv:

```bash
uv sync
```

## Training

### Basic Training

To start training with default configuration:

```bash
uv run train
```

### Configuration Options

The project uses Hydra for configuration management. All settings can be modified in `conf/config.yaml` or overridden via command line.

#### Model Configuration

- `model.d_model`: Model dimension (default: 512)
- `model.nhead`: Number of attention heads (default: 8)
- `model.num_encoder_layers`: Number of encoder layers (default: 6)
- `model.num_decoder_layers`: Number of decoder layers (default: 6)
- `model.dim_feedforward`: Feedforward dimension (default: 2048)
- `model.dropout`: Dropout rate (default: 0.1)

#### Diffusion Configuration

- `diffusion.timesteps`: Number of diffusion timesteps (default: 1000)
- `diffusion.beta_schedule`: Noise schedule type ('linear' or 'cosine')

#### Training Configuration

- `train.batch_size`: Training batch size (default: 8)
- `train.epochs`: Number of training epochs (default: 10)
- `train.accelerator`: Device to use ('gpu' or 'cpu')
- `train.fast_run`: Enable fast run mode for testing (default: false)

#### Logging Configuration

- `logging.use_wandb`: Enable Weights & Biases logging (default: true)
- `logging.use_tensorboard`: Enable TensorBoard logging (default: true)
- `logging.wandb_project`: WandB project name (default: 'midit')
- `logging.wandb_entity`: WandB entity/username (optional)

### Command Line Overrides

You can override any configuration parameter from the command line:

```bash
# Train with smaller model
uv run train model.d_model=256 model.nhead=4

# Train with larger batch size
uv run train train.batch_size=16

# Disable WandB logging
uv run train logging.use_wandb=false

# Enable fast run mode for quick testing
uv run train train.fast_run=true

# Change diffusion settings
uv run train diffusion.timesteps=500 diffusion.beta_schedule=cosine
```

### Fast Run Mode

For quick testing of the training pipeline:

```bash
uv run train train.fast_run=true
```

Fast run mode:
- Uses a small synthetic dataset (100 samples)
- Trains for only 2 epochs
- Limits batches per epoch (10 train, 5 validation)
- Perfect for testing changes without waiting for full training

### SLURM Cluster

If you are running the project on a SLURM cluster:

```bash
make train
```

## Project Structure

```
midit/
├── conf/                      # Configuration files
│   ├── config.yaml           # Main configuration
│   └── config.py             # Configuration dataclasses
├── midit/                     # Main package
│   ├── model/                # Model implementations
│   │   ├── diffusion_transformer.py
│   │   └── text_encoder.py
│   ├── loss/                 # Loss functions
│   │   └── diffusion_loss.py
│   ├── midicaps_dataloader.py  # MidiCaps dataset loader
│   └── train.py              # Training script
└── logs/                      # Training logs and checkpoints
```

## Dataset

The model is designed to work with the [MidiCaps dataset](https://huggingface.co/datasets/amaai-lab/MidiCaps) from HuggingFace, which provides paired text descriptions and MIDI files.

**Note**: The current implementation uses synthetic data for development. To use the real MidiCaps dataset, you'll need to:

1. Install the `datasets` library: `uv pip install datasets`
2. Update the dataloader in `midit/midicaps_dataloader.py` to load from HuggingFace

## Logging and Monitoring

### TensorBoard

To view TensorBoard logs:

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

### Weights & Biases

If WandB is enabled, logs will automatically be sent to your WandB account. Make sure to:

1. Install wandb: Already included in dependencies
2. Login: `wandb login`
3. Configure project name in `conf/config.yaml`

You can disable WandB logging:

```bash
uv run train logging.use_wandb=false
```

## Model Architecture

MIDIT uses a Masked Diffusion Transformer architecture:

1. **Text Encoder**: Encodes text descriptions using a transformer encoder
2. **Diffusion Process**: Gradually denoises MIDI sequences over multiple timesteps
3. **Transformer Decoder**: Predicts the denoised MIDI tokens conditioned on text
4. **Masking**: Supports variable-length sequences with attention masking

## Aknowledgements

This project is based on the following template repository:

```bibtex
@software{M_Spanio_DEI_Blade_Template,
    author = {M Spanio, Matteo},
    license = {MIT},
    title = {{DEI Blade Template}},
    url = {https://github.com/CSCPadova/dei-blade-template}
}
```

The following libraries were used in this project:

- [PyTorch](https://pytorch.org/) - Deep Learning framework
- [Torchaudio](https://pytorch.org/audio/) - Audio processing library for PyTorch
- [Lightning](https://www.pytorchlightning.ai/) - High-level interface for PyTorch
- [WandB](https://wandb.ai/) - Experiment tracking and visualization
- [Hydra](https://hydra.cc/) - Configuration management

## License

This project template is licensed under the MIT License - see the [LICENSE](LICENSE) file for details, please note that the generated project will inherit this license.
