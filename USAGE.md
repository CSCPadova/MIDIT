# MIDIT Usage Examples

This document provides examples of how to use MIDIT for training and inference.

## Quick Start

### Basic Training

```bash
uv run train
```

This will train the model with default settings from `conf/config.yaml`.

### Fast Testing Mode

For quick testing of the training pipeline:

```bash
uv run train train.fast_run=true
```

This mode:
- Uses only 100 training samples
- Trains for 2 epochs
- Limits to 10 training batches and 5 validation batches per epoch
- Perfect for development and testing

## Configuration Examples

### Model Size Adjustments

#### Small Model (for testing)
```bash
uv run train \
    model.d_model=128 \
    model.nhead=4 \
    model.num_encoder_layers=2 \
    model.num_decoder_layers=2
```

#### Large Model (for production)
```bash
uv run train \
    model.d_model=1024 \
    model.nhead=16 \
    model.num_encoder_layers=12 \
    model.num_decoder_layers=12
```

### Diffusion Settings

#### Use cosine schedule with more steps
```bash
uv run train \
    diffusion.timesteps=1000 \
    diffusion.beta_schedule=cosine
```

#### Faster diffusion with fewer steps
```bash
uv run train \
    diffusion.timesteps=100 \
    diffusion.beta_schedule=linear
```

### Training Configuration

#### Train on CPU
```bash
uv run train train.accelerator=cpu
```

#### Train with larger batch size (requires GPU)
```bash
uv run train \
    train.batch_size=32 \
    train.accelerator=gpu
```

#### Extended training
```bash
uv run train train.epochs=50
```

### Logging Options

#### Disable WandB, keep TensorBoard
```bash
uv run train logging.use_wandb=false
```

#### Disable TensorBoard, keep WandB
```bash
uv run train logging.use_tensorboard=false
```

#### Configure WandB project
```bash
uv run train \
    logging.wandb_project=my-midi-project \
    logging.wandb_entity=my-username
```

#### Disable all logging
```bash
uv run train \
    logging.use_wandb=false \
    logging.use_tensorboard=false
```

## Combined Examples

### Development Setup
Fast iteration with small model and no WandB:
```bash
uv run train \
    train.fast_run=true \
    logging.use_wandb=false \
    model.d_model=128 \
    model.nhead=4 \
    train.accelerator=cpu
```

### Production Training
Full dataset with optimal settings:
```bash
uv run train \
    train.batch_size=32 \
    train.epochs=50 \
    train.accelerator=gpu \
    model.d_model=512 \
    model.nhead=8 \
    diffusion.timesteps=1000 \
    diffusion.beta_schedule=cosine \
    logging.wandb_project=midit-production
```

### Experiment with Architecture
Test different model configurations:
```bash
# Deeper model
uv run train \
    model.num_encoder_layers=8 \
    model.num_decoder_layers=8 \
    train.fast_run=true

# Wider model
uv run train \
    model.d_model=768 \
    model.dim_feedforward=3072 \
    train.fast_run=true
```

## SLURM Cluster Usage

If running on a SLURM cluster, use the Makefile:

```bash
make train
```

This will submit the job via `sbatch slurm/train.sh`.

## Monitoring Training

### TensorBoard

Start TensorBoard to monitor training:

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

### WandB

If WandB is enabled, logs are automatically synced to your WandB account. You can view them at:
https://wandb.ai/your-entity/your-project

Make sure to login first:
```bash
wandb login
```

## Model Checkpoints

Model checkpoints are saved in `logs/checkpoints/`. The top 3 models based on validation loss are kept.

Checkpoint naming format: `midit-{epoch:02d}-{val_loss:.2f}.ckpt`

## Data Configuration

The current implementation uses synthetic data for development. To use real MidiCaps data:

1. Install the datasets library:
   ```bash
   uv pip install datasets
   ```

2. Update `midit/midicaps_dataloader.py` to load from HuggingFace:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("amaai-lab/MidiCaps", cache_dir=self.data_dir)
   ```

3. Adjust vocabulary sizes in config if needed:
   ```bash
   uv run train \
       data.midi_vocab_size=YOUR_VOCAB_SIZE \
       data.text_vocab_size=YOUR_VOCAB_SIZE
   ```

## Hyperparameter Search

You can use Hydra's multirun feature for hyperparameter search:

```bash
uv run train -m \
    model.d_model=256,512,768 \
    model.nhead=4,8,16 \
    train.fast_run=true
```

This will run 9 experiments with all combinations of the parameters.

## Troubleshooting

### Out of Memory

If you get OOM errors:
- Reduce batch size: `train.batch_size=4`
- Reduce model size: `model.d_model=256`
- Reduce sequence lengths: `data.max_midi_length=512`

### Slow Training

- Use GPU: `train.accelerator=gpu`
- Increase batch size: `train.batch_size=64`
- Reduce model size for testing: `train.fast_run=true`

### WandB Issues

If WandB is failing:
- Login: `wandb login`
- Disable it: `logging.use_wandb=false`
- Check internet connection

## Next Steps

After training, you can:

1. Load the best checkpoint for inference
2. Fine-tune on your own MIDI data
3. Export to ONNX for deployment
4. Integrate into a music generation application
