"""Training comparativo: Baseline vs Piano-Only"""

import os
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from midit.model.diffusion_transformer_v2 import MaskedDiffusionTransformer
from midit.midicaps_dataloader_v2_filtered import MidiCapsDataModule


def train_experiment(
    experiment_name: str,
    single_instrument_only: bool,
    data_dir: str = "data/midi_files",
    max_samples: int = 5000,
    epochs: int = 3,
    batch_size: int = 8,
):
    """Train un esperimento singolo."""
    
    print("\n" + "="*80)
    print(f"🚀 EXPERIMENT: {experiment_name}")
    print("="*80)
    print(f"   Piano-only filter: {single_instrument_only}")
    print(f"   Max samples: {max_samples:,}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print("="*80 + "\n")
    
    # Setup DataModule
    data_module = MidiCapsDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        max_midi_length=1024,
        max_text_length=256,
        fast_run=False,
        max_samples=max_samples,
        single_instrument_only=single_instrument_only,
    )
    
    # Model (piccolo per velocità)
    model = MaskedDiffusionTransformer(
        midi_vocab_size=768,
        text_vocab_size=32100,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_length=1024,
        timesteps=1000,
        lr=1e-4,
        beta_schedule="linear",
        text_encoder_model="google/flan-t5-base",
        text_encoder_freeze=True,
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir="logs/comparison",
        name=experiment_name
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"logs/comparison/{experiment_name}/checkpoints",
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
    ]
    
    # Trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # Train
    print(f"\n⏳ Starting training...")
    trainer.fit(model, data_module)
    
    # Results
    best_val_loss = trainer.callback_metrics.get("val_loss", None)
    
    print(f"\n✅ Training complete!")
    print(f"   Best val_loss: {best_val_loss:.4f}")
    
    return {
        'experiment': experiment_name,
        'piano_only': single_instrument_only,
        'best_val_loss': float(best_val_loss) if best_val_loss else None,
        'checkpoint': trainer.checkpoint_callback.best_model_path,
    }


def main():
    """Run comparison experiments."""
    
    print("\n" + "🎯"*40)
    print("COMPARISON: Baseline vs Piano-Only")
    print("🎯"*40 + "\n")
    
    CONFIG = {
        'data_dir': 'data/midi_files',
        'max_samples': 5000,
        'epochs': 3,
        'batch_size': 8,
    }
    
    results = []
    
    # Experiment 1: BASELINE
    print("\n" + "📊"*40)
    print("[1/2] BASELINE - All instruments")
    print("📊"*40 + "\n")
    
    try:
        result_baseline = train_experiment(
            experiment_name="baseline_all_instruments",
            single_instrument_only=False,
            **CONFIG
        )
        results.append(result_baseline)
    except Exception as e:
        print(f"❌ Baseline failed: {e}")
        results.append({'experiment': 'baseline', 'error': str(e)})
    
    # Experiment 2: PIANO ONLY
    print("\n" + "📊"*40)
    print("[2/2] PIANO ONLY")
    print("📊"*40 + "\n")
    
    try:
        result_piano = train_experiment(
            experiment_name="piano_only",
            single_instrument_only=True,
            **CONFIG
        )
        results.append(result_piano)
    except Exception as e:
        print(f"❌ Piano-only failed: {e}")
        results.append({'experiment': 'piano_only', 'error': str(e)})
    
    # Print comparison
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\n{result['experiment']}:")
        if 'best_val_loss' in result and result['best_val_loss']:
            print(f"   Val loss: {result['best_val_loss']:.4f}")
            print(f"   Checkpoint: {result.get('checkpoint', 'N/A')}")
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
    
    # Save results
    import json
    os.makedirs("logs/comparison", exist_ok=True)
    with open("logs/comparison/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Results saved to: logs/comparison/results.json")
    print("\n✅ Done!")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()