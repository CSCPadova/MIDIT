"""Training script for REMI+ tokenization with Piano-only filtering."""

import os
import hydra
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from midit.midicaps_dataloader_remi import MidiCapsDataModule
from midit.model.diffusion_transformer_v2_remi import MaskedDiffusionTransformerREMI
from conf.config import Config


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: Config) -> int:
    
    print("\n" + "="*80)
    print("🎹 TRAINING REMI+ (Piano-only)")
    print("="*80)

    torch.set_float32_matmul_precision("high")

    # Setup loggers
    loggers = []

    if cfg.logging.use_tensorboard:
        tb_logger = TensorBoardLogger(cfg.data.logs_dir, name="midit_remi")
        loggers.append(tb_logger)

    if cfg.logging.use_wandb and not cfg.train.fast_run:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project or "midit",
            entity=cfg.logging.wandb_entity,
            name="midit_remi_piano",
            save_dir=cfg.data.logs_dir,
        )
        loggers.append(wandb_logger)

    # Fast run adjustments
    if cfg.train.fast_run:
        print("\n🏃 FAST RUN MODE - Small model for testing")
        cfg.model.d_model = 256
        cfg.model.nhead = 4
        cfg.model.num_encoder_layers = 4
        cfg.model.num_decoder_layers = 4
        cfg.model.dim_feedforward = 1024
        cfg.train.batch_size = 4
        cfg.train.accumulate_grad_batches = 2

    # Hyperparameters
    hparams = {
        "tokenization": "REMI+",
        "vocab_size": 389,
        "filtering": "Piano-only",
        "d_model": cfg.model.d_model,
        "nhead": cfg.model.nhead,
        "num_encoder_layers": cfg.model.num_encoder_layers,
        "num_decoder_layers": cfg.model.num_decoder_layers,
        "batch_size": cfg.train.batch_size,
        "lr": cfg.optim.lr,
        "epochs": cfg.train.epochs,
        "timesteps": cfg.diffusion.timesteps,
        "beta_schedule": cfg.diffusion.beta_schedule,
        "fast_run": cfg.train.fast_run,
    }

    for logger in loggers:
        if hasattr(logger, "log_hyperparams"):
            logger.log_hyperparams(hparams)
    
    print(f"\n📊 Config:")
    print(f"   Vocab size: 389 (REMI+)")
    print(f"   Filtering: Piano-only")
    print(f"   Batch size: {cfg.train.batch_size}")
    print(f"   Epochs: {cfg.train.epochs}")
    print(f"   Max samples: {cfg.data.max_samples}")
    
    # Model REMI+
    model = MaskedDiffusionTransformerREMI(
        midi_vocab_size=389,  # REMI+ vocab
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
        text_encoder_model=cfg.text_encoder.model_name,
        text_encoder_freeze=cfg.text_encoder.freeze,
    )

    print(f"\n✅ Model created:")
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Data module REMI+
    data_module = MidiCapsDataModule(
        data_dir=cfg.data.data_dir + "/midi_files",
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        max_midi_length=cfg.data.max_midi_length,
        max_text_length=cfg.data.max_text_length,
        fast_run=cfg.train.fast_run,
        max_samples=cfg.data.max_samples,
        single_instrument_only=True,  # ← PIANO-ONLY!
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
    ]

    if not cfg.train.fast_run:
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(cfg.data.logs_dir, "checkpoints_remi"),
                filename="midit-remi-{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
            )
        )

    # Training limits
    max_epochs = cfg.train.epochs
    limit_train_batches = 1.0
    limit_val_batches = 1.0
    
    if cfg.train.fast_run:
        max_epochs = 2
        limit_train_batches = 10
        limit_val_batches = 5

    # Trainer
    trainer = L.Trainer(
        logger=loggers if loggers else None,
        accelerator=cfg.train.accelerator,
        default_root_dir=cfg.data.logs_dir,
        min_epochs=1,
        max_epochs=max_epochs,
        callbacks=callbacks,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )

    print("\n🚀 Starting training...\n")
    trainer.fit(model, data_module)

    if not cfg.train.fast_run:
        print("\n📊 Validating...")
        trainer.validate(model, data_module)

    print("\n" + "="*80)
    print("✅ Training completed!")
    print("="*80)

    return os.EX_OK


if __name__ == "__main__":
    train()