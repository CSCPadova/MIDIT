import os
import hydra
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from midit.model import MaskedDiffusionTransformer
from midit.midicaps_dataloader import MidiCapsDataModule
from conf.config import Config


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: Config) -> int:

    # makes torch happy, remove this line if you know your GPU
    torch.set_float32_matmul_precision("high")

    # Setup loggers
    loggers = []

    if cfg.logging.use_tensorboard:
        tb_logger = TensorBoardLogger(cfg.data.logs_dir, name="midit")
        loggers.append(tb_logger)

    # Disable wandb for fast_run to avoid protobuf issues
    if cfg.logging.use_wandb and not cfg.train.fast_run:
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project or "midit",
            entity=cfg.logging.wandb_entity,
            name="midit_diffusion_transformer",
            save_dir=cfg.data.logs_dir,
        )
        loggers.append(wandb_logger)

    # Log hyperparameters to all loggers
    hparams = {
        "d_model": cfg.model.d_model,
        "nhead": cfg.model.nhead,
        "num_encoder_layers": cfg.model.num_encoder_layers,
        "num_decoder_layers": cfg.model.num_decoder_layers,
        "batch_size": cfg.train.batch_size,
        "lr": cfg.optim.lr,
        "epochs": cfg.train.epochs,
        "accelerator": cfg.train.accelerator,
        "timesteps": cfg.diffusion.timesteps,
        "beta_schedule": cfg.diffusion.beta_schedule,
        "text_encoder_model": cfg.text_encoder.model_name,
        "text_encoder_freeze": cfg.text_encoder.freeze,
        "fast_run": cfg.train.fast_run,
    }

    for logger in loggers:
        if hasattr(logger, "log_hyperparams"):
            logger.log_hyperparams(hparams)

    # Model
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
        text_encoder_model=cfg.text_encoder.model_name,
        text_encoder_freeze=cfg.text_encoder.freeze,
    )

    # Data module
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

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.data.logs_dir, "checkpoints"),
            filename="midit-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        ),
    ]

    # Determine max_epochs and steps for fast_run
    max_epochs = cfg.train.epochs
    limit_train_batches = 1.0
    limit_val_batches = 1.0

    if cfg.train.fast_run:
        max_epochs = 2
        limit_train_batches = 10  # Only 10 batches per epoch
        limit_val_batches = 5  # Only 5 validation batches

    # Train model
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

    trainer.fit(model, data_module)

    # Only validate if not in fast_run mode
    if not cfg.train.fast_run:
        trainer.validate(model, data_module)

    return os.EX_OK


if __name__ == "__main__":
    train()
