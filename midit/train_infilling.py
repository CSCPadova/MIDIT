"""Phase 1b Infilling Fine-tuning — bar-fill masking on a pre-trained checkpoint.

Loads a Phase 1 checkpoint and fine-tunes with BarFillMasking.
Use conf/config_infilling.yaml (loaded via Hydra config_name override).

Usage:
    python midit/train_infilling.py --config-name=config_infilling \
        chunking.preprocessed_dir=/path/to/preprocessed_v2_50k
"""

import os
import hydra
import torch
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig

from midit.data.lakh_dataset import LakhDataset, collate_fn
from midit.model.masked_transformer import MaskedTransformerREMI


class DataLoaderWrapper:
    """Adapt DataLoader batch keys to MaskedTransformerREMI expectations.

    Identical to the wrapper in train_phase1.py — kept here to avoid a
    shared-state dependency between the two training scripts.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            if batch is None:
                continue
            yield {
                'midi_tokens':         batch['input_ids'],
                'targets':             batch['targets'],
                'midi_attention_mask': batch['attention_mask'],
                'bar_positions':       batch.get('bar_positions', None),
                'mask_labels':         batch.get('mask_labels', None),
            }

    def __len__(self):
        return len(self.dataloader)


@hydra.main(version_base=None, config_path="../conf", config_name="config_infilling")
def train_infilling(cfg: DictConfig) -> None:
    """Fine-tune a pre-trained MaskedTransformerREMI checkpoint with bar-fill masking."""

    print("\n" + "="*70)
    print("PHASE 1b INFILLING FINE-TUNING: MaskedTransformerREMI")
    print("="*70 + "\n")

    torch.set_float32_matmul_precision("high")

    # =================================
    # Checkpoint path
    # =================================
    checkpoint_path = getattr(cfg.train, 'checkpoint_path', None)
    if not checkpoint_path:
        raise ValueError(
            "cfg.train.checkpoint_path is not set. "
            "Set it in config_infilling.yaml or pass it on the CLI:\n"
            "  train.checkpoint_path=/path/to/phase1.ckpt"
        )
    print(f"[INFO] Loading checkpoint: {checkpoint_path}\n")

    # =================================
    # Dataset & DataLoader
    # cfg.masking.strategy = 'bar_fill' (from config_infilling.yaml)
    # LakhDataset routes to BarFillMasking automatically.
    # =================================
    print("Loading datasets...\n")

    train_dataset = LakhDataset(cfg, split='train', apply_masking=True)
    val_dataset   = LakhDataset(cfg, split='val',   apply_masking=True)

    batch_size = getattr(cfg.train, 'batch_size', 2)

    train_loader_base = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader_base = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    train_loader = DataLoaderWrapper(train_loader_base)
    val_loader   = DataLoaderWrapper(val_loader_base)

    print(f"Train dataset: {len(train_dataset)} files")
    print(f"Val dataset:   {len(val_dataset)} files")
    print(f"Batch size:    {batch_size}\n")

    # =================================
    # Compute total training steps
    # (DataLoaderWrapper causes trainer.estimated_stepping_batches to return
    #  inf, so we compute manually — same logic as train_phase1.py)
    # =================================
    max_epochs  = getattr(cfg.train, 'epochs', 10)
    accumulate  = getattr(cfg.train, 'accumulate_grad_batches', 4)
    limit       = getattr(cfg.train, 'limit_train_batches', 1.0)

    if isinstance(limit, float):
        effective_batches = int(len(train_loader_base) * limit)
    else:
        effective_batches = min(int(limit), len(train_loader_base))

    steps_per_epoch      = effective_batches // accumulate
    total_training_steps = steps_per_epoch * max_epochs

    lr           = getattr(cfg.train, 'lr', 3e-5)
    warmup_steps = getattr(cfg.train, 'warmup_steps', 200)

    print(f"total_training_steps: {total_training_steps} "
          f"({effective_batches} batches // {accumulate} accumulate "
          f"= {steps_per_epoch} steps/epoch x {max_epochs} epochs)")
    print(f"lr:           {lr}")
    print(f"warmup_steps: {warmup_steps}\n")

    # =================================
    # Load checkpoint and override hparams
    # Lightning bakes lr / warmup_steps / num_training_steps into the
    # checkpoint's hparams. Override them after loading so the scheduler
    # uses the fine-tuning values rather than the pre-training ones.
    # =================================
    model = MaskedTransformerREMI.load_from_checkpoint(
        checkpoint_path,
        map_location='cpu',
    )

    model.lr                 = lr
    model.warmup_steps       = warmup_steps
    model.num_training_steps = total_training_steps

    print(f"[OK] Checkpoint loaded and hparams overridden:")
    print(f"   lr:                  {model.lr}")
    print(f"   warmup_steps:        {model.warmup_steps}")
    print(f"   num_training_steps:  {model.num_training_steps}\n")

    # =================================
    # Logger & Callbacks
    # =================================
    logs_dir = (
        getattr(getattr(cfg, 'logging', None), 'logs_dir', None)
        or getattr(getattr(cfg, 'data', None), 'logs_dir', None)
        or 'logs'
    )

    checkpoints_dir = os.path.join(logs_dir, "checkpoints_infilling")

    print(f"Logs directory:  {logs_dir}")
    print(f"Checkpoints dir: {checkpoints_dir}\n")

    tb_logger = TensorBoardLogger(
        save_dir=logs_dir,
        name="infilling_finetuning",
    )

    loggers = [tb_logger]
    use_wandb = getattr(getattr(cfg, 'logging', None), 'use_wandb', False)
    if use_wandb:
        wandb_logger = WandbLogger(
            project='MIDIT-Infilling',
            save_dir=logs_dir,
        )
        loggers.append(wandb_logger)
        print(f"[INFO] WandbLogger enabled: project=MIDIT-Infilling\n")

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename="infilling-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=getattr(cfg.train, 'save_top_k', 3),
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # =================================
    # Trainer
    # =================================
    accelerator = getattr(cfg.train, 'accelerator', 'gpu')
    fast_run    = getattr(cfg.train, 'fast_run', False)

    trainer = L.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=cfg.train.precision if accelerator == "gpu" else "32",
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=accumulate,
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=cfg.train.val_check_interval,
        limit_train_batches=100 if fast_run else cfg.train.limit_train_batches,
        limit_val_batches=20 if fast_run else getattr(cfg.train, 'limit_val_batches', 1.0),
    )

    # =================================
    # Train
    # =================================
    print("Starting fine-tuning...\n")
    print("="*70 + "\n")

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE!")
    print("="*70 + "\n")

    best_model_path = callbacks[0].best_model_path
    if best_model_path:
        print(f"Best checkpoint: {best_model_path}\n")


if __name__ == "__main__":
    train_infilling()
