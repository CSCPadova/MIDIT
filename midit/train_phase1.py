"""Phase 1 Pre-training using existing MaskedTransformerREMI architecture."""

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
from midit.model.model_factory import build_model


class DataLoaderWrapper:
    """Wrapper to adapt DataLoader batch format to MaskedTransformerREMI expectations."""
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def __iter__(self):
        for batch in self.dataloader:
            if batch is None:
                continue
            
            # Adapt batch format:
            # input_ids → midi_tokens
            # attention_mask → midi_attention_mask
            adapted_batch = {
                'midi_tokens': batch['input_ids'],
                'targets': batch['targets'],
                'midi_attention_mask': batch['attention_mask'],
                'bar_positions': batch.get('bar_positions', None),
                'mask_labels': batch.get('mask_labels', None),
            }
            
            yield adapted_batch
    
    def __len__(self):
        return len(self.dataloader)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_phase1(cfg: DictConfig) -> None:
    """Main training function for Phase 1 using existing architecture."""
    
    print("\n" + "="*70)
    print("🎵 PHASE 1 PRE-TRAINING: MaskedTransformerREMI")
    print("="*70 + "\n")
    
    torch.set_float32_matmul_precision("high")
    
    # =================================
    # Dataset & DataLoader
    # =================================
    print("📦 Loading datasets...\n")
    
    train_dataset = LakhDataset(cfg, split='train', apply_masking=True)

    val_dataset = LakhDataset(cfg, split='val', apply_masking=True)
    
    batch_size = getattr(cfg.train, 'batch_size', 8)
    
    train_loader_base = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader_base = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Wrap loaders to adapt batch format
    train_loader = DataLoaderWrapper(train_loader_base)
    val_loader = DataLoaderWrapper(val_loader_base)
    
    print(f"✅ Train dataset: {len(train_dataset)} files")
    print(f"✅ Val dataset: {len(val_dataset)} files")
    print(f"✅ Batch size: {batch_size}\n")
    
    # =================================
    # Model Config
    # =================================
    print("🤖 Initializing MaskedTransformerREMI...\n")
    
    d_model = getattr(cfg.model, 'd_model', 512)
    nhead = getattr(cfg.model, 'nhead', 8)
    num_layers = getattr(cfg.model, 'num_encoder_layers', 12)
    dim_feedforward = getattr(cfg.model, 'dim_feedforward', 2048)
    dropout = getattr(cfg.model, 'dropout', 0.1)
    max_seq_length = getattr(cfg.chunking, 'max_tokens', 8192)

    # Compute total training steps for cosine LR decay
    # (Cannot rely on trainer.estimated_stepping_batches — returns inf with DataLoaderWrapper)
    max_epochs = getattr(cfg.train, 'epochs', 10)
    accumulate = getattr(cfg.train, 'accumulate_grad_batches', 1)
    limit = getattr(cfg.train, 'limit_train_batches', 1.0)
    if isinstance(limit, float):
        effective_batches = int(len(train_loader_base) * limit)
    else:
        effective_batches = min(int(limit), len(train_loader_base))
    steps_per_epoch      = effective_batches // accumulate
    total_training_steps = steps_per_epoch * max_epochs
    print(f"   total_training_steps: {total_training_steps} "
          f"({effective_batches} batches // {accumulate} accumulate "
          f"= {steps_per_epoch} steps/epoch x {max_epochs} epochs)\n")

    # Learning rate
    if hasattr(cfg, 'optim') and hasattr(cfg.optim, 'lr'):
        lr = cfg.optim.lr
    elif hasattr(cfg, 'optimizer') and hasattr(cfg.optimizer, 'lr'):
        lr = cfg.optimizer.lr
    elif hasattr(cfg.train, 'lr'):
        lr = cfg.train.lr
    else:
        lr = 3e-4
    
    print(f"📊 Config:")
    print(f"   d_model: {d_model}")
    print(f"   nhead: {nhead}")
    print(f"   num_layers: {num_layers}")
    print(f"   dim_feedforward: {dim_feedforward}")
    print(f"   dropout: {dropout}")
    print(f"   max_seq_length: {max_seq_length}")
    print(f"   lr: {lr}\n")
    
    # =================================
    # Model - Use Existing Architecture!
    # =================================
    
    # Vocab sizes from our analysis
    # =================================
    # Get actual vocab sizes from dataset
    # =================================
    model = build_model(cfg, train_dataset.tokenizer,
                        train_dataset.actual_vocab_sizes,
                        total_training_steps, lr)
    
    # =================================
    # Logger & Callbacks
    # =================================
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'logs_dir'):
        logs_dir = cfg.data.logs_dir
    elif hasattr(cfg, 'logging') and hasattr(cfg.logging, 'logs_dir'):
        logs_dir = cfg.logging.logs_dir
    else:
        logs_dir = 'logs'  # Default

    print(f"📁 Logs directory: {logs_dir}\n")
    
    tb_logger = TensorBoardLogger(
        save_dir=logs_dir,
        name="phase1_pretraining",
    )

    loggers = [tb_logger]
    use_wandb = getattr(getattr(cfg, 'logging', None), 'use_wandb', False)
    if use_wandb:
        wandb_project = getattr(cfg.logging, 'wandb_project', 'MIDIT-Phase1')
        wandb_logger = WandbLogger(project=wandb_project, save_dir=logs_dir)
        loggers.append(wandb_logger)
        print(f"[INFO] WandbLogger enabled: project={wandb_project}\n")

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(logs_dir, "checkpoints_phase1"),
            filename="phase1-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",


            save_top_k=cfg.train.save_top_k,

            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    # =================================
    # Trainer
    # =================================
    accelerator = getattr(cfg.train, 'accelerator', 'cpu')
    fast_run = getattr(cfg.train, 'fast_run', False)
    
    trainer = L.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=cfg.train.precision if accelerator == "gpu" else "32",
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        log_every_n_steps=cfg.train.log_every_n_steps,
        val_check_interval=cfg.train.val_check_interval,
        limit_train_batches=100 if fast_run else cfg.train.limit_train_batches,
        limit_val_batches=20 if fast_run else 1.0,
    )
    
    # =================================
    # Train
    # =================================
    print("🚀 Starting training...\n")
    print("="*70 + "\n")
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70 + "\n")
    
    best_model_path = callbacks[0].best_model_path
    if best_model_path:
        print(f"📁 Best checkpoint: {best_model_path}\n")


if __name__ == "__main__":
    train_phase1()
