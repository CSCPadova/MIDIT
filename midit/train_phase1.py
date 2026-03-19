"""Phase 1 Pre-training using existing MaskedTransformerREMI architecture."""

import os
import hydra
import torch 
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from midit.data.lakh_dataset import LakhDataset, collate_fn
from midit.model.masked_transformer import MaskedTransformerREMI


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
    
    # ========================================
    # Dataset & DataLoader
    # ========================================
    print("📦 Loading datasets...\n")
    
    train_dataset = LakhDataset(cfg, split='train', apply_masking=True)

    val_dataset = LakhDataset(cfg, split='val', apply_masking=True)
    
    batch_size = getattr(cfg.train, 'batch_size', 8)
    
    train_loader_base = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader_base = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    
    # Wrap loaders to adapt batch format
    train_loader = DataLoaderWrapper(train_loader_base)
    val_loader = DataLoaderWrapper(val_loader_base)
    
    print(f"✅ Train dataset: {len(train_dataset)} files")
    print(f"✅ Val dataset: {len(val_dataset)} files")
    print(f"✅ Batch size: {batch_size}\n")
    
    # ========================================
    # Model Config
    # ========================================
    print("🤖 Initializing MaskedTransformerREMI...\n")
    
    d_model = getattr(cfg.model, 'd_model', 512)
    nhead = getattr(cfg.model, 'nhead', 8)
    num_layers = getattr(cfg.model, 'num_encoder_layers', 12)
    dim_feedforward = getattr(cfg.model, 'dim_feedforward', 2048)
    dropout = getattr(cfg.model, 'dropout', 0.1)
    max_seq_length = getattr(cfg.chunking, 'max_tokens', 8192)
    
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
    
    # ========================================
    # Model - Use Existing Architecture!
    # ========================================
    
    # Vocab sizes from our analysis
    # ========================================
    # Get actual vocab sizes from dataset
    # ========================================
    print("📊 Retrieving actual vocab sizes from dataset...\n")

    # Get from train dataset (already initialized)
    vocab_sizes = train_dataset.actual_vocab_sizes.copy()

    print(f"✅ Using actual vocab sizes from tokenizer:")
    for token_type, size in vocab_sizes.items():
        print(f"   {token_type:10s}: {size:>4d}")
    print()

    # ========================================
    # Get tokenizer vocab size (for input embedding)
    # ========================================
    tokenizer_vocab_size = len(train_dataset.tokenizer.vocab)

    print(f"📊 Tokenizer info:")
    print(f"   Total vocab size: {tokenizer_vocab_size}")
    print(f"   (Input embedding will use this size)")
    print(f"   (Output heads use per-type sizes above)\n")

    special_tokens = {
        'pad': 0,
        'bos': 1,
        'eos': 2,
        'mask': 3,
    }

    # Verify no zero sizes
    for token_type, size in vocab_sizes.items():
        if size == 0:
            raise ValueError(f"Vocab size for {token_type} is 0! Check tokenizer config.")
    
    # Lambda weights (from config)
    from omegaconf import OmegaConf
    lambda_weights = OmegaConf.to_container(cfg.lambda_weights, resolve=True)
    
    print(f"\n{'='*70}")
    print(f"🔍 LAMBDA WEIGHTS FROM CONFIG:")
    for k, v in lambda_weights.items():
        print(f"   {k:10s}: {v}")
    print(f"{'='*70}\n")
    #lambda_weights = {
    #    'pitch': 1.0,
    #    'velocity': 1.0,
    #    'duration': 1.0,
    #    'program': 1.0,
    #    'position': 4.5,
    #    'rest': 10.0,
    #    'bar': 10.0,
    #    'timesig': 10.0,
    #    'tempo': 20.0,
    #}
    
    model = MaskedTransformerREMI(
        phase=1,
        vocab_sizes=vocab_sizes,  # Per output heads
        special_token_ids=special_tokens,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_length=max_seq_length,
        mask_prob=0.15,
        bar_level_masking=True,
        lambda_weights=lambda_weights,
        class_weights=OmegaConf.to_container(cfg.get('class_weights', None), resolve=True) if 'class_weights' in cfg else None,
        lr=lr,
        weight_decay=0.01,
        warmup_steps=1000,
    )

    # CRITICAL FIX: Override token embedding with correct size
    print(f"🔧 Fixing token embedding size to match tokenizer...\n")
    model.token_embedding = torch.nn.Embedding(tokenizer_vocab_size, d_model)
    print(f"✅ Token embedding size: {tokenizer_vocab_size} → {d_model}\n")
    
    # ========================================
    # Logger & Callbacks
    # ========================================
    if hasattr(cfg, 'data') and hasattr(cfg.data, 'logs_dir'):
        logs_dir = cfg.data.logs_dir
    elif hasattr(cfg, 'logging') and hasattr(cfg.logging, 'logs_dir'):
        logs_dir = cfg.logging.logs_dir
    else:
        logs_dir = 'logs'  # Default

    print(f"📁 Logs directory: {logs_dir}\n")
    
    logger = TensorBoardLogger(
        save_dir=logs_dir,
        name="phase1_pretraining",
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(logs_dir, "checkpoints_phase1"),
            filename="phase1-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    # ========================================
    # Trainer
    # ========================================
    max_epochs = getattr(cfg.train, 'epochs', 10)
    accelerator = getattr(cfg.train, 'accelerator', 'cpu')
    fast_run = getattr(cfg.train, 'fast_run', False)
    
    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if accelerator == "gpu" else "32",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.5,
        limit_train_batches=100 if fast_run else 1.0,
        limit_val_batches=20 if fast_run else 1.0,
    )
    
    # ========================================
    # Train
    # ========================================
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
