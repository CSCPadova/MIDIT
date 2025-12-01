"""Test DataLoader with preprocessed data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from midit.data.lakh_dataset import LakhDataset, collate_fn

print("\n" + "="*70)
print("🧪 TEST DATALOADER")
print("="*70 + "\n")

# ========================================
# Load Config
# ========================================
print("📝 Step 1: Load config...\n")

cfg = OmegaConf.load("conf/config.yaml")

print(f"✅ Config loaded")
print(f"   Chunking strategy: {cfg.chunking.strategy}")
print(f"   Bars per chunk: {cfg.chunking.bars_per_chunk}")
print(f"   Max tokens: {cfg.chunking.max_tokens}\n")

# ========================================
# Create Dataset
# ========================================
print("📝 Step 2: Create dataset...\n")

dataset = LakhDataset(
    data_dir="data/preprocessed_phase1",
    chunking_config=cfg.chunking,
    split='train',
)

print(f"✅ Dataset created: {len(dataset)} files\n")

# ========================================
# Create DataLoader
# ========================================
print("📝 Step 3: Create DataLoader...\n")

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,  # 0 for testing
)

print(f"✅ DataLoader created")
print(f"   Batch size: 2\n")

# ========================================
# Test Loading One Batch
# ========================================
print("📝 Step 4: Load one batch...\n")

for batch in dataloader:
    if batch is None:
        print("⚠️  Batch is None (all chunks were None)")
        continue
    
    print(f"✅ Batch loaded successfully!\n")
    
    print(f"📊 Batch contents:")
    print(f"   midi_tokens shape: {batch['midi_tokens'].shape}")
    print(f"   Targets:")
    for key, value in batch['targets'].items():
        print(f"      {key}: {value.shape}")
    print(f"   Metadata: {len(batch['metadata'])} items\n")
    
    print(f"📋 Example metadata:")
    print(f"   File: {batch['metadata'][0]['source_file']}")
    print(f"   Chunk length: {batch['metadata'][0]['chunk_length']}")
    if 'start_bar' in batch['metadata'][0]:
        print(f"   Bars: {batch['metadata'][0]['start_bar']}-{batch['metadata'][0]['end_bar']}\n")
    
    # Only test first batch
    break

print("="*70)
print("✅ DATALOADER TEST COMPLETE!")
print("="*70 + "\n")

print("💡 Next step: Test training loop with this DataLoader")
print()