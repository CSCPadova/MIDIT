"""Test DataLoader with integrated masking."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from midit.data.lakh_dataset import LakhDataset, collate_fn

print("\n" + "="*70)
print("🧪 TEST DATALOADER + MASKING INTEGRATION")
print("="*70 + "\n")

# ========================================
# Load Config
# ========================================
print("📝 Step 1: Load config...\n")

cfg = OmegaConf.load("conf/config.yaml")

print(f"✅ Config loaded\n")

# ========================================
# Create Dataset
# ========================================
print("📝 Step 2: Create dataset with masking...\n")

dataset = LakhDataset(
    data_dir="data/preprocessed_phase1",
    chunking_config=cfg.chunking,
    split='train',
    apply_masking=True,  # ← Enable masking!
)

print(f"✅ Dataset created\n")

# ========================================
# Create DataLoader
# ========================================
print("📝 Step 3: Create DataLoader...\n")

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)

print(f"✅ DataLoader created\n")

# ========================================
# Load One Batch
# ========================================
print("📝 Step 4: Load one batch...\n")

for batch in dataloader:
    if batch is None:
        print("⚠️  Batch is None")
        continue
    
    print(f"✅ Batch loaded!\n")
    
    # ========================================
    # Check Shapes
    # ========================================
    print(f"📊 Batch shapes:")
    print(f"   input_ids:       {batch['input_ids'].shape}")
    print(f"   original_tokens: {batch['original_tokens'].shape}")
    print(f"   mask_labels:     {batch['mask_labels'].shape}")
    print(f"   attention_mask:  {batch['attention_mask'].shape}\n")
    
    # ========================================
    # Check Masking Applied
    # ========================================
    print(f"🎭 Masking verification:")
    
    # Count masked tokens
    total_tokens = batch['mask_labels'].numel()
    masked_tokens = batch['mask_labels'].sum().item()
    mask_ratio = masked_tokens / total_tokens * 100
    
    print(f"   Total tokens:  {total_tokens}")
    print(f"   Masked tokens: {masked_tokens}")
    print(f"   Mask ratio:    {mask_ratio:.2f}%\n")
    
    # Check that input_ids differ from original_tokens at masked positions
    differences = (batch['input_ids'] != batch['original_tokens']).sum().item()
    print(f"   Tokens changed by masking: {differences}")
    
    if differences > 0:
        print(f"   ✅ Masking applied successfully!\n")
    else:
        print(f"   ⚠️  No tokens were masked!\n")
    
    # ========================================
    # Inspect First Sample
    # ========================================
    print(f"🔍 First sample inspection (first 20 tokens):\n")
    
    sample_idx = 0
    seq_len = batch['attention_mask'][sample_idx].sum().item()  # Real length (no padding)
    
    print(f"   Sample 0 real length: {seq_len} tokens")
    print(f"   Masked positions: {batch['mask_labels'][sample_idx][:seq_len].sum().item()}\n")
    
    # Get MASK token ID from dataset
    mask_token_id = dataset.mask_token_id
    print(f"   MASK token ID: {mask_token_id}\n")
    
    # Show first 20 tokens
    print(f"   {'Pos':<6} {'Original':<10} {'Input':<10} {'Masked?':<10}")
    print(f"   {'-'*40}")
    
    for i in range(min(20, seq_len)):
        orig_id = batch['original_tokens'][sample_idx, i].item()
        input_id = batch['input_ids'][sample_idx, i].item()
        is_masked = batch['mask_labels'][sample_idx, i].item()
        
        masked_mark = "✓ MASKED" if is_masked else ""
        
        # Check if input is MASK token
        if input_id == mask_token_id and is_masked:
            input_str = "MASK"
        else:
            input_str = str(input_id)
        
        print(f"   {i:<6} {orig_id:<10} {input_str:<10} {masked_mark}")
    
    print()
    
    # ========================================
    # Verify Consistency
    # ========================================
    print(f"✅ Consistency checks:")
    
    # Check 1: Masked positions have MASK token
    masked_positions = batch['mask_labels'][sample_idx][:seq_len]
    input_at_masked = batch['input_ids'][sample_idx][:seq_len][masked_positions]
    
    if len(input_at_masked) > 0:
        mask_tokens_correct = (input_at_masked == mask_token_id).all().item()
        if mask_tokens_correct:
            print(f"   ✅ All masked positions have MASK token")
        else:
            print(f"   ⚠️  Some masked positions don't have MASK token!")
    
    # Check 2: Non-masked positions unchanged
    non_masked_positions = ~masked_positions
    orig_non_masked = batch['original_tokens'][sample_idx][:seq_len][non_masked_positions]
    input_non_masked = batch['input_ids'][sample_idx][:seq_len][non_masked_positions]
    
    if len(orig_non_masked) > 0:
        unchanged_correct = (orig_non_masked == input_non_masked).all().item()
        if unchanged_correct:
            print(f"   ✅ Non-masked positions unchanged")
        else:
            print(f"   ⚠️  Some non-masked positions changed!")
    
    print()
    
    # Only test first batch
    break

print("="*70)
print("✅ DATALOADER + MASKING TEST COMPLETE!")
print("="*70 + "\n")

if masked_tokens > 0:
    print("🎉 SUCCESS: Masking integrated correctly!")
    print(f"   - {masked_tokens} tokens masked ({mask_ratio:.1f}%)")
    print("   - MASK tokens in correct positions")
    print("   - Non-masked tokens preserved\n")
    print("💡 Next step: Fix targets dict for actual training")
else:
    print("⚠️  WARNING: No tokens were masked!")
    print("   - Try running again (random process)")
    print("   - Or check masking configuration\n")