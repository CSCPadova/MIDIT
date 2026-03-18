# debug_dataset.py
import torch
from omegaconf import OmegaConf
from midit.data.lakh_dataset import LakhDataset, collate_fn
from torch.utils.data import DataLoader

# Load config
cfg = OmegaConf.load('conf/config.yaml')

# Create dataset
print("🔍 Creating dataset...")
ds = LakhDataset(
    data_dir="data/preprocessed_phase1",
    chunking_config=cfg.chunking,
    split='train',
    apply_masking=True,
)

print(f"✅ Dataset loaded: {len(ds)} files\n")

# Test single sample
print("="*70)
print("TEST 1: SINGLE SAMPLE")
print("="*70)
sample = ds[0]

print(f"\n📊 Sample structure:")
print(f"   input_ids shape: {sample['input_ids'].shape}")
print(f"   original_tokens shape: {sample['original_tokens'].shape}")
print(f"   mask_labels shape: {sample['mask_labels'].shape}")

# Check masking
total_tokens = len(sample['original_tokens'])
masked_count = sample['mask_labels'].sum().item()
mask_ratio = masked_count / total_tokens * 100

print(f"\n🎭 Masking statistics:")
print(f"   Total tokens: {total_tokens}")
print(f"   Masked tokens: {masked_count}")
print(f"   Mask ratio: {mask_ratio:.2f}%")
print(f"   Expected: ~10-15% with bar-level masking")

if mask_ratio < 5:
    print(f"   ⚠️ WARNING: Mask ratio too low!")
elif mask_ratio > 20:
    print(f"   ⚠️ WARNING: Mask ratio too high!")
else:
    print(f"   ✅ Mask ratio OK")

# Check targets
print(f"\n🎯 Targets per head:")
for head, targets in sample['targets'].items():
    valid_targets = (targets != -100).sum().item()
    valid_ratio = valid_targets / len(targets) * 100
    print(f"   {head:10s}: {valid_targets:4d} valid ({valid_ratio:5.2f}%)")
    
    if valid_targets == 0:
        print(f"      ❌ ERROR: No valid targets for {head}!")

# Check if masked positions have targets
masked_positions = sample['mask_labels'].nonzero(as_tuple=True)[0]
print(f"\n🔗 Targets at masked positions:")
for head, targets in sample['targets'].items():
    valid_at_masked = (targets[masked_positions] != -100).sum().item()
    print(f"   {head:10s}: {valid_at_masked}/{len(masked_positions)} targets at masked positions")

# Test batch
print("\n" + "="*70)
print("TEST 2: BATCH")
print("="*70)

dataloader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)
batch = next(iter(dataloader))

print(f"\n📦 Batch structure:")
print(f"   input_ids: {batch['input_ids'].shape}")
print(f"   mask_labels: {batch['mask_labels'].shape}")
print(f"   attention_mask: {batch['attention_mask'].shape}")

print(f"\n🎭 Masking per sample:")
for i in range(len(batch['input_ids'])):
    masked = batch['mask_labels'][i].sum().item()
    total = batch['attention_mask'][i].sum().item()
    ratio = masked / total * 100 if total > 0 else 0
    print(f"   Sample {i}: {masked}/{total} masked ({ratio:.2f}%)")

print(f"\n🎯 Valid targets per head (total across batch):")
for head, targets in batch['targets'].items():
    valid = (targets != -100).sum().item()
    total = targets.numel()
    ratio = valid / total * 100
    print(f"   {head:10s}: {valid:5d}/{total} ({ratio:5.2f}%)")

# Check input vs original
print(f"\n🔍 Token replacement check (sample 0, first 20 masked):")
masked_pos = batch['mask_labels'][0].nonzero(as_tuple=True)[0][:20]
if len(masked_pos) > 0:
    for pos in masked_pos[:5]:
        inp = batch['input_ids'][0, pos].item()
        orig = batch['original_tokens'][0, pos].item()
        print(f"   Position {pos:4d}: input={inp:3d}, original={orig:3d}, replaced={'✅' if inp != orig else '❌'}")

print("\n" + "="*70)
print("✅ DEBUG COMPLETE")
print("="*70)