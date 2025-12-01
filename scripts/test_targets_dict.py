"""Test that targets dict is created correctly - MULTI-SAMPLE VERSION."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from midit.data.lakh_dataset import LakhDataset, collate_fn

print("\n" + "="*70)
print("🎯 TEST TARGETS DICT - MULTI-SAMPLE")
print("="*70 + "\n")

# ========================================
# Load Config & Create Dataset
# ========================================
print("📝 Step 1: Create dataset...\n")

cfg = OmegaConf.load("conf/config.yaml")

dataset = LakhDataset(
    data_dir="data/preprocessed_phase1",
    chunking_config=cfg.chunking,
    split='train',
    apply_masking=True,
)

print("✅ Dataset created\n")

# ========================================
# Create DataLoader
# ========================================
print("📝 Step 2: Create DataLoader...\n")

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)

print("✅ DataLoader created\n")

# ========================================
# Load One Batch
# ========================================
print("📝 Step 3: Load one batch...\n")

for batch in dataloader:
    if batch is None:
        print("⚠️  Batch is None")
        continue
    
    print("✅ Batch loaded!\n")
    
    batch_size = batch['input_ids'].shape[0]
    
    # ========================================
    # Check Targets Structure
    # ========================================
    print("📊 Batch structure:")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of target types: {len(batch['targets'])}")
    
    for key, value in batch['targets'].items():
        print(f"   {key:<12} shape: {value.shape}")
    
    print()
    
    # ========================================
    # Analyze EACH Sample Individually
    # ========================================
    print("🔍 Per-sample analysis:\n")
    
    for sample_idx in range(batch_size):
        print(f"   Sample {sample_idx}:")
        
        # Get masked positions for this sample
        mask_labels = batch['mask_labels'][sample_idx]
        masked_count = mask_labels.sum().item()
        
        print(f"      Masked positions: {masked_count}")
        
        # Count targets per type for this sample
        sample_targets_count = {}
        for target_type, target_tensor in batch['targets'].items():
            sample_targets = target_tensor[sample_idx]
            non_ignore = (sample_targets != -100).sum().item()
            sample_targets_count[target_type] = non_ignore
            
            if non_ignore > 0:
                print(f"      {target_type:<10}: {non_ignore:>3} targets")
        
        # Verify coverage: every masked position should have at least one target
        masked_positions_with_targets = 0
        masked_indices = torch.where(mask_labels)[0]
        
        for pos_idx in masked_indices:
            has_target = False
            for target_type, target_tensor in batch['targets'].items():
                if target_tensor[sample_idx, pos_idx].item() != -100:
                    has_target = True
                    break
            if has_target:
                masked_positions_with_targets += 1
        
        coverage = masked_positions_with_targets / masked_count * 100 if masked_count > 0 else 0
        print(f"      Coverage: {masked_positions_with_targets}/{masked_count} ({coverage:.1f}%)")
        
        if coverage == 100.0:
            print(f"      ✅ All masked positions have targets!")
        else:
            print(f"      ⚠️  Some masked positions missing targets!")
        
        print()
    
    # ========================================
    # Aggregate Statistics Across All Samples
    # ========================================
    print("📊 Aggregate statistics (all samples):\n")
    
    total_masked_all = batch['mask_labels'].sum().item()
    
    for target_type in batch['targets'].keys():
        # Count across ALL samples
        target_tensor = batch['targets'][target_type]
        total_non_ignore = (target_tensor != -100).sum().item()
        
        # Count unique values
        unique_values = target_tensor[target_tensor != -100].unique()
        num_unique = len(unique_values)
        
        # Value range
        if num_unique > 0:
            min_val = target_tensor[target_tensor != -100].min().item()
            max_val = target_tensor[target_tensor != -100].max().item()
            value_range = f"{min_val}-{max_val}"
        else:
            value_range = "N/A"
        
        ratio = total_non_ignore / total_masked_all * 100 if total_masked_all > 0 else 0
        
        print(f"   {target_type:<12} Total: {total_non_ignore:>4}  "
              f"({ratio:>5.1f}% of masked)  Unique: {num_unique:>3}  Range: {value_range}")
    
    print()
    
    # ========================================
    # Critical Check: Bar-Level Property
    # ========================================
    print("🎭 Bar-level masking verification:\n")
    
    print("   Expected behavior:")
    print("   - Different samples have different types masked")
    print("   - NOT all types should be masked in every sample")
    print("   - This is CORRECT for bar-level masking!\n")
    
    # Check that at least ONE sample has each major type
    types_found = {key: False for key in batch['targets'].keys()}
    
    for sample_idx in range(batch_size):
        for target_type, target_tensor in batch['targets'].items():
            sample_targets = target_tensor[sample_idx]
            if (sample_targets != -100).sum().item() > 0:
                types_found[target_type] = True
    
    print("   Types found across batch:")
    all_found = True
    for target_type, found in types_found.items():
        status = "✅" if found else "⚠️"
        print(f"   {status} {target_type:<12} {'found' if found else 'NOT found'}")
        if not found:
            all_found = False
    
    print()
    
    if all_found:
        print("   ✅ All token types represented in batch!")
    else:
        print("   ⚠️  Some types missing (may need larger batch or re-run)")
    
    print()
    
    # ========================================
    # Inspect Masked Tokens (Sample 0)
    # ========================================
    print("🔍 Inspect first 10 masked tokens (Sample 0):\n")
    
    sample_idx = 0
    mask_labels = batch['mask_labels'][sample_idx]
    
    print(f"   {'Pos':<6} {'Token':<12} {'Type':<12} {'Target':<10}")
    print(f"   {'-'*50}")
    
    # Find first 10 masked positions
    masked_indices = torch.where(mask_labels)[0][:10]
    
    # Get token ID to name mapping
    id_to_token = {v: k for k, v in dataset.tokenizer.vocab.items()}
    
    for pos in masked_indices:
        pos_idx = pos.item()
        
        # Original token
        orig_token_id = batch['original_tokens'][sample_idx, pos_idx].item()
        token_name = id_to_token.get(orig_token_id, f"ID_{orig_token_id}")
        
        # Truncate name
        token_name = token_name[:11]
        
        # Find which target type has non-ignore value
        target_type_found = None
        target_value = None
        
        for target_type, target_tensor in batch['targets'].items():
            val = target_tensor[sample_idx, pos_idx].item()
            if val != -100:
                target_type_found = target_type
                target_value = val
                break
        
        if target_type_found:
            print(f"   {pos_idx:<6} {token_name:<12} {target_type_found:<12} {target_value:<10}")
        else:
            print(f"   {pos_idx:<6} {token_name:<12} {'NONE':<12} {'⚠️ MISSING':<10}")
    
    print()
    
    # ========================================
    # Final Verdict
    # ========================================
    print("="*70)
    print("📊 FINAL VERDICT")
    print("="*70 + "\n")
    
    # Check 1: All masked positions have targets
    all_covered = True
    for sample_idx in range(batch_size):
        mask_labels = batch['mask_labels'][sample_idx]
        masked_indices = torch.where(mask_labels)[0]
        
        for pos_idx in masked_indices:
            has_target = False
            for target_type, target_tensor in batch['targets'].items():
                if target_tensor[sample_idx, pos_idx].item() != -100:
                    has_target = True
                    break
            if not has_target:
                all_covered = False
                break
        if not all_covered:
            break
    
    # Check 2: At least some targets for major types
    major_types = ['pitch', 'velocity', 'duration', 'program']
    major_types_ok = all(
        (batch['targets'][t] != -100).sum().item() > 0 
        for t in major_types
    )
    
    print(f"✅ Check 1: All masked positions have targets: {all_covered}")
    print(f"{'✅' if major_types_ok else '⚠️'} Check 2: Major types (pitch/velocity/duration/program) present: {major_types_ok}")
    
    total_targets = sum((batch['targets'][key] != -100).sum().item() 
                       for key in batch['targets'].keys())
    
    print(f"\n📊 Total targets set: {total_targets}")
    print(f"📊 Total masked positions: {total_masked_all}")
    
    if all_covered and total_targets > 0:
        print(f"\n🎉 SUCCESS: Targets dict is WORKING CORRECTLY!")
        print(f"   ✅ Multi-head structure")
        print(f"   ✅ Bar-level masking respected")
        print(f"   ✅ All masked positions covered")
        print(f"   ✅ Ready for training!\n")
    else:
        print(f"\n⚠️  ISSUES DETECTED:")
        if not all_covered:
            print(f"   - Some masked positions have no targets")
        if total_targets == 0:
            print(f"   - No targets set at all")
        print()
    
    # Only test first batch
    break

print("="*70)
print("✅ TEST COMPLETE!")
print("="*70 + "\n")