"""Safety check: verify vocab sizes match actual targets."""

from midit.data.lakh_dataset import LakhDataset
from omegaconf import DictConfig

# Config minimo
chunking_config = DictConfig({
    'strategy': 'bar_based',
    'bars_per_chunk': 8,
    'max_tokens': 8192,
})

print("\n" + "="*70)
print("🔍 VOCAB SAFETY CHECK")
print("="*70 + "\n")

# Load dataset
print("📦 Loading dataset...")
ds = LakhDataset(
    data_dir="data/preprocessed_phase1",
    chunking_config=chunking_config,
    split='train',
    apply_masking=True,
)

print("\n📊 Declared vocab sizes:")
for token_type, size in ds.actual_vocab_sizes.items():
    print(f"   {token_type:10s}: {size:>4d}")

print("\n🔍 Testing 10 samples for max target values...")
print("-"*70)

max_targets = {k: -1 for k in ds.actual_vocab_sizes.keys()}

for i in range(min(10, len(ds))):
    sample = ds[i]
    if sample is None:
        continue
    
    targets = sample['targets']
    
    for token_type, target_tensor in targets.items():
        # Ignora -100 (ignore_index)
        valid_targets = target_tensor[target_tensor != -100]
        if len(valid_targets) > 0:
            max_val = valid_targets.max().item()
            max_targets[token_type] = max(max_targets[token_type], max_val)

print("\n📊 Maximum target values found:")
print(f"{'Type':<12} {'Declared':<10} {'Max Target':<12} {'Safe?':<10}")
print("-"*70)

all_safe = True
for token_type in ds.actual_vocab_sizes.keys():
    declared = ds.actual_vocab_sizes[token_type]
    max_target = max_targets[token_type]
    safe = max_target < declared if max_target >= 0 else True
    status = "✅ OK" if safe else "❌ FAIL"
    
    if not safe:
        all_safe = False
    
    print(f"{token_type:<12} {declared:<10} {max_target:<12} {status:<10}")

print("-"*70)

if all_safe:
    print("\n✅ ALL CHECKS PASSED! Safe to train.")
else:
    print("\n❌ SAFETY CHECK FAILED! Fix vocab sizes before training.")
    print("\n🔧 Likely fixes needed:")
    for token_type in ds.actual_vocab_sizes.keys():
        if max_targets[token_type] >= ds.actual_vocab_sizes[token_type]:
            print(f"   - {token_type}: increase to at least {max_targets[token_type] + 1}")

print("\n" + "="*70 + "\n")