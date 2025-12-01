"""Test bar-level masking on a single chunk."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from miditok import REMI, TokenizerConfig
from midit.data.masking import BarLevelMasking

print("\n" + "="*70)
print("🎭 TEST BAR-LEVEL MASKING")
print("="*70 + "\n")

# ========================================
# STEP 1: Setup Tokenizer
# ========================================
print("📝 Step 1: Initialize tokenizer...\n")

config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8},
    num_velocities=32,
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_programs=True,
    special_tokens=["PAD", "BOS", "EOS", "MASK"],
    use_controls=False,
)

tokenizer = REMI(config)

print(f"✅ Tokenizer created")
print(f"   Vocab size: {len(tokenizer.vocab)}")

# Get MASK token ID
mask_token_id = tokenizer.vocab.get("MASK", None)
if mask_token_id is None:
    print("   ⚠️  MASK token not found in vocab!")
    # Find special tokens
    special_tokens = [k for k in tokenizer.vocab.keys() if k in ["PAD", "BOS", "EOS", "MASK"]]
    print(f"   Special tokens found: {special_tokens}")
    # Use first special token as MASK for testing
    mask_token_id = tokenizer.vocab[special_tokens[0]] if special_tokens else 0

print(f"   MASK token ID: {mask_token_id}\n")

# ========================================
# Get MASK token ID correctly
# ========================================
print("🔍 Finding MASK token in vocab...")

# Try to find MASK token
mask_token_id = None
for token_name, token_id in tokenizer.vocab.items():
    if 'MASK' in token_name:
        mask_token_id = token_id
        print(f"   ✅ Found MASK token: '{token_name}' = ID {token_id}")
        break

if mask_token_id is None:
    print("   ⚠️  MASK token not found!")
    # Fallback: use max_id + 1
    mask_token_id = max(tokenizer.vocab.values()) + 1
    print(f"   Using fallback ID: {mask_token_id}")
else:
    print(f"   Using MASK token ID: {mask_token_id}")

print(f"   Mask token ID: {mask_token_id}\n")

# ========================================
# STEP 2: Load Preprocessed Chunk
# ========================================
print("📝 Step 2: Load preprocessed chunk...\n")

data_dir = Path("data/preprocessed_phase1")
pt_files = list(data_dir.glob("*.pt"))

if len(pt_files) == 0:
    print("❌ No .pt files found!")
    exit(1)

# Load first file
data = torch.load(pt_files[0])

print(f"✅ Loaded: {pt_files[0].name}")
print(f"   Total tokens: {len(data['tokens'])}")
print(f"   Total bars: {len(data['bar_positions']) - 1}")
print(f"   Source: {data['metadata']['source_file']}\n")

# Take a chunk (first 16 bars or all if less)
num_bars = len(data['bar_positions']) - 1
bars_to_use = min(16, num_bars)

start_bar = 0
end_bar = bars_to_use

start_idx = data['bar_positions'][start_bar]
end_idx = data['bar_positions'][end_bar]

chunk_tokens = data['tokens'][start_idx:end_idx]
chunk_bar_positions = [pos - start_idx for pos in data['bar_positions'][start_bar:end_bar+1]]

print(f"📏 Chunk extracted:")
print(f"   Bars: {start_bar} to {end_bar} ({bars_to_use} bars)")
print(f"   Tokens: {len(chunk_tokens)}")
print(f"   Bar positions: {chunk_bar_positions[:5]}...\n")

# ========================================
# STEP 3: Initialize Masking
# ========================================
print("📝 Step 3: Initialize bar-level masking...\n")

masker = BarLevelMasking(
    tokenizer_vocab=tokenizer.vocab,
    mask_token_id=mask_token_id,
    bar_mask_prob=0.15,
    type_mask_ratio=0.3,
)

# ========================================
# STEP 4: Apply Masking
# ========================================
print("📝 Step 4: Apply masking...\n")

masked_tokens, mask_labels = masker.apply_masking(
    tokens=chunk_tokens,
    bar_positions=chunk_bar_positions,
)

print(f"✅ Masking applied!")
print(f"   Original tokens: {len(chunk_tokens)}")
print(f"   Masked tokens: {len(masked_tokens)}")
print(f"   Mask labels: {len(mask_labels)}\n")

# ========================================
# STEP 5: Statistics
# ========================================
print("📝 Step 5: Masking statistics...\n")

stats = masker.get_masking_stats(mask_labels)

print(f"📊 Statistics:")
print(f"   Total tokens:  {stats['total_tokens']}")
print(f"   Masked tokens: {stats['masked_tokens']}")
print(f"   Mask ratio:    {stats['mask_ratio']:.2%}\n")

# ========================================
# STEP 6: Inspect Masked Positions
# ========================================
print("📝 Step 6: Inspect masked positions...\n")

# Find first masked bar
for bar_idx in range(len(chunk_bar_positions) - 1):
    start = chunk_bar_positions[bar_idx]
    end = chunk_bar_positions[bar_idx + 1]
    
    bar_mask_labels = mask_labels[start:end]
    
    if bar_mask_labels.sum() > 0:
        print(f"🎯 First masked bar found: Bar {bar_idx}")
        print(f"   Token range: {start} to {end}")
        print(f"   Tokens in bar: {end - start}")
        print(f"   Masked in bar: {bar_mask_labels.sum().item()}\n")
        
        # Show first 20 tokens of this bar
        print(f"   First 20 tokens in masked bar:")
        print(f"   {'Position':<10} {'Original':<15} {'Masked':<15} {'Type':<12} {'Masked?'}")
        print(f"   {'-'*70}")
        
        for i in range(min(20, end - start)):
            pos = start + i
            orig_id = chunk_tokens[pos].item()
            masked_id = masked_tokens[pos].item()
            is_masked = mask_labels[pos].item()
            
            # Get token names
            orig_name = masker.id_to_token.get(orig_id, f"ID_{orig_id}")
            masked_name = masker.id_to_token.get(masked_id, f"ID_{masked_id}")
            token_type = masker.get_token_type(orig_id) or "unknown"
            
            # Truncate long names
            orig_name = orig_name[:14]
            masked_name = masked_name[:14]
            
            mask_mark = "✓ MASKED" if is_masked else ""
            
            print(f"   {pos:<10} {orig_name:<15} {masked_name:<15} {token_type:<12} {mask_mark}")
        
        print()
        break

# ========================================
# STEP 7: Verify Bar-Level Property
# ========================================
print("📝 Step 7: Verify bar-level masking property...\n")

# Check that all tokens of same type in masked bars are masked
for bar_idx in range(len(chunk_bar_positions) - 1):
    start = chunk_bar_positions[bar_idx]
    end = chunk_bar_positions[bar_idx + 1]
    
    bar_mask_labels = mask_labels[start:end]
    
    if bar_mask_labels.sum() == 0:
        continue
    
    # Find which types are masked in this bar
    masked_types = set()
    for pos in range(start, end):
        if mask_labels[pos]:
            token_type = masker.get_token_type(chunk_tokens[pos].item())
            if token_type:
                masked_types.add(token_type)
    
    # Verify ALL tokens of these types are masked
    all_consistent = True
    for pos in range(start, end):
        token_type = masker.get_token_type(chunk_tokens[pos].item())
        if token_type in masked_types:
            if not mask_labels[pos]:
                all_consistent = False
                print(f"   ⚠️  Bar {bar_idx}: Type '{token_type}' not fully masked!")
    
    if all_consistent and len(masked_types) > 0:
        print(f"   ✅ Bar {bar_idx}: Types {masked_types} fully masked")

print()

# ========================================
# SUMMARY
# ========================================
print("="*70)
print("✅ MASKING TEST COMPLETE!")
print("="*70 + "\n")

if stats['masked_tokens'] > 0:
    print("✅ Masking is working!")
    print(f"   - {stats['masked_tokens']} tokens masked ({stats['mask_ratio']:.1%})")
    print("   - Bar-level property verified")
else:
    print("⚠️  No tokens were masked!")
    print("   - Try running again (random process)")
    print("   - Or increase bar_mask_prob")

print()