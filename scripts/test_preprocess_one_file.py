"""
Test Preprocessing: Process 1 MIDI file and save as .pt

Adapts validated code from token_per_barV2.py
"""

import torch
from pathlib import Path
from miditok import REMI, TokenizerConfig
import miditoolkit

print("\n" + "="*70)
print("🧪 TEST PREPROCESSING - SINGLE FILE")
print("="*70 + "\n")

# ========================================
# STEP 1: Setup Tokenizer
# ========================================
print("📝 Step 1: Setup REMI tokenizer...\n")

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

print(f"✅ Tokenizer initialized")
print(f"   Vocab size: {len(tokenizer.vocab)}\n")

# ========================================
# STEP 2: Load MIDI File
# ========================================
print("📝 Step 2: Load MIDI file...\n")

# Use first file from lakh_clean
midi_path = Path("data/lakh_clean/.38 Special/Caught Up In You.mid")

if not midi_path.exists():
    print(f"❌ File not found: {midi_path}")
    print("\n   Looking for any MIDI file...")
    midi_files = list(Path("data/lakh_clean").rglob("*.mid"))
    if len(midi_files) > 0:
        midi_path = midi_files[0]
        print(f"   Using: {midi_path.name}")
    else:
        print("❌ No MIDI files found!")
        exit(1)

print(f"📁 File: {midi_path.name}")
print(f"   Path: {midi_path}\n")

# ========================================
# STEP 3: Tokenize
# ========================================
print("📝 Step 3: Tokenize...\n")

try:
    tokens = tokenizer(str(midi_path))
    
    # Handle format (from token_per_barV2.py)
    if isinstance(tokens, list):
        all_ids = []
        for tok_seq in tokens:
            all_ids.extend(tok_seq.ids)
    else:
        all_ids = tokens.ids
    
    print(f"✅ Tokenized")
    print(f"   Total tokens: {len(all_ids)}\n")
    
except Exception as e:
    print(f"❌ Tokenization failed: {e}")
    exit(1)

# ========================================
# STEP 4: Extract Bar Positions
# ========================================
print("📝 Step 4: Extract bar positions...\n")

bar_positions = []
for idx, token_id in enumerate(all_ids):
    # Find token name
    token_name = None
    for name, tid in tokenizer.vocab.items():
        if tid == token_id:
            token_name = name
            break
    
    if token_name and "Bar" in token_name:
        bar_positions.append(idx)

# Add start and end
if 0 not in bar_positions:
    bar_positions = [0] + bar_positions
bar_positions.append(len(all_ids))

print(f"✅ Bar positions extracted")
print(f"   Total bars: {len(bar_positions) - 1}")
print(f"   First 5 positions: {bar_positions[:5]}")
print(f"   Last 5 positions: {bar_positions[-5:]}\n")

# Calculate tokens per bar
tokens_per_bar = []
for i in range(len(bar_positions) - 1):
    tokens_in_bar = bar_positions[i+1] - bar_positions[i]
    tokens_per_bar.append(tokens_in_bar)

if tokens_per_bar:
    import statistics
    print(f"   Tokens per bar:")
    print(f"      Mean: {statistics.mean(tokens_per_bar):.1f}")
    print(f"      Min: {min(tokens_per_bar)}")
    print(f"      Max: {max(tokens_per_bar)}\n")

# ========================================
# STEP 5: Create Metadata
# ========================================
print("📝 Step 5: Create metadata...\n")

try:
    midi = miditoolkit.MidiFile(str(midi_path))
    duration_seconds = midi.max_tick / midi.ticks_per_beat / 2.0
except:
    duration_seconds = 0.0

metadata = {
    'source_file': str(midi_path.name),
    'num_tokens': len(all_ids),
    'num_bars': len(bar_positions) - 1,
    'duration_seconds': duration_seconds,
    'tokens_per_bar_mean': statistics.mean(tokens_per_bar) if tokens_per_bar else 0,
}

print(f"✅ Metadata created")
for key, value in metadata.items():
    print(f"   {key}: {value}")
print()

# ========================================
# STEP 6: Save to .pt File
# ========================================
print("📝 Step 6: Save to .pt file...\n")

output_dir = Path("data/test_preprocessing")
output_dir.mkdir(exist_ok=True, parents=True)

output_path = output_dir / "test_song.pt"

data = {
    'tokens': torch.tensor(all_ids, dtype=torch.long),
    'bar_positions': bar_positions,
    'metadata': metadata,
}

torch.save(data, output_path)

print(f"✅ Saved to: {output_path}")
print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB\n")

# ========================================
# STEP 7: Reload and Validate
# ========================================
print("📝 Step 7: Reload and validate...\n")

loaded_data = torch.load(output_path)

print(f"✅ Reloaded successfully")
print(f"\n   Contents:")
print(f"   - tokens: {loaded_data['tokens'].shape}")
print(f"   - bar_positions: {len(loaded_data['bar_positions'])} positions")
print(f"   - metadata: {len(loaded_data['metadata'])} fields")

print(f"\n   Validation:")
print(f"   ✅ Tokens match: {len(loaded_data['tokens']) == len(all_ids)}")
print(f"   ✅ Bars match: {len(loaded_data['bar_positions']) == len(bar_positions)}")
print(f"   ✅ Metadata matches: {loaded_data['metadata'] == metadata}")

# ========================================
# STEP 8: Summary
# ========================================
print(f"\n{'='*70}")
print("✅ PREPROCESSING TEST COMPLETE!")
print(f"{'='*70}\n")

print(f"📊 Summary:")
print(f"   Input:  {midi_path.name}")
print(f"   Tokens: {len(all_ids):,}")
print(f"   Bars:   {len(bar_positions) - 1}")
print(f"   Output: {output_path}")
print(f"\n💡 Next step: Scale to 10 files, then 100, then full dataset")
print()