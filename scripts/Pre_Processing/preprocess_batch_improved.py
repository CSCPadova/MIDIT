"""
Batch Preprocessing: Improved Version
Based on feedback - production ready with optimizations
"""

import torch
from pathlib import Path
from miditok import REMI, TokenizerConfig
import statistics
from tqdm import tqdm

print("\n" + "="*70)
print("🎵 BATCH PREPROCESSING - IMPROVED VERSION")
print("="*70 + "\n")

# ========================================
# STEP 1: Setup
# ========================================
print("📝 Step 1: Setup tokenizer and paths...\n")

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
print(f"   Vocab size: {len(tokenizer.vocab)}")

# FIX 1: Pre-build reverse vocab (O(1) lookup instead of O(N))
print(f"   Building reverse vocab lookup...")
id_to_token = {v: k for k, v in tokenizer.vocab.items()}
print(f"   ✅ Reverse vocab ready\n")

# Output directory
output_dir = Path("data/preprocessed_phase1")
output_dir.mkdir(exist_ok=True, parents=True)

print(f"✅ Output directory: {output_dir}\n")

# ========================================
# STEP 2: Load Files
# ========================================
print("📝 Step 2: Find MIDI files...\n")

lakh_dir = Path("data/lakh_clean")
all_midi_files = list(lakh_dir.rglob("*.mid"))

# Take first 10
midi_files = all_midi_files[:10]

print(f"✅ Found {len(all_midi_files):,} total files")
print(f"   Processing first {len(midi_files)} files\n")

# ========================================
# STEP 3: Process Each File
# ========================================
print("📝 Step 3: Processing files...\n")

stats = {
    'success': 0,
    'errors': 0,
    'total_tokens': [],
    'total_bars': [],
    'tokens_per_bar': [],
}

for i, midi_path in enumerate(tqdm(midi_files, desc="Processing")):
    try:
        # Tokenize
        tokens = tokenizer(str(midi_path))
        
        # Handle format
        if isinstance(tokens, list):
            all_ids = []
            for tok_seq in tokens:
                all_ids.extend(tok_seq.ids)
        else:
            all_ids = tokens.ids
        
        # FIX 1: Fast token name lookup using reverse vocab
        bar_positions = []
        for idx, token_id in enumerate(all_ids):
            token_name = id_to_token.get(token_id, None)
            if token_name and "Bar" in token_name:
                bar_positions.append(idx)
        
        # FIX 4: Cleaner bar_positions initialization
        if not bar_positions or bar_positions[0] != 0:
            bar_positions.insert(0, 0)
        bar_positions.append(len(all_ids))
        
        # FIX 2: Extract additional metadata
        # Get tempo, time signature, programs
        first_tempo = None
        first_timesig = None
        programs_used = set()
        
        for token_id in all_ids[:100]:  # Check first 100 tokens
            token_name = id_to_token.get(token_id, None)
            if token_name:
                if token_name.startswith('Tempo_') and first_tempo is None:
                    first_tempo = token_name
                elif token_name.startswith('TimeSig_') and first_timesig is None:
                    first_timesig = token_name
                elif token_name.startswith('Program_'):
                    programs_used.add(token_name)
        
        # Calculate stats
        num_bars = len(bar_positions) - 1
        tokens_in_bars = []
        for j in range(len(bar_positions) - 1):
            tokens_in_bar = bar_positions[j+1] - bar_positions[j]
            tokens_in_bars.append(tokens_in_bar)
        
        # Enhanced metadata
        metadata = {
            'source_file': str(midi_path.name),
            'source_path': str(midi_path.relative_to(lakh_dir)),
            'num_tokens': len(all_ids),
            'num_bars': num_bars,
            'tokens_per_bar_mean': statistics.mean(tokens_in_bars) if tokens_in_bars else 0,
            'tokens_per_bar_min': min(tokens_in_bars) if tokens_in_bars else 0,
            'tokens_per_bar_max': max(tokens_in_bars) if tokens_in_bars else 0,
            # FIX 2: Additional musical metadata
            'initial_tempo': first_tempo,
            'initial_timesig': first_timesig,
            'num_programs': len(programs_used),
            'programs': list(programs_used),
        }
        
        # FIX 3: More descriptive filename
        # Include source identifier for easier tracking
        source_id = midi_path.stem[:20]  # First 20 chars of original name
        output_filename = f"lakh_{i:04d}_{source_id}.pt"
        output_path = output_dir / output_filename
        
        # Save
        data = {
            'tokens': torch.tensor(all_ids, dtype=torch.long),
            'bar_positions': bar_positions,
            'metadata': metadata,
        }
        
        torch.save(data, output_path)
        
        # Update stats
        stats['success'] += 1
        stats['total_tokens'].append(len(all_ids))
        stats['total_bars'].append(num_bars)
        stats['tokens_per_bar'].extend(tokens_in_bars)
        
    except Exception as e:
        stats['errors'] += 1
        print(f"\n   ⚠️  Error on {midi_path.name}: {e}")

# ========================================
# STEP 4: Enhanced Statistics
# ========================================
print(f"\n{'='*70}")
print("📊 BATCH STATISTICS")
print(f"{'='*70}\n")

print(f"✅ Successfully processed: {stats['success']}/{len(midi_files)}")
print(f"❌ Errors: {stats['errors']}/{len(midi_files)}\n")

if stats['success'] > 0:
    print(f"📊 Tokens:")
    print(f"   Total:  {sum(stats['total_tokens']):,}")
    print(f"   Mean:   {statistics.mean(stats['total_tokens']):,.0f}")
    print(f"   Median: {statistics.median(stats['total_tokens']):,.0f}")
    print(f"   Min:    {min(stats['total_tokens']):,}")
    print(f"   Max:    {max(stats['total_tokens']):,}\n")
    
    print(f"📊 Bars:")
    print(f"   Total:  {sum(stats['total_bars']):,}")
    print(f"   Mean:   {statistics.mean(stats['total_bars']):.1f}")
    print(f"   Min:    {min(stats['total_bars'])}")
    print(f"   Max:    {max(stats['total_bars'])}\n")
    
    print(f"📊 Tokens per bar:")
    print(f"   Mean:   {statistics.mean(stats['tokens_per_bar']):.1f}")
    print(f"   Median: {statistics.median(stats['tokens_per_bar']):.1f}")
    print(f"   Min:    {min(stats['tokens_per_bar'])}")
    print(f"   Max:    {max(stats['tokens_per_bar'])}\n")
    
    # Disk usage
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.pt"))
    print(f"💾 Disk usage:")
    print(f"   Total: {total_size / 1024 / 1024:.1f} MB")
    print(f"   Per file: {total_size / stats['success'] / 1024:.1f} KB\n")

print(f"{'='*70}")
print("✅ BATCH PREPROCESSING COMPLETE!")
print(f"{'='*70}\n")

print(f"📁 Output directory: {output_dir}")
print(f"📊 Files created: {stats['success']}")

# Show sample output filenames
print(f"\n💡 Sample output files:")
for f in list(output_dir.glob("*.pt"))[:3]:
    print(f"   - {f.name}")

print(f"\n💡 Next step: Scale to 500 files with preprocess_lakh.py")
print()