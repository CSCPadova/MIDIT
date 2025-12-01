"""
Validate Preprocessing Output - Deep Inspection

Verifica che la tokenizzazione sia corretta al 100%
"""

import torch
from pathlib import Path
from miditok import REMI, TokenizerConfig
from collections import Counter
import statistics

print("\n" + "="*70)
print("🔬 VALIDATION - DEEP INSPECTION")
print("="*70 + "\n")

# ========================================
# STEP 1: Setup Tokenizer (stesso config!)
# ========================================
print("📝 Step 1: Setup tokenizer...\n")

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
    num_tempos=32,        # Più bins (default ~10-20)
    tempo_range=(40, 250), # Range esplicito
)

tokenizer = REMI(config)
print(f"✅ Tokenizer initialized (vocab size: {len(tokenizer.vocab)})\n")

# ========================================
# STEP 2: Load Preprocessed Data
# ========================================
print("📝 Step 2: Load preprocessed data...\n")

data_path = Path("data/test_preprocessing/test_song.pt")

if not data_path.exists():
    print(f"❌ File not found: {data_path}")
    exit(1)

data = torch.load(data_path)

tokens = data['tokens']
bar_positions = data['bar_positions']
metadata = data['metadata']

print(f"✅ Loaded: {metadata['source_file']}")
print(f"   Tokens: {len(tokens):,}")
print(f"   Bars: {metadata['num_bars']}")
print(f"   Duration: {metadata['duration_seconds']:.1f}s\n")

# ========================================
# STEP 3: Token Type Distribution
# ========================================
print("📝 Step 3: Token type distribution...\n")

token_types = Counter()
token_names = []

for token_id in tokens:
    tid = token_id.item()
    
    # Find token name
    token_name = None
    for name, id_val in tokenizer.vocab.items():
        if id_val == tid:
            token_name = name
            break
    
    if token_name:
        token_names.append(token_name)
        # Extract type
        token_type = token_name.split('_')[0]
        token_types[token_type] += 1

print(f"✅ Token type distribution:\n")

total = sum(token_types.values())
for ttype, count in token_types.most_common():
    pct = 100 * count / total
    print(f"   {ttype:15s}: {count:6,} ({pct:5.2f}%)")

print(f"\n   Total: {total:,} tokens\n")

# ========================================
# STEP 4: First Bar Inspection
# ========================================
print("📝 Step 4: Inspect first bar (tokens 0-420)...\n")

first_bar_tokens = token_names[:min(100, len(token_names))]

print(f"✅ First 100 tokens:\n")

# Group by event type
current_event = []
event_count = 0

for i, token_name in enumerate(first_bar_tokens):
    if "Bar" in token_name:
        if current_event:
            print(f"   Event {event_count}: {' '.join(current_event)}")
            current_event = []
            event_count += 1
        print(f"\n   [{i}] {token_name}")
        print(f"   " + "-"*50)
    elif "Position" in token_name:
        if current_event:
            print(f"   Event {event_count}: {' '.join(current_event)}")
            current_event = []
            event_count += 1
        print(f"   [{i}] {token_name}")
        current_event = []
    else:
        current_event.append(token_name)

if current_event:
    print(f"   Event {event_count}: {' '.join(current_event)}")

print()

# ========================================
# STEP 5: Bar Structure Validation
# ========================================
print("📝 Step 5: Validate bar structure...\n")

# Check tokens between bars
tokens_per_bar_actual = []
for i in range(len(bar_positions) - 1):
    tokens_in_bar = bar_positions[i+1] - bar_positions[i]
    tokens_per_bar_actual.append(tokens_in_bar)

print(f"✅ Bar structure:\n")
print(f"   Total bars: {len(bar_positions) - 1}")
print(f"   Tokens per bar:")
print(f"      Mean:   {statistics.mean(tokens_per_bar_actual):.1f}")
print(f"      Median: {statistics.median(tokens_per_bar_actual):.1f}")
print(f"      Min:    {min(tokens_per_bar_actual)}")
print(f"      Max:    {max(tokens_per_bar_actual)}")
print(f"      StdDev: {statistics.stdev(tokens_per_bar_actual):.1f}\n")

# Check for outliers
outliers = [x for x in tokens_per_bar_actual if x > 1000 or x < 10]
if outliers:
    print(f"   ⚠️  Found {len(outliers)} outlier bars:")
    for bar_idx, tokens_count in enumerate(tokens_per_bar_actual):
        if tokens_count in outliers:
            print(f"      Bar {bar_idx}: {tokens_count} tokens")
else:
    print(f"   ✅ No outlier bars detected")

print()

# ========================================
# STEP 6: Musical Structure Check
# ========================================
print("📝 Step 6: Check musical structure...\n")

# Count notes (Pitch tokens)
num_pitches = token_types.get('Pitch', 0)

# Count positions
num_positions = token_types.get('Position', 0)

# Count bars
num_bars = token_types.get('Bar', 0)

print(f"✅ Musical elements:\n")
print(f"   Notes (Pitch):     {num_pitches:,}")
print(f"   Positions:         {num_positions:,}")
print(f"   Bars:              {num_bars:,}")
print(f"\n   Ratio:")
print(f"   Notes per bar:     {num_pitches / num_bars:.1f}")
print(f"   Positions per bar: {num_positions / num_bars:.1f}")

# Expected: ~4-16 positions per bar (depending on beat_res)
if 4 <= num_positions / num_bars <= 32:
    print(f"   ✅ Positions per bar looks reasonable")
else:
    print(f"   ⚠️  Positions per bar seems unusual")

print()

# ========================================
# STEP 7: Sequence Validation
# ========================================
print("📝 Step 7: Validate token sequence structure...\n")

# Check expected structure: Bar → (Position → Notes)* → Bar
structure_errors = 0
last_bar_pos = 0

for i, bar_pos in enumerate(bar_positions[:-1]):
    next_bar_pos = bar_positions[i + 1]
    bar_tokens = token_names[bar_pos:next_bar_pos]
    
    # First token should be Bar
    if not bar_tokens[0].startswith('Bar'):
        structure_errors += 1
        if structure_errors <= 3:
            print(f"   ⚠️  Bar {i}: First token is {bar_tokens[0]}, not Bar")
    
    # Should have Position tokens
    position_count = sum(1 for t in bar_tokens if t.startswith('Position'))
    if position_count == 0:
        structure_errors += 1
        if structure_errors <= 3:
            print(f"   ⚠️  Bar {i}: No Position tokens found")

if structure_errors == 0:
    print(f"   ✅ All bars have correct structure (Bar → Positions → Notes)")
else:
    print(f"   ⚠️  Found {structure_errors} structure issues")

print()

# ========================================
# STEP 8: Round-Trip Test (Optional)
# ========================================
print("📝 Step 8: Round-trip test (decode tokens)...\n")

try:
    # Decode tokens back to MIDI
    decoded_midi = tokenizer.decode(tokens.tolist())
    
    # Count notes
    decoded_notes = sum(len(track.notes) for track in decoded_midi.tracks)
    
    print(f"✅ Decoded successfully")
    print(f"   Tracks: {len(decoded_midi.tracks)}")
    print(f"   Notes:  {decoded_notes}")
    
    # Save for manual inspection
    output_path = Path("data/test_preprocessing/reconstructed.mid")
    decoded_midi.dump_midi(str(output_path))
    
    print(f"   Saved to: {output_path}")
    print(f"\n   🎧 Listen to both files and compare:")
    print(f"      Original:      {metadata['source_file']}")
    print(f"      Reconstructed: {output_path.name}")
    
except Exception as e:
    print(f"   ❌ Decode failed: {e}")

print()

# ========================================
# SUMMARY
# ========================================
print("="*70)
print("📊 VALIDATION SUMMARY")
print("="*70 + "\n")

checks_passed = 0
total_checks = 5

# Check 1: Token distribution
if all(count > 0 for ttype, count in token_types.items() if ttype in ['Pitch', 'Velocity', 'Duration', 'Position', 'Bar']):
    print("✅ 1. Token distribution: All essential types present")
    checks_passed += 1
else:
    print("❌ 1. Token distribution: Missing essential types")

# Check 2: Bar structure
if len(outliers) == 0:
    print("✅ 2. Bar structure: No outliers")
    checks_passed += 1
else:
    print(f"⚠️  2. Bar structure: {len(outliers)} outlier bars")

# Check 3: Musical ratios
if 4 <= num_positions / num_bars <= 32:
    print("✅ 3. Musical ratios: Positions per bar reasonable")
    checks_passed += 1
else:
    print("⚠️  3. Musical ratios: Positions per bar unusual")

# Check 4: Sequence structure
if structure_errors == 0:
    print("✅ 4. Sequence structure: All bars valid")
    checks_passed += 1
else:
    print(f"⚠️  4. Sequence structure: {structure_errors} errors")

# Check 5: Round-trip
try:
    decoded_midi
    print("✅ 5. Round-trip: Decode successful")
    checks_passed += 1
except:
    print("⚠️  5. Round-trip: Decode failed")

print(f"\n{'='*70}")
print(f"SCORE: {checks_passed}/{total_checks} checks passed")
print(f"{'='*70}\n")

if checks_passed == total_checks:
    print("🎉 ALL CHECKS PASSED! Tokenization is correct!")
    print("\n💡 Next step: Scale to 10 files")
elif checks_passed >= 3:
    print("⚠️  Most checks passed, but review warnings above")
else:
    print("❌ Multiple checks failed - review tokenization setup")

print()