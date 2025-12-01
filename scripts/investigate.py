"""
Investigation: Program -1 Tokens
What are they? Where do they come from?
Can we filter them?
"""

from miditok import REMI, TokenizerConfig
from pathlib import Path
import symusic

print("\n" + "="*70)
print("🔍 PROGRAM -1 INVESTIGATION")
print("="*70 + "\n")

# ========================================
# Setup Tokenizer
# ========================================
config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8},
    num_velocities=32,
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_programs=True,
    special_tokens=["PAD", "BOS", "EOS", "MASK"]
)

tokenizer = REMI(config)

print("✅ Tokenizer created\n")

# ========================================
# Load Files
# ========================================
lakh_path = Path("data/lakh_clean")
midi_files = list(lakh_path.rglob("*.mid"))[:50]  # 50 files per velocità

print(f"📁 Analyzing {len(midi_files)} files...\n")

# ========================================
# Analysis
# ========================================
results = {
    'files_with_minus1': 0,
    'files_without_minus1': 0,
    'total_program_minus1': 0,
    'total_program_valid': 0,
    'minus1_track_info': [],
    'minus1_file_examples': [],
}

print("🔄 Processing...\n")

for i, midi_path in enumerate(midi_files):
    if (i + 1) % 10 == 0:
        print(f"   Progress: {i+1}/{len(midi_files)}")
    
    try:
        # ========================================
        # Load Original MIDI to inspect tracks
        # ========================================
        score = symusic.Score(str(midi_path))
        
        # ========================================
        # Tokenize
        # ========================================
        tokens = tokenizer(str(midi_path))
        
        # Handle format
        if isinstance(tokens, list):
            all_ids = []
            for tok_seq in tokens:
                all_ids.extend(tok_seq.ids)
        else:
            all_ids = tokens.ids
        
        # ========================================
        # Count Program tokens
        # ========================================
        program_minus1_count = 0
        program_valid_count = 0
        
        for token_id in all_ids:
            token_name = list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(token_id)]
            
            if "Program" in token_name:
                if "Program_-1" in token_name:
                    program_minus1_count += 1
                else:
                    program_valid_count += 1
        
        # ========================================
        # Record Results
        # ========================================
        if program_minus1_count > 0:
            results['files_with_minus1'] += 1
            results['total_program_minus1'] += program_minus1_count
            
            # Store track info for first 5 examples
            if len(results['minus1_file_examples']) < 5:
                track_info = []
                for track_idx, track in enumerate(score.tracks):
                    track_info.append({
                        'track_idx': track_idx,
                        'name': track.name if hasattr(track, 'name') else 'unnamed',
                        'is_drum': track.is_drum,
                        'program': track.program,
                        'note_count': len(track.notes),
                    })
                
                results['minus1_file_examples'].append({
                    'file': midi_path.name,
                    'program_minus1_count': program_minus1_count,
                    'program_valid_count': program_valid_count,
                    'tracks': track_info,
                })
            
            results['minus1_track_info'].append({
                'file': midi_path.name,
                'n_tracks': len(score.tracks),
                'drum_tracks': sum(1 for t in score.tracks if t.is_drum),
                'non_drum_tracks': sum(1 for t in score.tracks if not t.is_drum),
            })
        else:
            results['files_without_minus1'] += 1
        
        results['total_program_valid'] += program_valid_count
        
    except Exception as e:
        # print(f"\n   Error on {midi_path.name}: {e}")
        pass

# ========================================
# RESULTS
# ========================================
print(f"\n{'='*70}")
print("📊 RESULTS")
print(f"{'='*70}\n")

total_files = results['files_with_minus1'] + results['files_without_minus1']

print(f"📁 FILES ANALYZED: {total_files}")
print(f"   ✅ Files WITHOUT Program -1: {results['files_without_minus1']} ({100*results['files_without_minus1']/total_files:.1f}%)")
print(f"   ⚠️  Files WITH Program -1:    {results['files_with_minus1']} ({100*results['files_with_minus1']/total_files:.1f}%)\n")

total_program = results['total_program_minus1'] + results['total_program_valid']
print(f"🎸 PROGRAM TOKENS:")
print(f"   Valid programs:  {results['total_program_valid']:>7,} ({100*results['total_program_valid']/total_program:.1f}%)")
print(f"   Program -1:      {results['total_program_minus1']:>7,} ({100*results['total_program_minus1']/total_program:.1f}%)")

print(f"\n{'-'*70}\n")

# ========================================
# PATTERN ANALYSIS
# ========================================
print("🔍 PATTERN ANALYSIS:\n")

if len(results['minus1_track_info']) > 0:
    avg_tracks = sum(x['n_tracks'] for x in results['minus1_track_info']) / len(results['minus1_track_info'])
    avg_drums = sum(x['drum_tracks'] for x in results['minus1_track_info']) / len(results['minus1_track_info'])
    avg_non_drums = sum(x['non_drum_tracks'] for x in results['minus1_track_info']) / len(results['minus1_track_info'])
    
    print(f"   Files with Program -1 have on average:")
    print(f"   - Total tracks: {avg_tracks:.1f}")
    print(f"   - Drum tracks: {avg_drums:.1f}")
    print(f"   - Non-drum tracks: {avg_non_drums:.1f}\n")

print(f"{'-'*70}\n")

# ========================================
# DETAILED EXAMPLES
# ========================================
print("📋 DETAILED EXAMPLES (first 5 files with Program -1):\n")

for idx, example in enumerate(results['minus1_file_examples'], 1):
    print(f"Example {idx}: {example['file']}")
    print(f"   Program -1 tokens: {example['program_minus1_count']}")
    print(f"   Valid Program tokens: {example['program_valid_count']}")
    print(f"   Tracks ({len(example['tracks'])} total):")
    
    for track in example['tracks']:
        drum_marker = "🥁" if track['is_drum'] else "🎵"
        prog_str = f"Program {track['program']}" if track['program'] >= 0 else "NO PROGRAM"
        print(f"      {drum_marker} Track {track['track_idx']}: {prog_str} | {track['note_count']} notes | {track['name']}")
    
    print()

print(f"{'='*70}\n")

# ========================================
# CONCLUSIONS
# ========================================
print("💡 CONCLUSIONS:\n")

minus1_percentage = 100 * results['total_program_minus1'] / total_program if total_program > 0 else 0

if minus1_percentage < 5:
    print(f"   ✅ Program -1 is RARE ({minus1_percentage:.1f}%)")
    print(f"   → Can safely filter without losing much data\n")
elif minus1_percentage < 30:
    print(f"   ⚠️  Program -1 is MODERATE ({minus1_percentage:.1f}%)")
    print(f"   → Investigate further before filtering\n")
else:
    print(f"   🚨 Program -1 is COMMON ({minus1_percentage:.1f}%)")
    print(f"   → Likely a feature, not a bug. Investigate meaning.\n")

print("🔧 POSSIBLE CAUSES:")
print("   1. Tracks without explicit program set (default to -1)")
print("   2. Drum tracks (if Program token still generated)")
print("   3. Empty/marker tracks")
print("   4. MidiTok internal representation for 'unknown instrument'\n")

print("📋 RECOMMENDED ACTION:")
if minus1_percentage < 10:
    print("   → FILTER Program -1 in preprocessing")
    print("   → Loss: <10% of program tokens (acceptable)")
    print("   → Benefit: Cleaner data, known instruments only\n")
else:
    print("   → INVESTIGATE FURTHER before filtering")
    print("   → Check MidiTok documentation on Program -1 meaning")
    print("   → Possibly map -1 to default instrument (Piano=0)?\n")

print(f"{'='*70}")
print("✅ INVESTIGATION COMPLETE!")
print(f"{'='*70}\n")