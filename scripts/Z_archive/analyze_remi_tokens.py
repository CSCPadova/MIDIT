"""
REMI Token Analysis - Using Working Config
Analyzes token distribution to calculate lambda_weights and max_seq_length.
"""

from pathlib import Path
from collections import Counter, defaultdict
import statistics
from miditok import REMI, TokenizerConfig
import warnings

warnings.filterwarnings('ignore')


def analyze_token_sequence(tokens, tokenizer):
    """Analyze a token sequence and extract statistics."""
    
    token_types = defaultdict(int)
    pitch_values = []
    velocity_values = []
    
    for token_id in tokens.ids:
        token_str = tokenizer.vocab[token_id]
        
        if "Pitch" in token_str:
            token_types["Pitch"] += 1
            try:
                pitch_values.append(int(token_str.split("_")[1]))
            except:
                pass
                
        elif "Velocity" in token_str:
            token_types["Velocity"] += 1
            try:
                velocity_values.append(int(token_str.split("_")[1]))
            except:
                pass
                
        elif "Duration" in token_str:
            token_types["Duration"] += 1
            
        elif "Position" in token_str:
            token_types["Position"] += 1
            
        elif "Bar" in token_str:
            token_types["Bar"] += 1
            
        elif "Tempo" in token_str:
            token_types["Tempo"] += 1
            
        elif "Program" in token_str:
            token_types["Program"] += 1
            
        elif "TimeSig" in token_str:
            token_types["TimeSig"] += 1
        
        elif "Rest" in token_str:
            token_types["Rest"] += 1
    
    return {
        'length': len(tokens.ids),
        'token_types': dict(token_types),
        'pitch_values': pitch_values,
        'velocity_values': velocity_values,
    }


def analyze_dataset(dataset_path, dataset_name, max_files=1000):
    """Run token-level analysis on a dataset."""
    
    print(f"\n{'='*70}")
    print(f"🎹 {dataset_name.upper()} - REMI TOKEN ANALYSIS")
    print(f"{'='*70}\n")
    
    # ========================================
    # Setup REMI Tokenizer
    # Using the SAME config as your working script!
    # ========================================
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8},
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
    )
    
    tokenizer = REMI(config)
    print(f"📊 REMI Configuration (same as your working script):")
    print(f"   Vocabulary size: {len(tokenizer.vocab)}")
    print(f"   Pitch range: {config.pitch_range}")
    print(f"   Beat resolution: {config.beat_res}")
    print(f"   Num velocities: {config.num_velocities}")
    print(f"   Use chords: {config.use_chords}")
    print(f"   Use rests: {config.use_rests}\n")
    
    # ========================================
    # Find MIDI Files
    # ========================================
    midi_files = list(Path(dataset_path).rglob("*.mid"))[:max_files]
    print(f"📁 Found {len(midi_files)} MIDI files")
    print(f"   Analyzing up to {max_files} files...\n")
    
    # ========================================
    # Initialize Statistics
    # ========================================
    all_analyses = []
    sequence_lengths = []
    token_type_totals = defaultdict(int)
    all_pitch_values = []
    all_velocity_values = []
    
    success_count = 0
    error_count = 0
    
    # ========================================
    # Process Files
    # ========================================
    print("🔄 Processing files...")
    for i, midi_path in enumerate(midi_files):
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i+1}/{len(midi_files)} ({success_count} success, {error_count} errors)")
        
        try:
            # Tokenize directly from file path (same as your script)
            tokens = tokenizer(str(midi_path))
            
            # Analyze tokens
            analysis = analyze_token_sequence(tokens, tokenizer)
            
            all_analyses.append(analysis)
            sequence_lengths.append(analysis['length'])
            
            for token_type, count in analysis['token_types'].items():
                token_type_totals[token_type] += count
            
            all_pitch_values.extend(analysis['pitch_values'])
            all_velocity_values.extend(analysis['velocity_values'])
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            # Uncomment to debug:
            # print(f"\n   ❌ Error on {midi_path.name}: {e}")
    
    # ========================================
    # Print Results
    # ========================================
    print(f"\n{'='*70}")
    print("📈 RESULTS")
    print(f"{'='*70}\n")
    
    print(f"✅ Successfully tokenized: {success_count}")
    print(f"❌ Errors: {error_count}")
    
    if success_count > 0:
        print(f"📊 Success rate: {100*success_count/(success_count+error_count):.1f}%\n")
    
    if not all_analyses:
        print("⚠️  No valid files to analyze!")
        return
    
    print(f"{'-'*70}\n")
    
    # ========================================
    # TOKEN TYPE DISTRIBUTION
    # ========================================
    print("🎵 TOKEN TYPE DISTRIBUTION:")
    
    total_tokens = sum(token_type_totals.values())
    print(f"\n   Total tokens analyzed: {total_tokens:,}\n")
    
    token_percentages = {}
    for token_type, count in sorted(token_type_totals.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_tokens
        token_percentages[token_type] = pct
        print(f"   {token_type:15s}: {count:10,} tokens ({pct:5.2f}%)")
    
    # ========================================
    # LAMBDA WEIGHTS (Multi-head loss)
    # ========================================
    print(f"\n   💡 SUGGESTED LAMBDA WEIGHTS (for multi-head loss):")
    print(f"   (Using inverse frequency method from Compound Word paper)\n")
    
    if token_percentages:
        max_pct = max(token_percentages.values())
        lambda_dict = {}
        for token_type, pct in sorted(token_percentages.items(), key=lambda x: -x[1]):
            lambda_weight = max_pct / pct if pct > 0 else 1.0
            lambda_dict[token_type] = lambda_weight
            print(f"      lambda_{token_type.lower():12s} = {lambda_weight:.3f}")
        
        print(f"\n   📋 Python dict for model config:")
        print(f"   lambda_weights = {{")
        for token_type, weight in sorted(lambda_dict.items()):
            print(f"       '{token_type.lower()}': {weight:.3f},")
        print(f"   }}")
    
    print(f"\n{'-'*70}\n")
    
    # ========================================
    # SEQUENCE LENGTHS
    # ========================================
    print("📏 SEQUENCE LENGTH STATISTICS:")
    print(f"\n   Mean:      {statistics.mean(sequence_lengths):,.0f} tokens")
    print(f"   Median:    {statistics.median(sequence_lengths):,.0f} tokens")
    
    if len(sequence_lengths) > 1:
        print(f"   Std Dev:   {statistics.stdev(sequence_lengths):,.0f} tokens")
    
    print(f"   Min:       {min(sequence_lengths):,.0f} tokens")
    print(f"   Max:       {max(sequence_lengths):,.0f} tokens")
    
    if len(sequence_lengths) > 1:
        pct_95 = sorted(sequence_lengths)[int(len(sequence_lengths)*0.95)]
        pct_99 = sorted(sequence_lengths)[int(len(sequence_lengths)*0.99)]
        print(f"   95th pct:  {pct_95:,.0f} tokens")
        print(f"   99th pct:  {pct_99:,.0f} tokens")
    
    print(f"\n   📊 Length distribution:")
    ranges = [
        ("< 512", lambda x: x < 512),
        ("512-1024", lambda x: 512 <= x < 1024),
        ("1024-2048", lambda x: 1024 <= x < 2048),
        ("2048-4096", lambda x: 2048 <= x < 4096),
        ("4096-8192", lambda x: 4096 <= x < 8192),
        ("> 8192", lambda x: x >= 8192)
    ]
    
    for range_name, condition in ranges:
        count = sum(1 for l in sequence_lengths if condition(l))
        pct = 100 * count / len(sequence_lengths)
        print(f"   {range_name:12s}: {count:5d} files ({pct:5.1f}%)")
    
    if len(sequence_lengths) > 1:
        # Round up to nearest multiple of 512
        suggested_95 = ((pct_95 // 512) + 1) * 512
        suggested_99 = ((pct_99 // 512) + 1) * 512
        
        print(f"\n   💡 SUGGESTED max_seq_length for model:")
        print(f"      Conservative (covers 95%): {suggested_95}")
        print(f"      Aggressive (covers 99%):   {suggested_99}")
        print(f"\n      Recommendation: Start with {suggested_95}, increase if needed")
    
    print(f"\n{'-'*70}\n")
    
    # ========================================
    # PITCH ANALYSIS
    # ========================================
    if all_pitch_values:
        print("🎹 PITCH RANGE:")
        print(f"\n   Configured range: {config.pitch_range[0]}-{config.pitch_range[1]}")
        print(f"   Actual range:     {min(all_pitch_values)}-{max(all_pitch_values)}")
        print(f"   Unique pitches:   {len(set(all_pitch_values))}")
        
        # Check if any pitches outside configured range
        outside_range = [p for p in all_pitch_values if p < config.pitch_range[0] or p > config.pitch_range[1]]
        if outside_range:
            print(f"\n   ⚠️  WARNING: {len(outside_range)} notes outside configured range!")
        else:
            print(f"\n   ✅ All pitches within configured range")
        
        print(f"\n   Most common pitches:")
        pitch_counter = Counter(all_pitch_values)
        for pitch, count in pitch_counter.most_common(10):
            midi_note = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][pitch % 12]
            octave = (pitch // 12) - 1
            print(f"      {pitch:3d} ({midi_note}{octave:2d}): {count:8,} times")
    
    print(f"\n{'-'*70}\n")
    
    # ========================================
    # VELOCITY ANALYSIS
    # ========================================
    if all_velocity_values:
        print("🔊 VELOCITY STATISTICS:")
        print(f"\n   Range:         {min(all_velocity_values)}-{max(all_velocity_values)}")
        print(f"   Mean:          {statistics.mean(all_velocity_values):.1f}")
        print(f"   Median:        {statistics.median(all_velocity_values):.1f}")
        
        if len(all_velocity_values) > 1:
            print(f"   Std Dev:       {statistics.stdev(all_velocity_values):.1f}")
        
        print(f"   Unique values: {len(set(all_velocity_values))}")
    
    print()


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("🔬 MIDIT - REMI TOKEN ANALYSIS")
    print("="*70)
    print("\nPurpose: Calculate lambda_weights and max_seq_length for model config")
    print("Using the SAME tokenizer config as your working script\n")
    
    # ========================================
    # Analyze Dataset
    # ========================================
    # Try Lakh Clean first (you showed this works)
    lakh_path = Path("data/lakh_clean")
    if lakh_path.exists():
        analyze_dataset(lakh_path, "Lakh Clean", max_files=1000)
    else:
        print(f"\n⚠️  Lakh Clean path not found: {lakh_path}")
    
    # Then MidiCaps if available
    midicaps_path = Path("data/midi_files/lmd_full")
    if midicaps_path.exists():
        analyze_dataset(midicaps_path, "MidiCaps", max_files=1000)
    else:
        print(f"\n⚠️  MidiCaps path not found: {midicaps_path}")
    
    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70 + "\n")
    
    print("📋 NEXT STEPS:")
    print("   1. Copy lambda_weights dict to model config")
    print("   2. Set max_seq_length in model config")
    print("   3. Proceed with preprocessing pipeline\n")