"""
Calcola tokens per N bars
"""

from miditok import REMI, TokenizerConfig
from pathlib import Path
from collections import defaultdict
import statistics

print("\n" + "="*70)
print("📊 TOKENS PER BAR ANALYSIS")
print("="*70 + "\n")

# Setup tokenizer
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

# Load files
lakh_path = Path("data/lakh_clean")
midi_files = list(lakh_path.rglob("*.mid"))[:100]  # 100 files per velocità

print(f"📁 Analyzing {len(midi_files)} files...\n")

# Collect data
tokens_per_bar = []  # Token tra ogni coppia di Bar consecutivi
bars_per_song = []

for i, midi_path in enumerate(midi_files):
    if (i + 1) % 20 == 0:
        print(f"   Progress: {i+1}/{len(midi_files)}")
    
    try:
        tokens = tokenizer(str(midi_path))
        
        # Handle format
        if isinstance(tokens, list):
            all_ids = []
            for tok_seq in tokens:
                all_ids.extend(tok_seq.ids)
        else:
            all_ids = tokens.ids
        
        # Find Bar positions
        bar_positions = []
        for idx, token_id in enumerate(all_ids):
            token_name = list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(token_id)]
            if "Bar" in token_name:
                bar_positions.append(idx)
        
        if len(bar_positions) < 2:
            continue
        
        bars_per_song.append(len(bar_positions))
        
        # Calculate tokens between consecutive bars
        for j in range(len(bar_positions) - 1):
            tokens_in_bar = bar_positions[j+1] - bar_positions[j]
            tokens_per_bar.append(tokens_in_bar)
    
    except:
        pass

# Calculate statistics
print(f"\n{'='*70}")
print("📈 RESULTS")
print(f"{'='*70}\n")

print(f"✅ Files analyzed: {len(bars_per_song)}")
print(f"📊 Total bars: {sum(bars_per_song)}")
print(f"📊 Bars per song (mean): {statistics.mean(bars_per_song):.1f}\n")

print(f"{'-'*70}\n")
print(f"🎵 TOKENS PER BAR:\n")

tokens_per_bar_mean = statistics.mean(tokens_per_bar)
tokens_per_bar_median = statistics.median(tokens_per_bar)

print(f"   Mean:   {tokens_per_bar_mean:.1f} tokens/bar")
print(f"   Median: {tokens_per_bar_median:.1f} tokens/bar")
print(f"   Min:    {min(tokens_per_bar)} tokens/bar")
print(f"   Max:    {max(tokens_per_bar)} tokens/bar\n")

print(f"{'-'*70}\n")
print(f"📏 TOKENS FOR N BARS:\n")

# Calculate for different bar counts
bar_counts = [4, 8, 16, 32, 64]
for n_bars in bar_counts:
    tokens = n_bars * tokens_per_bar_mean
    print(f"   {n_bars:3d} bars → ~{tokens:,.0f} tokens")

print(f"\n💡 SUGGESTED CHUNK SIZE:\n")

target_tokens = 8192
suggested_bars = target_tokens / tokens_per_bar_mean
print(f"   For max_seq_length = 8192:")
print(f"   → ~{suggested_bars:.0f} bars per chunk")

print(f"\n{'='*70}\n")