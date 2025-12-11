"""
Tokens Per Bar Analysis V2
- 500 files sample
- Outlier filtering
- Detailed percentiles
- Bar-based chunk recommendations
"""

from miditok import REMI, TokenizerConfig
from pathlib import Path
from collections import defaultdict
import statistics

print("\n" + "="*70)
print("📊 TOKENS PER BAR ANALYSIS V2")
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

print("✅ Tokenizer created")
print(f"   Vocab size: {len(tokenizer.vocab)}\n")

# ========================================
# Load Files
# ========================================
lakh_path = Path("data/lakh_clean")
midi_files = list(lakh_path.rglob("*.mid"))[:500]  # 500 files

print(f"📁 Analyzing {len(midi_files)} files...\n")

# ========================================
# Collect Data
# ========================================
tokens_per_bar_raw = []       # Tutti i dati (con outliers)
tokens_per_bar_filtered = []  # Dati filtrati (senza outliers)
bars_per_song = []
outliers_found = 0

success = 0
errors = 0

print("🔄 Processing...\n")

for i, midi_path in enumerate(midi_files):
    if (i + 1) % 50 == 0:
        print(f"   Progress: {i+1}/{len(midi_files)} ({success} ok, {errors} errors)")
    
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
            errors += 1
            continue
        
        bars_per_song.append(len(bar_positions))
        
        # Calculate tokens between consecutive bars
        for j in range(len(bar_positions) - 1):
            tokens_in_bar = bar_positions[j+1] - bar_positions[j]
            
            # Aggiungi a raw (tutti)
            tokens_per_bar_raw.append(tokens_in_bar)
            
            # Filter outliers (>1000 tokens/bar = anomalia)
            if tokens_in_bar <= 1000:
                tokens_per_bar_filtered.append(tokens_in_bar)
            else:
                outliers_found += 1
        
        success += 1
        
    except Exception as e:
        errors += 1
        # Uncomment to debug:
        # print(f"\n   Error on {midi_path.name}: {e}")

# ========================================
# RESULTS
# ========================================
print(f"\n{'='*70}")
print("📈 RESULTS")
print(f"{'='*70}\n")

print(f"✅ Files successfully analyzed: {success}")
print(f"❌ Files with errors: {errors}")
print(f"📊 Total bars analyzed: {sum(bars_per_song)}")
print(f"📊 Bars per song (mean): {statistics.mean(bars_per_song):.1f}")
print(f"🚨 Outliers found (>1000 tokens/bar): {outliers_found}\n")

print(f"{'-'*70}\n")

# ========================================
# RAW STATISTICS (with outliers)
# ========================================
print(f"🎵 TOKENS PER BAR (RAW - with outliers):\n")

mean_raw = statistics.mean(tokens_per_bar_raw)
median_raw = statistics.median(tokens_per_bar_raw)

print(f"   Mean:   {mean_raw:.1f} tokens/bar")
print(f"   Median: {median_raw:.1f} tokens/bar")
print(f"   Min:    {min(tokens_per_bar_raw)} tokens/bar")
print(f"   Max:    {max(tokens_per_bar_raw)} tokens/bar")

print(f"\n   ⚠️  Mean skewed by {outliers_found} outliers!\n")

print(f"{'-'*70}\n")

# ========================================
# FILTERED STATISTICS (without outliers)
# ========================================
print(f"🎵 TOKENS PER BAR (FILTERED - outliers removed):\n")

mean_filtered = statistics.mean(tokens_per_bar_filtered)
median_filtered = statistics.median(tokens_per_bar_filtered)
stdev_filtered = statistics.stdev(tokens_per_bar_filtered)

print(f"   Mean:   {mean_filtered:.1f} tokens/bar")
print(f"   Median: {median_filtered:.1f} tokens/bar")
print(f"   StdDev: {stdev_filtered:.1f} tokens/bar")
print(f"   Min:    {min(tokens_per_bar_filtered)} tokens/bar")
print(f"   Max:    {max(tokens_per_bar_filtered)} tokens/bar")

# Percentiles
sorted_data = sorted(tokens_per_bar_filtered)
p25 = sorted_data[int(len(sorted_data) * 0.25)]
p50 = sorted_data[int(len(sorted_data) * 0.50)]
p75 = sorted_data[int(len(sorted_data) * 0.75)]
p90 = sorted_data[int(len(sorted_data) * 0.90)]
p95 = sorted_data[int(len(sorted_data) * 0.95)]

print(f"\n   Percentiles:")
print(f"   25th: {p25} tokens/bar")
print(f"   50th: {p50} tokens/bar (median)")
print(f"   75th: {p75} tokens/bar")
print(f"   90th: {p90} tokens/bar")
print(f"   95th: {p95} tokens/bar")

print(f"\n{'-'*70}\n")

# ========================================
# TOKENS FOR N BARS (using filtered mean)
# ========================================
print(f"📏 TOKENS FOR N BARS (using filtered mean={mean_filtered:.1f}):\n")

bar_counts = [4, 8, 16, 24, 32, 48, 64]
for n_bars in bar_counts:
    tokens_mean = n_bars * mean_filtered
    tokens_p75 = n_bars * p75  # Worst case (75th percentile)
    
    fit_8k = "✅" if tokens_mean <= 8192 else "❌"
    
    print(f"   {n_bars:3d} bars → ~{tokens_mean:>6,.0f} tokens (mean)  "
          f"| ~{tokens_p75:>6,.0f} tokens (75th)  {fit_8k}")

print(f"\n{'-'*70}\n")

# ========================================
# CHUNK SIZE RECOMMENDATIONS
# ========================================
print(f"💡 CHUNK SIZE RECOMMENDATIONS:\n")

target_tokens = 8192

# Conservative (75th percentile)
suggested_bars_p75 = int(target_tokens / p75)
tokens_at_p75 = suggested_bars_p75 * p75

# Moderate (mean)
suggested_bars_mean = int(target_tokens / mean_filtered)
tokens_at_mean = suggested_bars_mean * mean_filtered

# Aggressive (median)
suggested_bars_median = int(target_tokens / median_filtered)
tokens_at_median = suggested_bars_median * median_filtered

print(f"   Target: max_seq_length = 8192 tokens\n")

print(f"   🟢 CONSERVATIVE (75th percentile = {p75} tokens/bar):")
print(f"      → {suggested_bars_p75} bars per chunk")
print(f"      → ~{tokens_at_p75:,.0f} tokens/chunk")
print(f"      → Fits 75% of bars safely\n")

print(f"   🟡 MODERATE (mean = {mean_filtered:.0f} tokens/bar):")
print(f"      → {suggested_bars_mean} bars per chunk")
print(f"      → ~{tokens_at_mean:,.0f} tokens/chunk")
print(f"      → Balanced approach\n")

print(f"   🔴 AGGRESSIVE (median = {median_filtered:.0f} tokens/bar):")
print(f"      → {suggested_bars_median} bars per chunk")
print(f"      → ~{tokens_at_median:,.0f} tokens/chunk")
print(f"      → May exceed 8192 for dense bars\n")

print(f"{'-'*70}\n")

# ========================================
# MUSICAL CONTEXT
# ========================================
print(f"🎼 MUSICAL CONTEXT:\n")

phrase_bars = 4
verse_bars = 16
chorus_bars = 16

print(f"   Typical structures:")
print(f"   - Phrase:  {phrase_bars} bars  → ~{phrase_bars * mean_filtered:,.0f} tokens")
print(f"   - Verse:   {verse_bars} bars → ~{verse_bars * mean_filtered:,.0f} tokens")
print(f"   - Chorus:  {chorus_bars} bars → ~{chorus_bars * mean_filtered:,.0f} tokens\n")

if suggested_bars_mean >= 16:
    print(f"   ✅ Recommended {suggested_bars_mean} bars can capture full Verse/Chorus!")
elif suggested_bars_mean >= 8:
    print(f"   ⚠️  Recommended {suggested_bars_mean} bars captures half Verse/Chorus")
else:
    print(f"   ❌ Recommended {suggested_bars_mean} bars too short for sections")

print(f"\n{'='*70}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*70}\n")

print("📋 RECOMMENDATIONS:")
print(f"   1. Use BAR-BASED chunking with {suggested_bars_mean} bars")
print(f"   2. Overlap: 2-4 bars (maintains musical continuity)")
print(f"   3. Max tokens per chunk: ~{tokens_at_mean:,.0f} (fits 8192 limit)")
print(f"   4. Coverage: Full Verse or Chorus per chunk\n")

print("🧪 EXPERIMENT SETUP:")
print(f"""
Experiment A: Token-based
├─ chunk_size: 8192 tokens
├─ overlap: 512 tokens
└─ Pro: Uniform length, Contra: May split bars

Experiment B: Bar-based (RECOMMENDED)
├─ chunk_size: {suggested_bars_mean} bars
├─ overlap: 2 bars
└─ Pro: Respects structure, Contra: Variable length (~{tokens_at_mean-1000:.0f}-{tokens_at_mean+1000:.0f})
""")