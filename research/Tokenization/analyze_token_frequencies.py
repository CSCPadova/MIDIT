"""
Analizza frequency distribution per ogni head.
Questo ci dirà ESATTAMENTE quanto è grave l'imbalance.
"""
import torch
from pathlib import Path
from tqdm import tqdm
from miditok import REMI, TokenizerConfig
from collections import Counter, defaultdict

# Tokenizer
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
tokenizer = REMI(tokenizer_config=config)
id_to_token = {v: k for k, v in tokenizer.vocab.items()}

print("=" * 80)
print("📊 TOKEN FREQUENCY ANALYSIS")
print("=" * 80)

# Analyze dataset
data_dir = Path("data/preprocessed_phase1")
all_files = sorted(data_dir.glob("*.pt"))

print(f"\n📂 Analyzing {len(all_files)} files...")

# Count tokens by type
frequencies = defaultdict(Counter)
total_tokens = Counter()

for file_path in tqdm(all_files[:1000], desc="Analyzing"):  # Sample 1000 files
    try:
        data = torch.load(file_path)
        tokens = data['tokens']
        
        for token_id in tokens.tolist():
            if token_id not in id_to_token:
                continue
            
            token_name = id_to_token[token_id]
            
            # Classify by head
            if token_name.startswith('Pitch_') or token_name.startswith('PitchDrum_'):
                frequencies['pitch'][token_id] += 1
                total_tokens['pitch'] += 1
            elif token_name.startswith('Velocity_'):
                frequencies['velocity'][token_id] += 1
                total_tokens['velocity'] += 1
            elif token_name.startswith('Duration_'):
                frequencies['duration'][token_id] += 1
                total_tokens['duration'] += 1
            elif token_name.startswith('Program_'):
                frequencies['program'][token_id] += 1
                total_tokens['program'] += 1
            elif token_name.startswith('Position_'):
                frequencies['position'][token_id] += 1
                total_tokens['position'] += 1
            elif token_name.startswith('Bar_'):
                frequencies['bar'][token_id] += 1
                total_tokens['bar'] += 1
            elif token_name.startswith('Tempo_'):
                frequencies['tempo'][token_id] += 1
                total_tokens['tempo'] += 1
            elif token_name.startswith('TimeSig_'):
                frequencies['timesig'][token_id] += 1
                total_tokens['timesig'] += 1
            elif token_name.startswith('Rest_'):
                frequencies['rest'][token_id] += 1
                total_tokens['rest'] += 1
                
    except Exception as e:
        continue

# Analyze imbalance
print("\n" + "=" * 80)
print("📈 FREQUENCY DISTRIBUTION PER HEAD:")
print("=" * 80)

for head_name in sorted(frequencies.keys()):
    freqs = frequencies[head_name]
    total = total_tokens[head_name]
    
    if len(freqs) == 0:
        continue
    
    # Stats
    most_common = freqs.most_common(1)[0]
    least_common = freqs.most_common()[-1]
    
    imbalance_ratio = most_common[1] / least_common[1]
    
    print(f"\n{head_name.upper()}:")
    print(f"   Total tokens: {total:>10,}")
    print(f"   Unique values: {len(freqs):>3}")
    print(f"   Most common: {id_to_token[most_common[0]]:20} ({most_common[1]:>8,} = {most_common[1]/total*100:>5.1f}%)")
    print(f"   Least common: {id_to_token[least_common[0]]:20} ({least_common[1]:>8,} = {least_common[1]/total*100:>5.1f}%)")
    print(f"   Imbalance ratio: {imbalance_ratio:>8.1f}×")
    
    # Top 5
    print(f"   Top 5:")
    for token_id, count in freqs.most_common(5):
        print(f"      {id_to_token[token_id]:25} {count:>8,} ({count/total*100:>5.1f}%)")
    
    # Check if 80% mass in top N tokens
    cumsum = 0
    for i, (token_id, count) in enumerate(freqs.most_common()):
        cumsum += count
        if cumsum >= total * 0.8:
            print(f"   ⚠️ 80% mass in top {i+1} tokens (out of {len(freqs)})")
            break

# Compute suggested weights
print("\n" + "=" * 80)
print("💡 SUGGESTED LOSS WEIGHTS:")
print("=" * 80)

print("\nBased on imbalance + importance:")
print("   pitch:    1.0  (difficult, important)")
print("   velocity: 0.5  (easy, repetitive)")
print("   duration: 0.8  (moderate)")
print("   program:  1.0  (important)")
print("   position: 0.2  (very easy)")
print("   bar:      0.3  (trivial)")
print("   tempo:    1.5  (rare, critical)")
print("   timesig:  1.5  (rare, critical)")
print("   rest:     0.5  (moderate)")

# Save frequencies
torch.save({
    'frequencies': dict(frequencies),
    'total_tokens': dict(total_tokens)
}, 'token_frequencies.pt')

print("\n💾 Saved: token_frequencies.pt")
print("=" * 80)
