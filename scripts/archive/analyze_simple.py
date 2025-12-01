"""
REMI Token Analysis - Basato sul TUO script che FUNZIONA
"""

from miditok import REMI, TokenizerConfig
from pathlib import Path
from collections import defaultdict, Counter
import statistics

print("\n" + "="*70)
print("🔬 REMI TOKEN ANALYSIS")
print("="*70 + "\n")

# ========================================
# SETUP (identico al tuo script)
# ========================================
print("🎹 Setup REMI Tokenizer\n")

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

print("✅ Tokenizer creato!")
print(f"   Vocab size: {len(tokenizer.vocab)}\n")

# ========================================
# TROVA FILES
# ========================================
lakh_path = Path("data/lakh_clean")
midi_files = list(lakh_path.rglob("*.mid"))[:1000]

print(f"📁 Trovati {len(midi_files)} file MIDI")
print(f"   Analizzo fino a 1000 file...\n")

# ========================================
# ANALIZZA
# ========================================
token_counts = defaultdict(int)
sequence_lengths = []
all_pitches = []
all_velocities = []

success = 0
errors = 0

print("🔄 Processing...\n")

for i, midi_path in enumerate(midi_files):
    if (i + 1) % 100 == 0:
        print(f"   {i+1}/1000 files... ({success} ok, {errors} errors)")
    
    try:
        tokens = tokenizer(str(midi_path))
        
        # Conta lunghezza
        sequence_lengths.append(len(tokens[0]))
        
        # Conta token types
        for token_id in tokens[0]:
            token_name = list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(token_id)]
            
            # Estrai tipo
            if "Pitch" in token_name:
                token_counts["Pitch"] += 1
                pitch_val = int(token_name.split("_")[1])
                all_pitches.append(pitch_val)
            elif "Velocity" in token_name:
                token_counts["Velocity"] += 1
                vel_val = int(token_name.split("_")[1])
                all_velocities.append(vel_val)
            elif "Duration" in token_name:
                token_counts["Duration"] += 1
            elif "Position" in token_name:
                token_counts["Position"] += 1
            elif "Bar" in token_name:
                token_counts["Bar"] += 1
            elif "Tempo" in token_name:
                token_counts["Tempo"] += 1
            elif "TimeSig" in token_name:
                token_counts["TimeSig"] += 1
            elif "Rest" in token_name:
                token_counts["Rest"] += 1
        
        success += 1
        
    except:
        errors += 1

# ========================================
# RISULTATI
# ========================================
print(f"\n{'='*70}")
print("📈 RISULTATI")
print(f"{'='*70}\n")

print(f"✅ Success: {success}")
print(f"❌ Errors: {errors}")
print(f"📊 Success rate: {100*success/(success+errors):.1f}%\n")

if success == 0:
    print("⚠️  Nessun file valido!")
    exit(1)

print(f"{'-'*70}\n")

# TOKEN DISTRIBUTION
print("🎵 TOKEN TYPE DISTRIBUTION:\n")

total = sum(token_counts.values())
print(f"   Total tokens: {total:,}\n")

# Percentuali
token_pcts = {}
for token_type, count in sorted(token_counts.items(), key=lambda x: -x[1]):
    pct = 100 * count / total
    token_pcts[token_type] = pct
    print(f"   {token_type:15s}: {count:10,} ({pct:5.2f}%)")

# Lambda weights
print(f"\n   💡 LAMBDA WEIGHTS:\n")
max_pct = max(token_pcts.values())
lambdas = {}
for token_type, pct in sorted(token_pcts.items()):
    lam = max_pct / pct if pct > 0 else 1.0
    lambdas[token_type] = lam
    print(f"      lambda_{token_type.lower():12s} = {lam:.3f}")

print(f"\n   📋 Python dict:")
print(f"   lambda_weights = {{")
for t, w in sorted(lambdas.items()):
    print(f"       '{t.lower()}': {w:.3f},")
print(f"   }}")

print(f"\n{'-'*70}\n")

# SEQUENCE LENGTHS
print("📏 SEQUENCE LENGTHS:\n")

print(f"   Mean:    {statistics.mean(sequence_lengths):,.0f}")
print(f"   Median:  {statistics.median(sequence_lengths):,.0f}")
print(f"   Min:     {min(sequence_lengths):,.0f}")
print(f"   Max:     {max(sequence_lengths):,.0f}")

pct95 = sorted(sequence_lengths)[int(len(sequence_lengths)*0.95)]
pct99 = sorted(sequence_lengths)[int(len(sequence_lengths)*0.99)]
print(f"   95th:    {pct95:,.0f}")
print(f"   99th:    {pct99:,.0f}")

# Distribuzione
print(f"\n   Length distribution:")
ranges = [
    ("< 512", lambda x: x < 512),
    ("512-1024", lambda x: 512 <= x < 1024),
    ("1024-2048", lambda x: 1024 <= x < 2048),
    ("2048-4096", lambda x: 2048 <= x < 4096),
    ("4096+", lambda x: x >= 4096)
]

for name, cond in ranges:
    count = sum(1 for l in sequence_lengths if cond(l))
    pct = 100 * count / len(sequence_lengths)
    print(f"   {name:12s}: {count:4d} ({pct:5.1f}%)")

# Suggerimento
suggested = ((pct95 // 512) + 1) * 512
print(f"\n   💡 SUGGESTED max_seq_length: {suggested}")

print(f"\n{'-'*70}\n")

# PITCH
if all_pitches:
    print("🎹 PITCH:\n")
    print(f"   Range: {min(all_pitches)}-{max(all_pitches)}")
    print(f"   Unique: {len(set(all_pitches))}")
    
    print(f"\n   Top 10:")
    for pitch, count in Counter(all_pitches).most_common(10):
        note = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][pitch%12]
        octave = (pitch//12)-1
        print(f"      {pitch:3d} ({note}{octave}): {count:6,}")

print(f"\n{'-'*70}\n")

# VELOCITY
if all_velocities:
    print("🔊 VELOCITY:\n")
    print(f"   Range: {min(all_velocities)}-{max(all_velocities)}")
    print(f"   Mean:  {statistics.mean(all_velocities):.1f}")
    print(f"   Unique: {len(set(all_velocities))}")

print(f"\n{'='*70}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*70}\n")

print("📋 NEXT STEPS:")
print("   1. Copy lambda_weights to model config")
print("   2. Set max_seq_length in model config")
print("   3. Start preprocessing\n")