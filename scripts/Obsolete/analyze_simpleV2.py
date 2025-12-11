"""
REMI Token Analysis V3 - Con gestione corretta use_programs
"""

from miditok import REMI, TokenizerConfig
from pathlib import Path
from collections import defaultdict, Counter
import statistics

print("\n" + "="*70)
print("🔬 REMI TOKEN ANALYSIS V3")
print("="*70 + "\n")

# ========================================
# SETUP - Con programs + special tokens
# ========================================
print("🎹 Setup REMI+ Tokenizer\n")

config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8},
    num_velocities=32,
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_programs=True,                           # ← Multi-instrument
    special_tokens=["PAD", "BOS", "EOS", "MASK"] # ← MLM tokens
)

tokenizer = REMI(config)

print("✅ Tokenizer creato!")
print(f"   Vocab size: {len(tokenizer.vocab)}")
print(f"   Multi-instrument: {config.use_programs}")
print(f"   Special tokens: {config.special_tokens}")
print(f"   One token stream: {tokenizer.one_token_stream}\n")

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
all_programs = []

success = 0
errors = 0

print("🔄 Processing...\n")

for i, midi_path in enumerate(midi_files):
    if (i + 1) % 100 == 0:
        print(f"   {i+1}/1000 files... ({success} ok, {errors} errors)")
    
    try:
        tokens = tokenizer(str(midi_path))
        
        # ========================================
        # GESTIONE FORMATO OUTPUT
        # ========================================
        # Con use_programs=True → tokens è un TokSequence
        # Con use_programs=False → tokens è una lista di TokSequence
        
        if isinstance(tokens, list):
            # Multi-track: lista di TokSequence
            all_token_ids = []
            for tok_seq in tokens:
                all_token_ids.extend(tok_seq.ids)
        else:
            # Single stream: tokens è già un TokSequence
            all_token_ids = tokens.ids
        
        # Conta lunghezza
        sequence_lengths.append(len(all_token_ids))
        
        # Conta token types
        for token_id in all_token_ids:
            token_name = list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(token_id)]
            
            # Estrai tipo
            if "Pitch" in token_name:
                token_counts["Pitch"] += 1
                try:
                    pitch_val = int(token_name.split("_")[1])
                    all_pitches.append(pitch_val)
                except:
                    pass
                    
            elif "Velocity" in token_name:
                token_counts["Velocity"] += 1
                try:
                    vel_val = int(token_name.split("_")[1])
                    all_velocities.append(vel_val)
                except:
                    pass
                    
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
                
            elif "Program" in token_name:
                token_counts["Program"] += 1
                try:
                    prog_val = int(token_name.split("_")[1])
                    all_programs.append(prog_val)
                except:
                    pass
        
        success += 1
        
    except Exception as e:
        errors += 1
        # Uncomment per debug:
        # print(f"\n   Error on {midi_path.name}: {e}")

# ========================================
# RISULTATI
# ========================================
print(f"\n{'='*70}")
print("📈 RISULTATI")
print(f"{'='*70}\n")

print(f"✅ Success: {success}")
print(f"❌ Errors: {errors}")

if success > 0:
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

print(f"\n{'-'*70}\n")

# PROGRAMS
if all_programs:
    print("🎸 INSTRUMENTS (PROGRAMS):\n")
    print(f"   Total Program tokens: {len(all_programs):,}")
    print(f"   Unique instruments: {len(set(all_programs))}")
    
    print(f"\n   Top 10 instruments:")
    # General MIDI names
    gm_names = {
        0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano",
        24: "Acoustic Guitar (nylon)", 25: "Acoustic Guitar (steel)",
        32: "Acoustic Bass", 33: "Electric Bass (finger)",
        48: "String Ensemble 1", 49: "String Ensemble 2",
        73: "Flute", 80: "Lead 1 (square)"
    }
    
    for prog, count in Counter(all_programs).most_common(10):
        prog_name = gm_names.get(prog, f"Program {prog}")
        print(f"      {prog:3d} ({prog_name}): {count:6,}")

print(f"\n{'='*70}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*70}\n")

print("📋 CONFIG FINALE:")
print("""
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
""")

print(f"lambda_weights = {{")
for t, w in sorted(lambdas.items()):
    print(f"    '{t.lower()}': {w:.3f},")
print(f"}}")

print(f"\nmax_seq_length = {suggested}")

print("\n📋 NEXT STEPS:")
print("   1. Usa questo config nel model code")
print("   2. Implementa multi-head output architecture")
print("   3. Start preprocessing pipeline\n")