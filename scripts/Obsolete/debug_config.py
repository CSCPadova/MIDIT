"""
Debug: Quale parametro rompe la tokenizzazione?
"""

from miditok import REMI, TokenizerConfig
from pathlib import Path

print("\n" + "="*70)
print("🔍 DEBUG - Trova il parametro che rompe")
print("="*70 + "\n")

# Un file di test
test_file = list(Path("data/lakh_clean").rglob("*.mid"))[0]
print(f"📁 Test file: {test_file.name}\n")

# ========================================
# TEST 1: Config base (che funziona)
# ========================================
print("TEST 1: Config base (V1 - funzionava)")
print("-" * 70)

config1 = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8},
    num_velocities=32,
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
)

try:
    tokenizer1 = REMI(config1)
    tokens1 = tokenizer1(str(test_file))
    print(f"✅ SUCCESS - {len(tokens1[0])} tokens")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# ========================================
# TEST 2: + special_tokens
# ========================================
print("TEST 2: Config base + special_tokens")
print("-" * 70)

config2 = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8},
    num_velocities=32,
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    special_tokens=["PAD", "BOS", "EOS", "MASK"]  # ← AGGIUNTO
)

try:
    tokenizer2 = REMI(config2)
    tokens2 = tokenizer2(str(test_file))
    print(f"✅ SUCCESS - {len(tokens2[0])} tokens")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# ========================================
# TEST 3: + use_programs
# ========================================
print("TEST 3: Config base + use_programs")
print("-" * 70)

config3 = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8},
    num_velocities=32,
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_programs=True,  # ← AGGIUNTO
)

try:
    tokenizer3 = REMI(config3)
    tokens3 = tokenizer3(str(test_file))
    print(f"✅ SUCCESS - {len(tokens3[0])} tokens")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# ========================================
# TEST 4: + entrambi
# ========================================
print("TEST 4: Config base + special_tokens + use_programs")
print("-" * 70)

config4 = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8},
    num_velocities=32,
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_programs=True,  # ← AGGIUNTO
    special_tokens=["PAD", "BOS", "EOS", "MASK"]  # ← AGGIUNTO
)

try:
    tokenizer4 = REMI(config4)
    tokens4 = tokenizer4(str(test_file))
    print(f"✅ SUCCESS - {len(tokens4[0])} tokens")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

print("="*70)
print("CONCLUSIONE:")
print("="*70)
print("Guarda sopra quale TEST ha funzionato e quale no.")
print("Questo ci dice quale parametro causa il problema.\n")