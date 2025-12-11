"""
REMI Token Analysis - DEBUG VERSION
Shows ALL errors to identify the problem.
"""

from pathlib import Path
from miditok import REMI, TokenizerConfig
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("🔬 DEBUG MODE - REMI TOKEN ANALYSIS")
print("="*70 + "\n")

# ========================================
# Setup REMI+ Tokenizer
# ========================================
print("📊 Setting up tokenizer...")
config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8},
    use_programs=True,
    one_token_stream_for_programs=True,
    use_time_signatures=True,
    use_tempos=True,
    use_chords=False,
    special_tokens=["PAD", "BOS", "EOS", "MASK"]
)

tokenizer = REMI(config)
print(f"   ✅ Tokenizer created")
print(f"   Vocabulary size: {len(tokenizer.vocab)}\n")

# ========================================
# Test on FIRST 5 files
# ========================================
dataset_path = Path("data/lakh_clean")
if not dataset_path.exists():
    dataset_path = Path("data/midi_files/lmd_full")

if not dataset_path.exists():
    print("❌ ERROR: No dataset found!")
    print(f"   Tried: data/lakh_clean")
    print(f"   Tried: data/midi_files/lmd_full")
    exit(1)

midi_files = list(dataset_path.rglob("*.mid"))[:5]
print(f"📁 Found {len(midi_files)} MIDI files for testing")
print(f"   Testing FIRST 5 files...\n")

# ========================================
# Test each file individually
# ========================================
for i, midi_path in enumerate(midi_files, 1):
    print(f"{'='*70}")
    print(f"TEST {i}/5: {midi_path.name}")
    print(f"{'='*70}")
    print(f"Path: {midi_path}")
    
    try:
        # Test 1: File exists?
        print(f"\n✓ File exists: {midi_path.exists()}")
        print(f"✓ File size: {midi_path.stat().st_size / 1024:.2f} KB")
        
        # Test 2: Tokenize
        print(f"\n🔄 Tokenizing...")
        tokens = tokenizer(str(midi_path))
        
        # Test 3: Check tokens
        print(f"✅ SUCCESS!")
        print(f"   Tokens: {len(tokens.ids)} total")
        print(f"   First 10 tokens: {tokens.tokens[:10]}")
        
    except Exception as e:
        print(f"\n❌ FAILED!")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        
        # Print full traceback
        import traceback
        print(f"\n   Full traceback:")
        traceback.print_exc()
    
    print()

print("="*70)
print("DEBUG COMPLETE")
print("="*70)