"""
Debug: Test tokenization su 1 file (FIXED for MidiTok v3.0+)
"""

from pathlib import Path
from miditok import REMI, TokenizerConfig
import symusic

# Setup tokenizer
config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8},
    use_programs=True,
    use_time_signatures=True,
    use_tempos=True,
    use_chords=False,
    special_tokens=["PAD", "BOS", "EOS", "MASK"]
)

tokenizer = REMI(config)
print(f"✅ Tokenizer created, vocab size: {len(tokenizer.vocab)}\n")

# Trova primo MIDI
midi_path = list(Path("data/lakh_clean").rglob("*.mid"))[0]
print(f"📁 Testing file: {midi_path.name}\n")

try:
    # Step 1: Load MIDI with symusic
    print("Step 1: Loading MIDI with symusic...")
    midi = symusic.Score(str(midi_path))
    print(f"   ✅ Loaded: {len(midi.tracks)} tracks")
    
    # Step 2: Filter drums
    print("\nStep 2: Filtering drums...")
    non_drum_tracks = [track for track in midi.tracks if not track.is_drum]
    print(f"   ✅ After filter: {len(non_drum_tracks)} tracks")
    
    if len(non_drum_tracks) == 0:
        print("   ❌ No non-drum tracks!")
        exit()
    
    # Step 3: Check notes
    print("\nStep 3: Checking notes...")
    total_notes = sum(len(track.notes) for track in non_drum_tracks)
    print(f"   ✅ Total notes: {total_notes}")
    
    if total_notes == 0:
        print("   ❌ No notes!")
        exit()
    
    # Step 4: Update MIDI
    print("\nStep 4: Updating MIDI object...")
    midi.tracks = non_drum_tracks
    print(f"   ✅ MIDI updated")
    
    # Step 5: Tokenize
    print("\nStep 5: Tokenizing...")
    tokens = tokenizer(midi)
    print(f"   ✅ Tokenized: {len(tokens.ids)} tokens")
    
    # Step 6: Decode tokens (CORRECT WAY)
    print("\nStep 6: First 20 tokens:")
    for i, token_id in enumerate(tokens.ids[:20]):
        # Use tokenizer's decode method or vocab access
        try:
            # Method 1: Direct access
            token_str = tokenizer.vocab[token_id]
        except (KeyError, TypeError):
            # Method 2: Use __getitem__
            try:
                token_str = tokenizer[token_id]
            except:
                # Method 3: Just show ID
                token_str = f"ID:{token_id}"
        
        print(f"   {i}: {token_str}")
    
    print("\n✅ SUCCESS! Tokenization works!")
    
    # Bonus: Try decode
    print("\nBonus: Testing decode...")
    try:
        decoded = tokenizer.decode(tokens.ids[:50])
        print(f"   ✅ Decode works! Reconstructed {len(decoded.tracks)} tracks")
    except Exception as e:
        print(f"   ⚠️  Decode failed: {e}")
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}")
    print(f"   Message: {str(e)}")
    import traceback
    print("\n📋 Full traceback:")
    traceback.print_exc()