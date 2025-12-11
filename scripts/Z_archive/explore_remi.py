"""Esplora REMI+ tokenization."""

from miditok import REMI, TokenizerConfig
import pretty_midi

print("🎹 Setup REMI+ Tokenizer\n")

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

print(f"✅ Tokenizer creato!")
print(f"   Vocab size: {len(tokenizer.vocab)}\n")

# Inverti vocab per lookup ID -> Nome
id_to_token = {v: k for k, v in tokenizer.vocab.items()}

midi_path = "data/midi_files/lmd_full/1/1a20c7ecf60469366d1121ea48fd3f6e.mid"

print(f"📂 Carico MIDI: {midi_path}\n")

try:
    # Tokenize
    tokens = tokenizer(midi_path)
    token_ids = tokens[0]  # Primo track
    
    print(f"✅ MIDI tokenizzato!")
    print(f"   Token totali: {len(token_ids)}\n")
    
    print("📝 Primi 50 token:\n")
    for i, tid in enumerate(token_ids[:50]):
        token_name = id_to_token.get(tid, f"UNKNOWN_{tid}")
        print(f"   [{i:2d}] ID={tid:3d} → {token_name}")
    
    # Confronto con 4-token
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = [n for inst in midi.instruments if not inst.is_drum for n in inst.notes]
    
    print(f"\n{'='*60}")
    print("🎯 CONFRONTO")
    print(f"{'='*60}\n")
    print(f"4-token:  {len(notes)} note × 4 = {len(notes)*4} token")
    print(f"REMI+:    {len(token_ids)} token (~{len(token_ids)/len(notes):.1f} per nota)")
    print(f"\n✅ Completato!")
    
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()