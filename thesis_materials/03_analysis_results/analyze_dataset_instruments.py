"""Esplora REMI+ tokenization per capire come funziona."""

from miditok import REMI, TokenizerConfig
from pathlib import Path
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

print("✅ Tokenizer creato!")
print(f"   Vocab size: {len(tokenizer.vocab)}")
print()

print("📂 Carico MIDI Piano di esempio...\n")

midi_path = "data/midi_files/lmd_full/1/1a20c7ecf60469366d1121ea48fd3f6e.mid"

try:
    tokens = tokenizer(midi_path)
    
    print(f"✅ MIDI tokenizzato!")
    print(f"   Numero token: {len(tokens[0])}")
    print(f"\n📝 Primi 50 token:")
    
    for i, token_id in enumerate(tokens[0][:50]):
        token_name = list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(token_id)]
        print(f"   [{i:2d}] {token_id:3d} → {token_name}")
        
    print("\n" + "="*60)
    print("🎯 CONFRONTO: 4-token vs REMI+")
    print("="*60)
    
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    notes.sort(key=lambda x: x.start)
    
    print(f"\n4-token encoding:")
    print(f"   Note MIDI: {len(notes)}")
    print(f"   Token totali: {len(notes) * 4}")
    
    print(f"\nREMI+ encoding:")
    print(f"   Token totali: {len(tokens[0])}")
    print(f"   Token per nota: ~{len(tokens[0])/len(notes):.1f}")
    
    print("\n✅ Script completato!")
    
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()