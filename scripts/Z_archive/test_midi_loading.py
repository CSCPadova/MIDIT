"""Test caricamento MIDI con pretty_midi"""
import json
import pretty_midi
from pathlib import Path

base_path = Path("data/midi_files")
train_json = base_path / "train.json"

print("🎵 Test Caricamento MIDI con pretty_midi\n")
print("="*80)

# Carica primi 10 samples
samples = []
with open(train_json, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        samples.append(json.loads(line))

success = 0
errors = []

for i, sample in enumerate(samples):
    midi_path = base_path / sample['location']
    
    print(f"\nSample {i+1}/10:")
    print(f"  Caption: {sample['caption'][:60]}...")
    print(f"  Tempo metadata: {sample['tempo']} BPM")
    print(f"  Key: {sample['key']}")
    print(f"  Duration: {sample['duration']}s")
    
    try:
        # Carica MIDI
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Verifica info
        actual_tempo = midi.estimate_tempo()
        duration = midi.get_end_time()
        n_instruments = len(midi.instruments)
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        
        print(f"  ✅ MIDI caricato:")
        print(f"     Tempo stimato: {actual_tempo:.1f} BPM")
        print(f"     Duration reale: {duration:.1f}s")
        print(f"     Instruments: {n_instruments}")
        print(f"     Total notes: {total_notes}")
        
        # Verifica coerenza
        tempo_diff = abs(actual_tempo - sample['tempo'])
        duration_diff = abs(duration - sample['duration'])
        
        if tempo_diff > 20:
            print(f"     ⚠️ Tempo mismatch: {tempo_diff:.1f} BPM diff")
        if duration_diff > 5:
            print(f"     ⚠️ Duration mismatch: {duration_diff:.1f}s diff")
        
        success += 1
        
    except Exception as e:
        print(f"  ❌ Errore: {e}")
        errors.append((sample['location'], str(e)))

print("\n" + "="*80)
print(f"📊 RISULTATI: {success}/10 MIDI caricati con successo")

if errors:
    print(f"\n⚠️ Errori riscontrati:")
    for path, error in errors:
        print(f"   - {path}: {error}")
else:
    print("\n✅ PERFETTO! Tutti i MIDI sono caricabili e validi!")
    print("🚀 Pronto per creare il DataLoader!")

