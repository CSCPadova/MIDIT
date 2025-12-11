"""Analizza struttura del dataset estratto - FIXED per JSONL"""
from pathlib import Path
import json

base_path = Path("data/midi_files")

print("🔍 Analisi Dataset MidiCaps\n")
print("="*80)

# 1. Controlla train.json (formato JSONL)
train_json = base_path / "train.json"
if train_json.exists():
    print("📄 train.json (JSONL format):")
    
    # Leggi JSONL (una riga = un JSON)
    data = []
    with open(train_json, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"   Total samples: {len(data)}")
    
    # Mostra primo sample
    if data:
        first = data[0]
        print(f"\n   📋 Primo sample di esempio:")
        for key, value in first.items():
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"      {key}: {value}")
        
        # Mostra altri 3 esempi per capire la struttura
        print(f"\n   📋 Altri 3 esempi di 'location':")
        for i in range(1, min(5, len(data))):
            print(f"      {i}. location: {data[i].get('location', 'N/A')}")
            print(f"         caption: {data[i].get('caption', 'N/A')[:60]}...")
else:
    print("⚠️  train.json non trovato!")

# 2. Cerca MIDI files
print("\n\n🎵 Ricerca MIDI files...")
midi_files = list(base_path.rglob("*.mid"))

print(f"   Trovati: {len(midi_files)} MIDI files")

if midi_files:
    print("\n   📁 Primi 10 MIDI files:")
    for i, midi_path in enumerate(midi_files[:10]):
        rel_path = midi_path.relative_to(base_path)
        size_kb = midi_path.stat().st_size / 1024
        print(f"      {i+1}. {rel_path} ({size_kb:.1f} KB)")
    
    # Analizza struttura directory
    print("\n   📊 Struttura directory MIDI:")
    # Conta files per subdirectory di primo livello
    lmd_full = base_path / "lmd_full"
    if lmd_full.exists():
        subdirs = [d for d in lmd_full.iterdir() if d.is_dir()]
        print(f"      lmd_full/ ha {len(subdirs)} subdirectories")
        print(f"      Esempi: {[d.name for d in subdirs[:5]]}")
        
        # Conta totale MIDI in lmd_full
        midi_count = len(list(lmd_full.rglob("*.mid")))
        print(f"      Totale MIDI in lmd_full/: {midi_count}")

else:
    print("   ⚠️  Nessun MIDI file trovato!")

print("\n" + "="*80)

# 3. Test Match: caption → MIDI
if data and midi_files:
    print("\n🔗 Test Match: Caption → MIDI File")
    print("="*80)
    
    # Prova a matchare i primi 5 samples
    matches = 0
    for i, sample in enumerate(data[:5]):
        location = sample.get('location', '')
        caption = sample.get('caption', '')[:60]
        
        print(f"\n   Sample {i+1}:")
        print(f"      Caption: {caption}...")
        print(f"      Location: {location}")
        
        # Costruisci possibili path
        possible_paths = [
            base_path / location,
            base_path / "lmd_full" / location,
        ]
        
        found = False
        for path in possible_paths:
            if path.exists():
                print(f"      ✅ MIDI trovato: {path.relative_to(base_path)}")
                matches += 1
                found = True
                break
        
        if not found:
            print(f"      ❌ MIDI non trovato")
    
    print(f"\n   📊 Match rate: {matches}/5")
    
    if matches == 0:
        print("\n   🔍 DEBUG: Esempio di path nel dataset vs filesystem")
        print(f"      Location nel JSON: {data[0].get('location', 'N/A')}")
        print(f"      Path reale MIDI: {midi_files[0].relative_to(base_path)}")

print("\n✅ Analisi completata!")

