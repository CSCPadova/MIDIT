"""Preprocessing: MIDI files → tokens per MIDIT"""
import json
from pathlib import Path
from miditoolkit import MidiFile
from datasets import load_dataset
from tqdm import tqdm

def midi_to_tokens(midi_path, max_length=1024):
    """Converte file MIDI in sequenza di tokens."""
    try:
        midi = MidiFile(midi_path)
        tokens = []
        
        # Estrai note da tutti gli strumenti
        for instrument in midi.instruments:
            for note in instrument.notes:
                # Token format: [type, pitch, velocity, time]
                tokens.append({
                    'type': 'note_on',
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'time': note.start
                })
                tokens.append({
                    'type': 'note_off',
                    'pitch': note.pitch,
                    'time': note.end
                })
        
        # Ordina per tempo
        tokens.sort(key=lambda x: x['time'])
        
        # Converti in numeri (semplificato)
        numeric_tokens = []
        for token in tokens[:max_length]:
            # Encoding semplice: type*1000 + pitch*10 + velocity
            if token['type'] == 'note_on':
                numeric_tokens.append(1000 + token['pitch']*10 + token['velocity'])
            else:
                numeric_tokens.append(2000 + token['pitch']*10)
        
        return numeric_tokens
    
    except Exception as e:
        print(f"Errore su {midi_path}: {e}")
        return None

def preprocess_dataset(num_samples=100):
    """Preprocessa dataset MIDICaps."""
    print("📥 Caricamento dataset...")
    dataset = load_dataset("amaai-lab/MidiCaps", split="train")
    
    results = []
    print(f"🔄 Preprocessing {num_samples} samples...")
    
    for i in tqdm(range(num_samples)):
        sample = dataset[i]
        midi_path = sample['location']
        
        # Converti MIDI → tokens
        tokens = midi_to_tokens(midi_path)
        
        if tokens:
            results.append({
                'location': midi_path,
                'caption': sample['caption'],
                'note_token_sequence': tokens,
                'genre': sample['genre'],
                'mood': sample['mood']
            })
    
    # Salva
    output_path = 'midicaps_preprocessed.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Salvati {len(results)} samples in {output_path}")

if __name__ == "__main__":
    preprocess_dataset(num_samples=100)
