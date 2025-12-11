"""Verifica che dataloader carichi dati corretti"""
from midit.Midicaps_dataloader1 import MidiCapsDataset
import torch

ds = MidiCapsDataset('.', split='train', fast_run=True)

print("="*60)
print("VERIFICA DATALOADER")
print("="*60)

# Prendi 5 samples
for i in range(5):
    sample = ds[i]
    raw = ds.dataset[i]
    
    print(f"\n📊 SAMPLE {i}:")
    print(f"Caption: {raw['caption'][:80]}...")
    print(f"Tempo reale: {raw['tempo']} BPM")
    print(f"Genre reale: {raw['genre']}")
    
    # Verifica token
    midi_tokens = sample['midi_tokens']
    non_zero = midi_tokens[midi_tokens != 0]
    
    # Cerca tempo nei token
    tempo_tokens = [t.item() for t in non_zero if 100 <= t < 500]
    print(f"Token tempo generati: {tempo_tokens[:3]}")
    if tempo_tokens:
        decoded_tempo = tempo_tokens[0] - 100
        print(f"Decoded tempo: {decoded_tempo} BPM")
        print(f"Match? {abs(decoded_tempo - raw['tempo']) < 50}")
    
    # Cerca genre
    genre_tokens = [t.item() for t in non_zero if 800 <= t < 864]
    print(f"Genre tokens generati: {genre_tokens[:3]}")
    print(f"Expected genre: {raw['genre']}")
