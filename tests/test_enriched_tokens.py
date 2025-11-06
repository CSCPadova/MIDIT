"""Test dettagliato token arricchiti"""
from midit.Midicaps_dataloader1 import MidiCapsDataset
import torch

ds = MidiCapsDataset('.', split='train', fast_run=True)

print("="*60)
print("TEST TOKEN ARRICCHITI")
print("="*60)

# Analizza primi 10 samples
lengths = []
token_ranges = {
    '100-499 (tempo)': 0,
    '200-327 (key)': 0,
    '300-427 (chords)': 0,
    '500-563 (time_sig)': 0,
    '600-609 (duration)': 0,
    '700-827 (instruments)': 0,
    '800-863 (genre)': 0,
    '900-963 (mood)': 0,
}

for i in range(10):
    sample = ds[i]
    tokens = sample['midi_tokens']
    non_zero = tokens[tokens != 0]
    lengths.append(len(non_zero))
    
    # Conta token per range
    for token in non_zero:
        t = token.item()
        if 100 <= t < 500: token_ranges['100-499 (tempo)'] += 1
        elif 200 <= t < 328: token_ranges['200-327 (key)'] += 1
        elif 300 <= t < 428: token_ranges['300-427 (chords)'] += 1
        elif 500 <= t < 564: token_ranges['500-563 (time_sig)'] += 1
        elif 600 <= t < 610: token_ranges['600-609 (duration)'] += 1
        elif 700 <= t < 828: token_ranges['700-827 (instruments)'] += 1
        elif 800 <= t < 864: token_ranges['800-863 (genre)'] += 1
        elif 900 <= t < 964: token_ranges['900-963 (mood)'] += 1

import numpy as np
print(f"\n📊 STATISTICHE (10 samples):")
print(f"   Lunghezza media: {np.mean(lengths):.1f} tokens")
print(f"   Min: {min(lengths)}, Max: {max(lengths)}")

print(f"\n📈 DISTRIBUZIONE TOKEN:")
for range_name, count in token_ranges.items():
    print(f"   {range_name:30s}: {count:3d} token")

print("\n" + "="*60)

# Confronto con vecchio
print("\n📊 CONFRONTO:")
print(f"   Prima:  ~15 token per sample (solo tempo, key, chords)")
print(f"   Dopo:   ~{np.mean(lengths):.0f} token per sample (8 features!)")
print(f"   Incremento: +{(np.mean(lengths)/15 - 1)*100:.0f}%")
print("="*60)
