from midit.Midicaps_dataloader1 import MidiCapsDataset

ds = MidiCapsDataset('.', split='train', fast_run=True)

# Prendi primo sample
sample = ds[0]
tokens = sample['midi_tokens']
non_zero = tokens[tokens != 0].tolist()

print("="*60)
print(f"PRIMI 50 TOKEN DEL SAMPLE 0:")
print("="*60)
for i, token in enumerate(non_zero[:50]):
    # Identifica range
    if 100 <= token < 500:
        tipo = "TEMPO"
    elif 200 <= token < 328:
        tipo = "KEY"
    elif 300 <= token < 428:
        tipo = "CHORD"
    elif 500 <= token < 564:
        tipo = "TIME_SIG"
    elif 600 <= token < 610:
        tipo = "DURATION"
    elif 700 <= token < 828:
        tipo = "INSTRUMENT"
    elif 800 <= token < 864:
        tipo = "GENRE"
    elif 900 <= token < 964:
        tipo = "MOOD"
    else:
        tipo = "UNKNOWN"
    
    print(f"{i:2d}. Token {token:4d} → {tipo}")

# Accedi ai dati raw
raw_sample = ds.dataset[0]
print("\n" + "="*60)
print("DATI RAW SAMPLE 0:")
print("="*60)
print(f"tempo: {raw_sample['tempo']}")
print(f"key: {raw_sample['key']}")
print(f"time_sig: {raw_sample['time_signature']}")
print(f"duration: {raw_sample['duration']}")
print(f"chords: {raw_sample['chord_summary'][:3]}")
print(f"instruments: {raw_sample['instrument_summary'][:3]}")
print(f"genre: {raw_sample['genre']}")
print(f"mood: {raw_sample['mood'][:3]}")
