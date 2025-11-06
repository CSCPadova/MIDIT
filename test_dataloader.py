from midit.Midicaps_dataloader1 import MidiCapsDataset

print("🔄 Testing dataloader...")
ds = MidiCapsDataset('.', split='train', fast_run=True)
print(f"✅ Dataset: {len(ds)} samples")

sample = ds[0]
print(f"✅ MIDI tokens shape: {sample['midi_tokens'].shape}")
print(f"✅ Text tokens shape: {sample['text_tokens'].shape}")
print("✅ Dataloader funziona!")
