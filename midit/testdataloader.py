"""Test new dataloader"""

print("Testing new dataloader...")

# Import
from midit.data.midicaps_dataloader1 import MidiCapsDataset

print("✓ Import successful!")

# Load
print("\nLoading dataset...")
dataset = MidiCapsDataset(
    data_dir='./data',
    split='train',
    fast_run=True
)

print(f"✓ Loaded {len(dataset)} examples")

# Test
example = dataset[0]
print(f"✓ MIDI: {example['midi_tokens'].shape}")
print(f"✓ Text: {example['text_tokens'].shape}")

print("\n✅ SUCCESS!")
