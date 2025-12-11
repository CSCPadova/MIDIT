from midit.midicaps_dataloader_v2 import MidiCapsDataset
from midit.model.diffusion_transformer_v2 import MaskedDiffusionTransformer
import torch

print('🧪 Test loading...')
dataset = MidiCapsDataset('data/midi_files', split='train', max_samples=10)
sample = dataset[0]

print(f'✅ MIDI shape: {sample["midi_tokens"].shape}')
print(f'✅ MIDI dtype: {sample["midi_tokens"].dtype}')
print(f'✅ MIDI range: [{sample["midi_tokens"].min():.2f}, {sample["midi_tokens"].max():.2f}]')

print('\n🧪 Test model...')
model = MaskedDiffusionTransformer(
    midi_vocab_size=768,
    text_vocab_size=32128,
    d_model=256,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
)
print('✅ Model initialized!')