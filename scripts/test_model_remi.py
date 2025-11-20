"""Test modello con vocab REMI+."""

import torch
from midit.model.diffusion_transformer_v2_remi import MaskedDiffusionTransformerREMI

print("🧪 Testing model with REMI+ vocab\n")

# Crea modello con vocab REMI+
model = MaskedDiffusionTransformerREMI(
    midi_vocab_size=389,  # REMI+
    text_vocab_size=32100,
    d_model=256,
    nhead=4,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    max_seq_length=1024,
    timesteps=1000,
    lr=1e-4,
)

print(f"✅ Model created!")
print(f"   MIDI vocab: {model.hparams.midi_vocab_size}")
print(f"   Max seq length: {model.hparams.max_seq_length}")
print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
batch_size = 2
seq_len = 1024

midi_tokens = torch.rand(batch_size, seq_len)  # REMI+ normalizzato [0,1]
text_tokens = torch.randint(0, 32100, (batch_size, 256))
timestep = torch.randint(0, 1000, (batch_size,))  # Random diffusion timestep
text_mask = torch.ones(batch_size, 256)
midi_mask = torch.ones(batch_size, seq_len)

print(f"\n🔄 Testing forward pass...")
print(f"   MIDI tokens: {midi_tokens.shape}")
print(f"   Text tokens: {text_tokens.shape}")
print(f"   Timestep: {timestep.shape}")

try:
    with torch.no_grad():
        output = model(
            midi_tokens,      # (batch, seq_len)
            text_tokens,      # (batch, text_len)
            timestep,         # (batch,)
            text_mask,        # (batch, text_len)
            midi_mask,        # (batch, seq_len)
        )
    
    print(f"\n✅ Forward pass OK!")
    print(f"   Input:  {midi_tokens.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"   Output dtype: {output.dtype}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test completed!")