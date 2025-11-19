import torch
from midit.model import MaskedDiffusionTransformer
from transformers import T5Tokenizer

# Load checkpoint
ckpt_path = r"C:\Users\simoz\Desktop\checkpoints\midit-epoch=08-val_loss=1.91.ckpt"
checkpoint = torch.load(ckpt_path, map_location='cpu')

# Init model
model = MaskedDiffusionTransformer(
    midi_vocab_size=768,
    text_vocab_size=32128,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
)

model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Tokenizer
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')

# Test prompts
prompts = [
    "A relaxing piano melody in C major with a slow tempo",
    "An energetic drum beat with fast tempo",
    "A sad violin solo in minor key"
]

for i, prompt in enumerate(prompts):
    print(f"\n{'='*60}")
    print(f"TEST {i+1}: {prompt}")
    print('='*60)
    
    # Tokenize
    text_inputs = tokenizer(
        prompt, 
        return_tensors='pt', 
        padding='max_length', 
        max_length=128,
        truncation=True
    )
    
    text_tokens = text_inputs['input_ids']
    text_mask = text_inputs['attention_mask']
    
    print(f"Text tokens shape: {text_tokens.shape}")
    print(f"Text mask shape: {text_mask.shape}")
    
    # Generate
    with torch.no_grad():
        midi_length = 256
        generated = torch.randint(0, 768, (1, midi_length))
        
        # Simple denoising (10 steps)
        for t in range(100, 0, -10):
            timesteps = torch.tensor([t])
            try:
                logits = model(generated, text_tokens, text_mask, timesteps)
                generated = logits.argmax(dim=-1)
            except Exception as e:
                print(f"❌ Error at timestep {t}: {e}")
                break
    
    # Analyze tokens
    tokens = generated[0].numpy()
    
    print(f"\n📊 GENERATED TOKENS ANALYSIS:")
    print(f"Total tokens: {len(tokens)}")
    
    # Split by type (4-token encoding)
    pitch_idx = tokens[0::4]
    vel_idx = tokens[1::4]
    dur_idx = tokens[2::4]
    delta_idx = tokens[3::4]
    
    print(f"\nPitch (0-127): {len(pitch_idx[pitch_idx < 128])} valid")
    print(f"  Range: {pitch_idx.min()}-{pitch_idx.max()}")
    print(f"  Sample: {pitch_idx[:5].tolist()}")
    
    print(f"\nVelocity (128-255): {len(vel_idx[(vel_idx >= 128) & (vel_idx < 256)])} valid")
    print(f"  Range: {vel_idx.min()}-{vel_idx.max()}")
    print(f"  Sample: {vel_idx[:5].tolist()}")
    
    print(f"\nDuration (256-511): {len(dur_idx[(dur_idx >= 256) & (dur_idx < 512)])} valid")
    print(f"  Range: {dur_idx.min()}-{dur_idx.max()}")
    
    print(f"\nDelta (512-767): {len(delta_idx[(delta_idx >= 512) & (delta_idx < 768)])} valid")
    print(f"  Range: {delta_idx.min()}-{delta_idx.max()}")

print("\n" + "="*60)
print("✅ Generation test complete!")