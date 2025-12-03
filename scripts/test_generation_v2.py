#!/usr/bin/env python3
"""Test MIDI generation from trained model."""

import torch
from midit.model.diffusion_transformer_v2 import MaskedDiffusionTransformer
from transformers import T5Tokenizer
import pretty_midi

# Load checkpoint
checkpoint_path = "logs/checkpoints/midit-epoch=01-val_loss=0.03.ckpt"
print(f"📂 Loading checkpoint: {checkpoint_path}")

model = MaskedDiffusionTransformer.load_from_checkpoint(checkpoint_path)
model.eval()
model.to('cpu')

# Tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Test prompts
prompts = [
    "A simple piano melody in C major",
    "Relaxing piano music",
    "Fast energetic drums"
]

print("\n🎵 Generating MIDI...\n")

for i, prompt in enumerate(prompts):
    print(f"Prompt {i+1}: {prompt}")
    
    # Tokenize text
    text_encoding = tokenizer(
        prompt,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            text_tokens=text_encoding['input_ids'],
            text_attention_mask=text_encoding['attention_mask'],
            max_length=512,
            num_inference_steps=50
        )
    
    # Analyze tokens
    tokens = generated[0].cpu().numpy()
    print(f"  Token range: [{tokens.min():.3f}, {tokens.max():.3f}]")
    print(f"  Token mean: {tokens.mean():.3f}")
    
    # Check validity (rough estimate)
    valid_count = ((tokens >= 0) & (tokens <= 1)).sum()
    print(f"  Valid tokens: {valid_count}/{len(tokens)} ({100*valid_count/len(tokens):.1f}%)")
    
    # Sample first 20 tokens
    print(f"  First 20 tokens: {tokens[:20]}")
    print()

print("✅ Test complete!")
