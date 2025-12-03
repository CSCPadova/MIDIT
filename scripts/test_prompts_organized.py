#!/usr/bin/env python3
"""Test generation with organized prompts."""

import torch
import numpy as np
from midit.model.diffusion_transformer_v2 import MaskedDiffusionTransformer
from transformers import T5Tokenizer

# Checkpoint
checkpoint_path = "logs/checkpoints/midit-epoch=09-val_loss=0.02.ckpt"
print(f"📂 Loading checkpoint: {checkpoint_path}")

model = MaskedDiffusionTransformer.load_from_checkpoint(checkpoint_path)
model.eval()
model.to('cpu')

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# TEST SET 1: PROMPT SEMPLICI
simple_prompts = [
    "A simple piano melody in C major",
    "Relaxing piano music",
    "Fast energetic drums"
]

# TEST SET 2: PROMPT DETTAGLIATI
detailed_prompts = [
    "A very slow piano melody using only white keys C D E F G A B, no sharps or flats, in middle register C4 to C5",
    "Extremely relaxing slow piano music with long sustained notes, using only natural notes, no sharps or flats",
    "Fast repetitive drum pattern with short sharp notes, low pitch percussion below C4, rapid succession"
]

print("\n🎵 TEST 1: SIMPLE PROMPTS\n")
for i, prompt in enumerate(simple_prompts):
    print(f"  {i}: {prompt}")
    text_encoding = tokenizer(prompt, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        generated = model.generate(
            text_tokens=text_encoding['input_ids'],
            text_attention_mask=text_encoding['attention_mask'],
            max_length=512,
            num_inference_steps=50
        )
    
    tokens = generated[0].cpu().numpy()
    np.save(f"/tmp/simple_{i}.npy", tokens)

print("\n🎵 TEST 2: DETAILED PROMPTS\n")
for i, prompt in enumerate(detailed_prompts):
    print(f"  {i}: {prompt[:60]}...")
    text_encoding = tokenizer(prompt, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        generated = model.generate(
            text_tokens=text_encoding['input_ids'],
            text_attention_mask=text_encoding['attention_mask'],
            max_length=512,
            num_inference_steps=50
        )
    
    tokens = generated[0].cpu().numpy()
    np.save(f"/tmp/detailed_{i}.npy", tokens)

print("\n✅ Saved:")
print("  Simple: /tmp/simple_*.npy")
print("  Detailed: /tmp/detailed_*.npy")
print("\n📦 Compress with:")
print("  tar -czf /tmp/test_epoch9.tar.gz /tmp/simple_*.npy /tmp/detailed_*.npy")

