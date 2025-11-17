#!/usr/bin/env python3
"""Generate actual MIDI files from trained model."""

import torch
import pretty_midi
from midit.model.diffusion_transformer_v2 import MaskedDiffusionTransformer
from midit.midicaps_dataloader_v2 import denormalize_tokens_by_type
from transformers import T5Tokenizer

def tokens_to_midi(tokens, output_path="output.mid", tempo=120):
    """Convert token sequence to MIDI file."""
    
    # Denormalize tokens
    denorm_tokens = denormalize_tokens_by_type(tokens)
    
    # Create MIDI
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  # Piano
    
    current_time = 0.0
    
    # Parse 4-token groups into notes
    for i in range(0, len(denorm_tokens) - 3, 4):
        pitch = denorm_tokens[i]
        velocity_token = denorm_tokens[i + 1]
        duration_token = denorm_tokens[i + 2]
        delta_token = denorm_tokens[i + 3]
        
        # Convert back to actual values
        velocity = velocity_token - 128
        duration_bin = duration_token - 256
        delta_bin = delta_token - 512
        
        # Convert bins to seconds (assume max 2.0s, 256 bins)
        duration_sec = (duration_bin / 256.0) * 2.0
        delta_sec = (delta_bin / 256.0) * 2.0
        
        # Skip invalid notes
        if not (0 <= pitch <= 127):
            continue
        if not (0 <= velocity <= 127):
            velocity = max(0, min(127, velocity))
        if duration_sec <= 0:
            duration_sec = 0.1
        
        # Update time
        current_time += delta_sec
        
        # Create note
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=current_time,
            end=current_time + duration_sec
        )
        instrument.notes.append(note)
        
        # Stop if too long
        if current_time > 30.0:  # Max 30 seconds
            break
    
    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"  💾 Saved: {output_path}")
    print(f"  📊 Notes: {len(instrument.notes)}")
    print(f"  ⏱️  Duration: {current_time:.2f}s")


# Load model
checkpoint_path = "logs/checkpoints/midit-epoch=01-val_loss=0.03.ckpt"
print(f"📂 Loading checkpoint: {checkpoint_path}\n")

model = MaskedDiffusionTransformer.load_from_checkpoint(checkpoint_path)
model.eval()
model.to('cpu')

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Test prompts
prompts = [
    ("A simple piano melody in C major", "piano_c_major.mid"),
    ("Relaxing piano music", "relaxing_piano.mid"),
    ("Fast energetic drums", "drums.mid"),
]

print("🎵 Generating MIDI files...\n")

for prompt, filename in prompts:
    print(f"Prompt: {prompt}")
    
    # Tokenize
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
    
    # Convert to MIDI
    tokens = generated[0].cpu().numpy()
    tokens_to_midi(tokens, output_path=f"outputs/{filename}")
    print()

print("✅ All MIDI files generated!")
print("\n📁 Files saved in: outputs/")
print("🎧 Download and listen to evaluate quality!")