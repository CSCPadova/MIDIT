#!/usr/bin/env python3
"""Convert saved tokens to MIDI files."""

import numpy as np
import pretty_midi

def denormalize_tokens_by_type(normalized):
    """Riporta token normalizzati ai range originali."""
    tokens = []
    for i, val in enumerate(normalized):
        token_type = i % 4
        
        if token_type == 0:  # Pitch
            tokens.append(int(round(val * 127.0)))
        elif token_type == 1:  # Velocity
            tokens.append(int(round(val * 127.0 + 128)))
        elif token_type == 2:  # Duration
            tokens.append(int(round(val * 255.0 + 256)))
        elif token_type == 3:  # Delta
            tokens.append(int(round(val * 255.0 + 512)))
    
    return tokens

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
        
        # Convert bins to seconds
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
        if current_time > 30.0:
            break
    
    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"  💾 {output_path}")
    print(f"  📊 Notes: {len(instrument.notes)}, Duration: {current_time:.2f}s")

# Generate MIDIs
prompts = [
    ("tokens_0.npy", "piano_c_major.mid", "A simple piano melody in C major"),
    ("tokens_1.npy", "relaxing_piano.mid", "Relaxing piano music"),
    ("tokens_2.npy", "drums.mid", "Fast energetic drums"),
]

print("🎵 Converting tokens to MIDI...\n")

for token_file, midi_file, prompt in prompts:
    print(f"Prompt: {prompt}")
    tokens = np.load(f"tmp/{token_file}")
    tokens_to_midi(tokens, output_path=f"outputs/{midi_file}")
    print()

print("✅ Done! Listen to outputs/*.mid")
