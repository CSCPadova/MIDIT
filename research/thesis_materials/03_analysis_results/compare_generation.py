"""Generate and compare MIDI from baseline vs piano-only."""

import torch
import numpy as np
from pathlib import Path
from transformers import T5Tokenizer
import pretty_midi

from midit.model.diffusion_transformer_v2 import MaskedDiffusionTransformer
from midit.midicaps_dataloader_v2_filtered import denormalize_tokens_by_type


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    print(f"📂 Loading: {checkpoint_path}")
    model = MaskedDiffusionTransformer.load_from_checkpoint(
        checkpoint_path, map_location='cpu'
    )
    model.eval()
    return model


def generate_tokens(model, prompt: str, max_length: int = 512):
    """Generate MIDI tokens."""
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    text_encoding = tokenizer(
        prompt, max_length=256, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    
    with torch.no_grad():
        generated = model.generate(
            text_tokens=text_encoding['input_ids'],
            text_attention_mask=text_encoding['attention_mask'].float(),
            max_length=max_length,
            num_inference_steps=50,
        )
    
    return generated.squeeze(0).cpu().numpy()


def tokens_to_midi(tokens, output_path: str):
    """Convert tokens to MIDI."""
    discrete_tokens = denormalize_tokens_by_type(tokens)
    
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    
    current_time = 0.0
    pitches = []
    
    for i in range(0, len(discrete_tokens) - 3, 4):
        pitch = discrete_tokens[i]
        velocity = discrete_tokens[i+1] - 128
        duration_token = discrete_tokens[i+2] - 256
        delta_token = discrete_tokens[i+3] - 512
        
        if not (0 <= pitch <= 127):
            continue
        velocity = max(0, min(127, velocity))
        
        duration = (duration_token / 256.0) * 2.0
        delta = (delta_token / 256.0) * 2.0
        current_time += delta
        
        note = pretty_midi.Note(
            velocity=velocity, pitch=pitch,
            start=current_time, end=current_time + duration
        )
        instrument.notes.append(note)
        pitches.append(pitch)
    
    midi.instruments.append(instrument)
    midi.write(output_path)
    
    return {
        'num_notes': len(pitches),
        'pitch_range': (min(pitches), max(pitches)) if pitches else (0, 0),
        'pitch_mean': np.mean(pitches) if pitches else 0,
        'pitch_std': np.std(pitches) if pitches else 0,
        'unique_pitches': len(set(pitches)),
    }


def main():
    print("\n🎵 MIDI GENERATION COMPARISON\n")
    
    baseline_ckpt = "logs/comparison/baseline_all_instruments/checkpoints/best-epoch=02-val_loss=0.0250.ckpt"
    piano_ckpt = "logs/comparison/piano_only/checkpoints/best-epoch=02-val_loss=0.0195.ckpt"
    
    print("📂 Loading models...")
    baseline_model = load_model(baseline_ckpt)
    piano_model = load_model(piano_ckpt)
    
    prompts = [
        "A simple piano melody in C major",
        "Fast energetic piano",
        "Slow relaxing piano music",
    ]
    
    output_dir = Path("outputs/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"🎹 PROMPT {i+1}: {prompt}")
        print(f"{'='*60}")
        
        print(f"\n[1/2] Baseline...")
        baseline_tokens = generate_tokens(baseline_model, prompt)
        baseline_path = output_dir / f"baseline_{i+1}.mid"
        baseline_stats = tokens_to_midi(baseline_tokens, str(baseline_path))
        
        print(f"  Range: {baseline_stats['pitch_range'][0]}-{baseline_stats['pitch_range'][1]}")
        print(f"  Std: {baseline_stats['pitch_std']:.1f}")
        print(f"  Unique: {baseline_stats['unique_pitches']}")
        
        print(f"\n[2/2] Piano-only...")
        piano_tokens = generate_tokens(piano_model, prompt)
        piano_path = output_dir / f"piano_{i+1}.mid"
        piano_stats = tokens_to_midi(piano_tokens, str(piano_path))
        
        print(f"  Range: {piano_stats['pitch_range'][0]}-{piano_stats['pitch_range'][1]}")
        print(f"  Std: {piano_stats['pitch_std']:.1f}")
        print(f"  Unique: {piano_stats['unique_pitches']}")
        
        range_diff = (baseline_stats['pitch_range'][1] - baseline_stats['pitch_range'][0]) - \
                     (piano_stats['pitch_range'][1] - piano_stats['pitch_range'][0])
        std_diff = baseline_stats['pitch_std'] - piano_stats['pitch_std']
        
        print(f"\n  📊 Improvement:")
        print(f"     Range: {range_diff:+d} semitones {'✅' if range_diff > 0 else '❌'}")
        print(f"     Std: {std_diff:+.1f} {'✅' if std_diff > 0 else '❌'}")
    
    print(f"\n{'='*60}")
    print(f"✅ Done! Files in: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

