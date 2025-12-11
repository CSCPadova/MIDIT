"""Test generation con dataset piano filtrato."""
import torch
from pathlib import Path
from midit.model.masked_transformer import MaskedTransformerREMI
from miditok import REMI, TokenizerConfig
import pretty_midi
import numpy as np

def generate_multihead(model, seq_len=128, temperature=1.2):
    """Genera usando pitch + velocity + duration."""
    device = next(model.parameters()).device
    mask_id = 3
    tokens = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    
    for iteration in range(20):
        with torch.no_grad():
            hidden_states = model(tokens)
            logits_dict = model.multi_head_output(hidden_states)
            
            pitch_logits = logits_dict['pitch'] / temperature
            velocity_logits = logits_dict['velocity'] / temperature
            duration_logits = logits_dict['duration'] / temperature
            
            pitch_probs = torch.softmax(pitch_logits, dim=-1)
            velocity_probs = torch.softmax(velocity_logits, dim=-1)
            duration_probs = torch.softmax(duration_logits, dim=-1)
            
            pitch_sampled = torch.multinomial(pitch_probs.view(-1, pitch_probs.size(-1)), 1).view(1, -1)
            velocity_sampled = torch.multinomial(velocity_probs.view(-1, velocity_probs.size(-1)), 1).view(1, -1)
            duration_sampled = torch.multinomial(duration_probs.view(-1, duration_probs.size(-1)), 1).view(1, -1)
            
            confidence = pitch_probs.max(dim=-1).values
            
            num_unmask = max(1, int(seq_len * 0.1))
            mask_positions = (tokens[0] == mask_id).nonzero(as_tuple=False).squeeze(-1)
            
            if len(mask_positions) == 0:
                break
            
            mask_confidence = confidence[0, mask_positions]
            
            if len(mask_positions) > num_unmask:
                top_confident = torch.topk(mask_confidence, num_unmask).indices
                positions_to_unmask = mask_positions[top_confident]
            else:
                positions_to_unmask = mask_positions
            
            tokens[0, positions_to_unmask] = pitch_sampled[0, positions_to_unmask]
    
    return pitch_sampled[0].cpu().tolist(), velocity_sampled[0].cpu().tolist(), duration_sampled[0].cpu().tolist()


def sequences_to_midi(pitch_seq, velocity_seq, duration_seq, tokenizer):
    """Converti a MIDI."""
    pitch_tokens = sorted([t for t in tokenizer.vocab.keys() if t.startswith('Pitch_')])
    velocity_tokens = sorted([t for t in tokenizer.vocab.keys() if t.startswith('Velocity_')])
    duration_tokens = sorted([t for t in tokenizer.vocab.keys() if t.startswith('Duration_')])
    
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0)
    
    current_time = 0.0
    pitches = []
    
    for pitch_idx, vel_idx, dur_idx in zip(pitch_seq, velocity_seq, duration_seq):
        if pitch_idx < 0 or pitch_idx >= len(pitch_tokens):
            continue
        if vel_idx < 0 or vel_idx >= len(velocity_tokens):
            vel_idx = 15
        if dur_idx < 0 or dur_idx >= len(duration_tokens):
            dur_idx = 7
        
        pitch_name = pitch_tokens[pitch_idx]
        vel_name = velocity_tokens[vel_idx]
        dur_name = duration_tokens[dur_idx]
        
        midi_pitch = int(pitch_name.split('_')[1])
        midi_velocity = int(vel_name.split('_')[1])
        
        try:
            dur_parts = dur_name.split('_')[1].split('.')
            beats = int(dur_parts[0])
            frames = int(dur_parts[1]) if len(dur_parts) > 1 else 0
            duration_seconds = beats * 0.5 + frames * 0.125
        except:
            duration_seconds = 0.5
        
        note = pretty_midi.Note(
            velocity=midi_velocity,
            pitch=midi_pitch,
            start=current_time,
            end=current_time + duration_seconds
        )
        instrument.notes.append(note)
        pitches.append(midi_pitch)
        
        current_time += duration_seconds * 0.5
    
    midi.instruments.append(instrument)
    return midi, pitches


print("=" * 80)
print("🎹 TEST GENERATION - Dataset Piano Filtrato")
print("=" * 80)

# Load model
print("\n📦 Loading checkpoint...")
model = MaskedTransformerREMI.load_from_checkpoint(
    "logs/checkpoints_phase1/phase1-epoch=44-val_loss=0.0982.ckpt"
)
model.eval()
print("✅ Model loaded")

# Tokenizer
print("\n🔤 Initializing tokenizer...")
config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8},
    num_velocities=32,
    use_chords=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_programs=True,
    special_tokens=["PAD", "BOS", "EOS", "MASK"],
    use_controls=False,
)
tokenizer = REMI(tokenizer_config=config)
print("✅ Ready")

# Generate
output_dir = Path('generated_PIANO_TEST')
output_dir.mkdir(exist_ok=True)

print("\n🎵 Generating 3 samples...")

for i in range(3):
    print(f"\n   Sample {i+1}/3...")
    
    pitch_seq, vel_seq, dur_seq = generate_multihead(model, temperature=1.2)
    midi, pitches = sequences_to_midi(pitch_seq, vel_seq, dur_seq, tokenizer)
    
    # Analyze
    intervals = [abs(pitches[j+1] - pitches[j]) for j in range(len(pitches)-1)]
    small_intervals = sum(1 for x in intervals if x <= 2)
    large_intervals = sum(1 for x in intervals if x > 7)
    
    print(f"      Pitch range: {min(pitches)}-{max(pitches)}")
    print(f"      Avg interval: {np.mean(intervals):.1f}")
    print(f"      Gradi congiunti: {small_intervals/len(intervals)*100:.1f}%")
    print(f"      Salti ampi (>7): {large_intervals/len(intervals)*100:.1f}%")
    
    output_path = output_dir / f"sample_{i:03d}.mid"
    midi.write(str(output_path))
    print(f"      💾 {output_path}")

print("\n" + "=" * 80)
print("🎯 CONFRONTO CON MULTI-TRACK:")
print("=" * 80)
print("   Multi-track dataset:")
print("      Gradi congiunti: 19.8%")
print("      Salti ampi: 61.2%")
print()
print("   ASPETTATIVA con Piano dataset:")
print("      Gradi congiunti: >40%")
print("      Salti ampi: <30%")
print()
print("   → Se valori migliorano = PROBLEMA RISOLTO! ✅")
print("=" * 80)
