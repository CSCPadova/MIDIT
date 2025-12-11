"""Test finale: genera usando TUTTI gli heads disponibili."""
import torch
from pathlib import Path
from midit.model.masked_transformer import MaskedTransformerREMI
from miditok import REMI, TokenizerConfig
import pretty_midi

def generate_with_all_heads(model, seq_len=128, temperature=1.2):
    """Genera usando pitch + velocity + duration heads."""
    device = next(model.parameters()).device
    mask_id = 3
    tokens = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    
    print(f"   Generazione {seq_len} note con multi-head...")
    
    pitch_sequence = []
    velocity_sequence = []
    duration_sequence = []
    
    for iteration in range(20):
        with torch.no_grad():
            hidden_states = model(tokens)
            logits_dict = model.multi_head_output(hidden_states)
            
            # Sample dai 3 heads principali
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
            
            # Unmask
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
        
        remaining = (tokens[0] == mask_id).sum().item()
        if iteration % 5 == 0:
            print(f"      Iter {iteration+1}: {remaining} remaining")
    
    # Store sequences
    pitch_sequence = pitch_sampled[0].cpu().tolist()
    velocity_sequence = velocity_sampled[0].cpu().tolist()
    duration_sequence = duration_sampled[0].cpu().tolist()
    
    return pitch_sequence, velocity_sequence, duration_sequence


def sequences_to_midi(pitch_seq, velocity_seq, duration_seq, tokenizer):
    """Converti 3 sequenze in MIDI."""
    pitch_tokens = sorted([t for t in tokenizer.vocab.keys() if t.startswith('Pitch_')])
    velocity_tokens = sorted([t for t in tokenizer.vocab.keys() if t.startswith('Velocity_')])
    duration_tokens = sorted([t for t in tokenizer.vocab.keys() if t.startswith('Duration_')])
    
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0)  # Piano
    
    current_time = 0.0
    
    print(f"\n   Conversione {len(pitch_seq)} note...")
    
    valid_notes = 0
    for pitch_idx, vel_idx, dur_idx in zip(pitch_seq, velocity_seq, duration_seq):
        # Validate indices
        if pitch_idx < 0 or pitch_idx >= len(pitch_tokens):
            continue
        if vel_idx < 0 or vel_idx >= len(velocity_tokens):
            vel_idx = 15  # Default medium velocity
        if dur_idx < 0 or dur_idx >= len(duration_tokens):
            dur_idx = 7  # Default quarter note
        
        # Get values
        pitch_name = pitch_tokens[pitch_idx]
        vel_name = velocity_tokens[vel_idx]
        dur_name = duration_tokens[dur_idx]
        
        midi_pitch = int(pitch_name.split('_')[1])
        midi_velocity = int(vel_name.split('_')[1])
        
        # Parse duration (format: Duration_X.Y.Z)
        try:
            dur_parts = dur_name.split('_')[1].split('.')
            beats = int(dur_parts[0])
            frames = int(dur_parts[1]) if len(dur_parts) > 1 else 0
            duration_seconds = beats * 0.5 + frames * 0.125  # Rough approximation
        except:
            duration_seconds = 0.5  # Default
        
        note = pretty_midi.Note(
            velocity=midi_velocity,
            pitch=midi_pitch,
            start=current_time,
            end=current_time + duration_seconds
        )
        instrument.notes.append(note)
        valid_notes += 1
        
        # Time step: use duration for spacing
        current_time += duration_seconds * 0.5  # Overlap allowed
    
    midi.instruments.append(instrument)
    
    print(f"   ✅ {valid_notes} note valide create")
    
    return midi


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1.2)
    args = parser.parse_args()
    
    print("🎵 MULTI-HEAD GENERATION TEST (Pitch + Velocity + Duration)")
    print("=" * 80)
    print("⚠️ Ultimo test prima di decisione finale!\n")
    
    output_dir = Path('generated_MULTIHEAD_FINAL')
    output_dir.mkdir(exist_ok=True)
    
    print("📦 Loading model...")
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval()
    print("✅ Model loaded\n")
    
    print("🔤 Initializing tokenizer...")
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
    print("✅ Tokenizer ready\n")
    
    for i in range(args.num_samples):
        print(f"🎹 Sample {i+1}/{args.num_samples}")
        
        try:
            pitch_seq, vel_seq, dur_seq = generate_with_all_heads(
                model,
                seq_len=args.seq_len,
                temperature=args.temperature
            )
            
            # Stats
            unique_pitch = len(set(pitch_seq))
            unique_vel = len(set(vel_seq))
            unique_dur = len(set(dur_seq))
            
            print(f"\n   📊 Diversità:")
            print(f"      Pitch: {unique_pitch} unique")
            print(f"      Velocity: {unique_vel} unique")
            print(f"      Duration: {unique_dur} unique")
            
            midi = sequences_to_midi(pitch_seq, vel_seq, dur_seq, tokenizer)
            
            output_path = output_dir / f"sample_{i:03d}.mid"
            midi.write(str(output_path))
            
            file_size = output_path.stat().st_size
            print(f"\n   💾 {output_path} ({file_size} bytes)")
            
            if file_size > 1000:
                print(f"   ✅ SUCCESS!\n")
            else:
                print(f"   ⚠️ Small file\n")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 80)
    print("🎉 Test completato!")
    print("📁 Scarica i MIDI e confronta con pitch-only!")
    print("\n💡 Se migliore → heads funzionano (problema era mixing)")
    print("   Se uguale/peggio → conferma single-head necessario")


if __name__ == '__main__':
    main()
