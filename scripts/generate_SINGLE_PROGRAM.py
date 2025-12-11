"""Genera MIDI usando pitch head filtrato per singolo program."""
import torch
from pathlib import Path
from midit.model.masked_transformer import MaskedTransformerREMI
from miditok import REMI, TokenizerConfig
import pretty_midi

def generate_single_program(model, target_program=0, seq_len=128, temperature=1.0):
    """Genera usando pitch + program head, filtra per target_program."""
    device = next(model.parameters()).device
    mask_id = 3
    tokens = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    
    pitch_sequence = []
    program_sequence = []
    
    print(f"   Generazione per Program_{target_program} (128 note)...")
    
    for iteration in range(20):
        with torch.no_grad():
            hidden_states = model(tokens)
            logits_dict = model.multi_head_output(hidden_states)
            
            # Sample ENTRAMBI pitch e program
            pitch_logits = logits_dict['pitch'] / temperature
            program_logits = logits_dict['program'] / temperature
            
            pitch_probs = torch.softmax(pitch_logits, dim=-1)
            program_probs = torch.softmax(program_logits, dim=-1)
            
            pitch_sampled = torch.multinomial(pitch_probs.view(-1, pitch_probs.size(-1)), 1).view(1, -1)
            program_sampled = torch.multinomial(program_probs.view(-1, program_probs.size(-1)), 1).view(1, -1)
            
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
    
    # Decode pitch + program separati
    pitch_sequence = pitch_sampled[0].cpu().tolist()
    program_sequence = program_sampled[0].cpu().tolist()
    
    # FILTRA solo target_program
    filtered_pitches = []
    for pitch_idx, prog_idx in zip(pitch_sequence, program_sequence):
        if prog_idx == target_program:
            filtered_pitches.append(pitch_idx)
    
    print(f"   Filtrate {len(filtered_pitches)} note per Program_{target_program}")
    
    return filtered_pitches


def pitch_indices_to_midi(pitch_indices, tokenizer, program=0):
    """Converti a MIDI."""
    pitch_tokens = [t for t in tokenizer.vocab.keys() if t.startswith('Pitch_')]
    pitch_tokens.sort()
    
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=program)
    
    current_time = 0.0
    time_step = 0.25
    
    for idx in pitch_indices:
        if idx < 0 or idx >= len(pitch_tokens):
            continue
        
        token_name = pitch_tokens[idx]
        midi_pitch = int(token_name.split('_')[1])
        
        note = pretty_midi.Note(
            velocity=80,
            pitch=midi_pitch,
            start=current_time,
            end=current_time + 0.5
        )
        instrument.notes.append(note)
        current_time += time_step
    
    midi.instruments.append(instrument)
    return midi


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--program', type=int, default=0, help='Program ID (0=Piano, 87=Synth Lead, -1=Drums)')
    parser.add_argument('--num_samples', type=int, default=3)
    args = parser.parse_args()
    
    print(f"🎵 Single-Program Generation (Program_{args.program})")
    print("=" * 80)
    
    output_dir = Path(f'generated_PROGRAM_{args.program}')
    output_dir.mkdir(exist_ok=True)
    
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval()
    
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
    
    for i in range(args.num_samples):
        print(f"\n🎹 Sample {i+1}/{args.num_samples}")
        
        try:
            pitch_indices = generate_single_program(
                model,
                target_program=args.program,
                temperature=1.2
            )
            
            if len(pitch_indices) < 10:
                print(f"   ⚠️ Troppo poche note filtrate ({len(pitch_indices)}), skip")
                continue
            
            midi = pitch_indices_to_midi(pitch_indices, tokenizer, args.program)
            
            output_path = output_dir / f"sample_{i:03d}.mid"
            midi.write(str(output_path))
            
            print(f"   💾 {output_path} ({output_path.stat().st_size} bytes)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎉 Done! Try program 0 (Piano), 87 (Synth), or -1 (Drums)")


if __name__ == '__main__':
    main()
