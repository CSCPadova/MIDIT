"""Quick hack: genera MIDI usando solo pitch head - v2 API fix."""
import torch
from pathlib import Path
from midit.model.masked_transformer import MaskedTransformerREMI
from miditok import REMI, TokenizerConfig
import symusic

def generate_pitch_sequence(model, seq_len=256, temperature=1.0):
    """Genera sequenza di pitch usando iterative unmasking."""
    device = next(model.parameters()).device
    mask_id = 3
    tokens = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    
    print(f"   Generazione {seq_len} pitch tokens...")
    
    for iteration in range(20):
        with torch.no_grad():
            hidden_states = model(tokens)
            logits_dict = model.multi_head_output(hidden_states)
            pitch_logits = logits_dict['pitch']
            pitch_logits = pitch_logits / temperature
            probs = torch.softmax(pitch_logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, -1)
            confidence = probs.max(dim=-1).values
            
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
            
            tokens[0, positions_to_unmask] = sampled[0, positions_to_unmask]
        
        remaining = (tokens[0] == mask_id).sum().item()
        if iteration % 5 == 0:
            print(f"      Iter {iteration+1}: {remaining} remaining")
    
    tokens[tokens == mask_id] = 0
    return tokens[0].cpu()


def pitch_indices_to_midi_notes(pitch_indices, tokenizer):
    """Converti indici pitch in note MIDI."""
    # Get pitch token names
    pitch_tokens = [t for t in tokenizer.vocab.keys() if t.startswith('Pitch_')]
    pitch_tokens.sort()
    
    # ✅ FIX: Usa API corretta
    score = symusic.Score(ticks_per_quarter=480)
    
    track = symusic.Track(
        name="Generated",
        program=0,
        is_drum=False
    )
    
    current_time = 0
    time_step = 120  # 16th note
    
    print(f"\n   Conversione {len(pitch_indices)} pitch in note...")
    
    notes = []
    for idx in pitch_indices:
        if idx < 0 or idx >= len(pitch_tokens):
            continue
        
        token_name = pitch_tokens[idx]
        midi_pitch = int(token_name.split('_')[1])
        
        note = symusic.Note(
            time=current_time,
            duration=240,
            pitch=midi_pitch,
            velocity=80
        )
        notes.append(note)
        current_time += time_step
    
    track.notes = notes
    score.tracks = [track]
    score.tempos = [symusic.Tempo(time=0, qpm=120)]
    score.time_signatures = [symusic.TimeSignature(time=0, numerator=4, denominator=4)]
    
    print(f"   ✅ {len(notes)} note create")
    
    return score


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--output_dir', default='generated_PITCH_ONLY')
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1.2)
    args = parser.parse_args()
    
    print("🎵 QUICK HACK: Pitch-Only Generation v2")
    print("=" * 80)
    print("⚠️ Genera solo pitch, velocity/duration fissi!")
    print(f"Temperature: {args.temperature}\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    print(f"✅ Tokenizer ready\n")
    
    for i in range(args.num_samples):
        print(f"🎹 Sample {i+1}/{args.num_samples}")
        
        try:
            pitch_indices = generate_pitch_sequence(
                model,
                seq_len=args.seq_len,
                temperature=args.temperature
            )
            
            unique = torch.unique(pitch_indices)
            print(f"   Unique pitch: {len(unique)}")
            
            midi = pitch_indices_to_midi_notes(pitch_indices.tolist(), tokenizer)
            
            output_path = output_dir / f"sample_{i:03d}.mid"
            midi.dump_midi(str(output_path))
            
            file_size = output_path.stat().st_size
            print(f"   💾 {output_path} ({file_size} bytes)")
            
            if file_size > 500:
                print(f"   ✅ SUCCESS!\n")
            else:
                print(f"   ⚠️ Small\n")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("🎉 Complete! Scarica i MIDI e ascolta!")


if __name__ == '__main__':
    main()
