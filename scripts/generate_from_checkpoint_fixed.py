"""Generate MIDI from trained MaskedTransformerREMI checkpoint (multi-head compatible)."""
import torch
from pathlib import Path
from midit.model.masked_transformer import MaskedTransformerREMI
from midit.model.multi_head_output import MultiHeadOutput
from miditok import REMI, TokenizerConfig

def generate_midi_sample(model, tokenizer, num_bars=8, num_iterations=10, temperature=1.0):
    """Generate MIDI using iterative unmasking, multi-head aware."""
    device = next(model.parameters()).device
    seq_len = num_bars * 128  # typical seq_len
    mask_id = 3  # REMI MASK token
    
    # Initialize all tokens as masked
    tokens = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    
    print(f"   Starting with {seq_len} masked tokens...")
    
    for iteration in range(num_iterations):
        with torch.no_grad():
            # Forward pass
            hidden_states = model(tokens)  # (1, seq_len, d_model)
            
            # Multi-head logits
            predictions = model.multi_head_output.forward(hidden_states)
            
            # --- Pitch sampling (main head) ---
            pitch_logits = predictions['pitch']  # (1, seq_len, vocab_pitch)
            pitch_probs = torch.softmax(pitch_logits / temperature, dim=-1)
            
            # Sample for masked positions
            mask_positions = (tokens[0] == mask_id).nonzero(as_tuple=False).squeeze(-1)
            if len(mask_positions) == 0:
                break
            
            sampled_pitch = torch.multinomial(
                pitch_probs.view(-1, pitch_probs.size(-1)), 1
            ).view(1, -1)
            
            # Choose which masked positions to unmask (10% per iteration)
            num_unmask = max(1, int(seq_len * 0.1))
            if len(mask_positions) > num_unmask:
                # Pick top confident positions
                confidence = pitch_probs[0, mask_positions].max(dim=-1).values
                top_confident = torch.topk(confidence, num_unmask).indices
                positions_to_unmask = mask_positions[top_confident]
            else:
                positions_to_unmask = mask_positions
            
            # Update pitch tokens
            tokens[0, positions_to_unmask] = sampled_pitch[0, positions_to_unmask]
            
            remaining = (tokens[0] == mask_id).sum().item()
            print(f"   Iteration {iteration+1}/{num_iterations}: {remaining} tokens remaining")
    
    # Replace any remaining MASK with PAD
    tokens[tokens == mask_id] = 0
    
    # Optional: set other heads (velocity/duration/bar/tempo/rest) to median values
    # Here just duplicating pitch tokens for simplicity (REMI will complain if missing)
    # Can be improved sampling all heads later
    return tokens[0].cpu().numpy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', default='generated_samples')
    parser.add_argument('--num_bars', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    
    print("🎵 MIDI Generation from MaskedTransformerREMI (multi-head fixed)")
    print("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📦 Loading model...")
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval().to('cpu')
    print(f"✅ Model loaded")
    
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
    tokenizer = REMI(config)
    print(f"✅ Tokenizer ready (vocab size: {len(tokenizer.vocab)})\n")
    
    for i in range(args.num_samples):
        print(f"🎹 Generating sample {i+1}/{args.num_samples}...")
        try:
            tokens = generate_midi_sample(
                model,
                tokenizer,
                num_bars=args.num_bars,
                temperature=args.temperature
            )
            midi = tokenizer.decode(tokens.tolist())
            output_path = output_dir / f"generated_{i:03d}.mid"
            midi.dump_midi(str(output_path))
            print(f"   ✅ Saved: {output_path}\n")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("🎉 Generation complete!")
    print(f"📁 Files saved in: {output_dir}")


if __name__ == '__main__':
    main()
