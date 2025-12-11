"""Generate MIDI from trained MaskedTransformerREMI checkpoint (multi-head fixed)."""
import torch
from pathlib import Path
from midit.model.masked_transformer import MaskedTransformerREMI
from miditok import REMI, TokenizerConfig

def generate_midi_sample(model, tokenizer, num_bars=8, num_iterations=50, temperature=1.0):
    """Generate MIDI using iterative unmasking."""
    device = next(model.parameters()).device
    
    seq_len = num_bars * 128
    mask_id = 3
    tokens = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    
    print(f"   Starting with {seq_len} masked tokens...")
    
    for iteration in range(num_iterations):
        with torch.no_grad():
            logits = model(tokens)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
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
        print(f"   Iteration {iteration+1}/{num_iterations}: {remaining} tokens remaining")
    
    tokens[tokens == mask_id] = 0
    return tokens[0].cpu().numpy()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--num_iterations', type=int, default=50)  # ← corretto
    parser.add_argument('--output_dir', default='generated_fixed_v3')
    parser.add_argument('--num_bars', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=1.5)
    args = parser.parse_args()
    
    print("🎵 MIDI Generation from MaskedTransformerREMI (multi-head fixed)")
    print("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📦 Loading model...")
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval()
    model = model.to('cpu')
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
                num_iterations=args.num_iterations,
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
            print()
            continue
    
    print("🎉 Generation complete!")
    print(f"📁 Files saved in: {output_dir}")


if __name__ == '__main__':
    main()
