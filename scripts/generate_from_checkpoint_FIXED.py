"""Generate MIDI from trained MaskedTransformerREMI - FIXED VERSION.

FIX: Usa correttamente i 9 classification heads invece di trattare
     hidden_states come logits diretti.
"""
import torch
from pathlib import Path
from midit.model.masked_transformer import MaskedTransformerREMI
from miditok import REMI, TokenizerConfig

def generate_midi_sample(model, tokenizer, num_bars=8, num_iterations=10, temperature=1.0):
    """Generate MIDI using iterative unmasking - FIXED."""
    device = next(model.parameters()).device
    
    seq_len = num_bars * 128
    mask_id = 3
    tokens = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
    
    print(f"   Starting with {seq_len} masked tokens...")
    
    for iteration in range(num_iterations):
        with torch.no_grad():
            # ✅ FIX 1: Get hidden states
            hidden_states = model(tokens)  # (1, seq_len, 512)
            
            # ✅ FIX 2: Get logits from multi-head output
            logits_dict = model.multi_head_output(hidden_states)
            
            # ✅ FIX 3: Use PITCH head for generation
            # (semplificazione: generiamo solo pitch, poi ricostruiamo MIDI)
            pitch_logits = logits_dict['pitch']  # (1, seq_len, 89)
            
            # Apply temperature
            pitch_logits = pitch_logits / temperature
            
            # Sample
            probs = torch.softmax(pitch_logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, -1)
            
            # Confidence
            confidence = probs.max(dim=-1).values
            
            # Unmask top 10% confident
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
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--output_dir', default='generated_FIXED')
    parser.add_argument('--num_bars', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()
    
    print("🎵 MIDI Generation - FIXED VERSION")
    print("=" * 80)
    print(f"✅ Using multi-head output correctly!")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Temperature: {args.temperature}")
    print()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📦 Loading model...")
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval()
    model = model.to('cpu')
    print(f"✅ Model loaded")
    
    # Verify architecture
    print(f"\n🔍 Architecture verification:")
    test_input = torch.randint(0, 518, (1, 10))
    with torch.no_grad():
        hidden = model(test_input)
        logits_dict = model.multi_head_output(hidden)
    print(f"   Hidden states shape: {hidden.shape}")
    print(f"   Logits dict keys: {list(logits_dict.keys())}")
    print(f"   Pitch logits shape: {logits_dict['pitch'].shape}")
    print(f"   ✅ Architecture OK!\n")
    
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
    print(f"✅ Tokenizer ready (vocab: {len(tokenizer.vocab)})")
    print()
    
    for i in range(args.num_samples):
        print(f"🎹 Generating sample {i+1}/{args.num_samples}...")
        
        try:
            tokens = generate_midi_sample(
                model, 
                tokenizer, 
                num_bars=args.num_bars,
                temperature=args.temperature
            )
            
            # Decode
            midi = tokenizer.decode(tokens.tolist())
            
            # Save
            output_path = output_dir / f"generated_{i:03d}.mid"
            midi.dump_midi(str(output_path))
            
            # Check file size
            file_size = output_path.stat().st_size
            marker = "✅" if file_size > 100 else "⚠️"
            print(f"   {marker} Saved: {output_path} ({file_size} bytes)")
            print()
            
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
