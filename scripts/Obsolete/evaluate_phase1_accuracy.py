"""
Valuta accuracy per-head del checkpoint Phase 1.
Carica direttamente file .pt preprocessati.
"""
import torch
from pathlib import Path
from tqdm import tqdm
from midit.model.masked_transformer import MaskedTransformerREMI
from miditok import REMI, TokenizerConfig
from collections import defaultdict
import numpy as np
import random

def create_masked_input(tokens, mask_ratio=0.15):
    """
    Maschera tokens per test ricostruzione.
    Usa stessa strategia del training.
    """
    mask_id = 3  # MASK token
    seq_len = len(tokens)
    
    # Random positions to mask
    num_mask = max(1, int(seq_len * mask_ratio))
    mask_positions = random.sample(range(seq_len), num_mask)
    
    # Create masked input
    input_ids = tokens.clone()
    input_ids[mask_positions] = mask_id
    
    return input_ids, mask_positions


def split_by_token_type(tokens, tokenizer):
    """
    Separa tokens per tipo (pitch, velocity, duration, etc.)
    """
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    
    targets = {
        'pitch': [],
        'velocity': [],
        'duration': [],
        'program': [],
        'position': [],
        'bar': [],
        'tempo': [],
        'timesig': [],
        'rest': []
    }
    
    indices = {key: [] for key in targets.keys()}
    
    for i, token_id in enumerate(tokens.tolist()):
        if token_id not in id_to_token:
            continue
        
        token_name = id_to_token[token_id]
        
        if token_name.startswith('Pitch_') or token_name.startswith('PitchDrum_'):
            targets['pitch'].append(token_id)
            indices['pitch'].append(i)
        elif token_name.startswith('Velocity_'):
            targets['velocity'].append(token_id)
            indices['velocity'].append(i)
        elif token_name.startswith('Duration_'):
            targets['duration'].append(token_id)
            indices['duration'].append(i)
        elif token_name.startswith('Program_'):
            targets['program'].append(token_id)
            indices['program'].append(i)
        elif token_name.startswith('Position_'):
            targets['position'].append(token_id)
            indices['position'].append(i)
        elif token_name.startswith('Bar_'):
            targets['bar'].append(token_id)
            indices['bar'].append(i)
        elif token_name.startswith('Tempo_'):
            targets['tempo'].append(token_id)
            indices['tempo'].append(i)
        elif token_name.startswith('TimeSig_'):
            targets['timesig'].append(token_id)
            indices['timesig'].append(i)
        elif token_name.startswith('Rest_'):
            targets['rest'].append(token_id)
            indices['rest'].append(i)
    
    return targets, indices


def evaluate_reconstruction_accuracy(model, data_dir, tokenizer, num_samples=100):
    """
    Calcola accuracy di ricostruzione per ogni head.
    """
    print("=" * 80)
    print("🔬 PHASE 1 RECONSTRUCTION ACCURACY TEST")
    print("=" * 80)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get preprocessed files
    data_path = Path(data_dir)
    all_files = sorted(data_path.glob("*.pt"))
    
    if len(all_files) == 0:
        print(f"❌ No .pt files found in {data_dir}")
        return {}
    
    # Sample random files
    sample_files = random.sample(all_files, min(num_samples, len(all_files)))
    
    print(f"\n📊 Testing on {len(sample_files)} files...")
    
    # Statistics
    correct = defaultdict(int)
    total = defaultdict(int)
    predictions_by_head = defaultdict(list)
    targets_by_head = defaultdict(list)
    
    # Reverse vocab
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    
    with torch.no_grad():
        for file_path in tqdm(sample_files, desc="Evaluating"):
            try:
                # Load preprocessed file
                data = torch.load(file_path)
                tokens = data['tokens']
                
                # Skip very short sequences
                if len(tokens) < 50:
                    continue
                
                # Truncate if too long
                if len(tokens) > 512:
                    tokens = tokens[:512]
                
                # Create masked input
                input_ids, mask_positions = create_masked_input(tokens, mask_ratio=0.15)
                
                # To device
                input_ids = input_ids.unsqueeze(0).to(device)  # (1, seq_len)
                tokens = tokens.to(device)
                
                # Forward pass
                hidden_states = model(input_ids)
                logits_dict = model.multi_head_output(hidden_states)
                
                # Split ground truth by token type
                gt_by_type, gt_indices = split_by_token_type(tokens, tokenizer)
                
                # Evaluate each head
                for head_name, logits in logits_dict.items():
                    # Get predictions at masked positions
                    pred = torch.argmax(logits, dim=-1).squeeze(0)  # (seq_len,)
                    
                    # Filter only masked positions of this token type
                    valid_positions = [pos for pos in mask_positions if pos in gt_indices[head_name]]
                    
                    if len(valid_positions) == 0:
                        continue
                    
                    # Compare predictions with ground truth
                    for pos in valid_positions:
                        pred_token = pred[pos].item()
                        gt_token = tokens[pos].item()
                        
                        predictions_by_head[head_name].append(pred_token)
                        targets_by_head[head_name].append(gt_token)
                        
                        if pred_token == gt_token:
                            correct[head_name] += 1
                        total[head_name] += 1
                        
            except Exception as e:
                print(f"\n⚠️ Error on {file_path.name}: {e}")
                continue
    
    # Calculate and display results
    print("\n" + "=" * 80)
    print("📊 RESULTS - Reconstruction Accuracy per Head:")
    print("=" * 80)
    
    results = {}
    
    for head_name in sorted(correct.keys()):
        if total[head_name] == 0:
            continue
        
        acc = correct[head_name] / total[head_name] * 100
        results[head_name] = {
            'accuracy': acc,
            'correct': correct[head_name],
            'total': total[head_name]
        }
        
        # Status indicator
        if acc >= 70:
            status = "✅ GOOD"
        elif acc >= 50:
            status = "⚠️ FAIR"
        else:
            status = "❌ POOR"
        
        print(f"\n{head_name.upper():12} {status}")
        print(f"   Accuracy: {acc:>6.2f}% ({correct[head_name]:>6}/{total[head_name]:<6})")
        
        # Unique predictions
        unique_pred = len(set(predictions_by_head[head_name]))
        unique_target = len(set(targets_by_head[head_name]))
        print(f"   Unique predicted: {unique_pred:>3} (target has {unique_target:>3})")
        
        # Mode collapse detection
        if unique_pred == 1:
            most_common_id = predictions_by_head[head_name][0]
            if most_common_id in id_to_token:
                most_common = id_to_token[most_common_id]
                print(f"   ⚠️ MODE COLLAPSE! Always predicts: {most_common}")
        elif unique_pred < 5:
            print(f"   ⚠️ LOW DIVERSITY! Only {unique_pred} predictions")
    
    # Overall assessment
    print("\n" + "=" * 80)
    print("🎯 OVERALL ASSESSMENT:")
    print("=" * 80)
    
    if len(results) == 0:
        print("\n❌ No results! Check if files are valid.")
        return {}
    
    avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
    print(f"\nAverage accuracy: {avg_accuracy:.2f}%")
    
    good_heads = sum(1 for r in results.values() if r['accuracy'] >= 70)
    fair_heads = sum(1 for r in results.values() if 50 <= r['accuracy'] < 70)
    poor_heads = sum(1 for r in results.values() if r['accuracy'] < 50)
    
    print(f"\nHead status:")
    print(f"   ✅ Good (≥70%): {good_heads}/{len(results)}")
    print(f"   ⚠️ Fair (50-70%): {fair_heads}/{len(results)}")
    print(f"   ❌ Poor (<50%): {poor_heads}/{len(results)}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS:")
    print("=" * 80)
    
    if poor_heads > 0:
        print("\n⚠️ CRITICAL ISSUES DETECTED:")
        for head_name, stats in results.items():
            if stats['accuracy'] < 50:
                print(f"   → {head_name}: {stats['accuracy']:.1f}% accuracy")
                print(f"     Recommendation: Weighted loss + class balancing")
    
    if good_heads >= len(results) * 0.7:
        print("\n✅ PHASE 1 LOOKS GOOD!")
        print("   Most heads learned correctly.")
        print("   Consider proceeding to Phase 2.")
    elif fair_heads + good_heads >= len(results) * 0.7:
        print("\n⚠️ PHASE 1 NEEDS OPTIMIZATION:")
        print("   Some heads underperforming.")
        print("   Recommend: Weighted loss + 20 more epochs.")
    else:
        print("\n❌ PHASE 1 NEEDS MAJOR FIX:")
        print("   Multiple heads failing.")
        print("   Recommend: Full re-train with fixes.")
    
    print("=" * 80)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='logs/checkpoints_phase1/phase1-epoch=44-val_loss=0.0982.ckpt')
    parser.add_argument('--data_dir', default='data/preprocessed_phase1')
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()
    
    print("🔧 Loading checkpoint...")
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval()
    print("✅ Checkpoint loaded")
    
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
    print("✅ Tokenizer ready")
    
    # Run evaluation
    results = evaluate_reconstruction_accuracy(
        model, args.data_dir, tokenizer, 
        num_samples=args.num_samples
    )
    
    # Save results
    if results:
        output_file = Path('evaluation_results.txt')
        with open(output_file, 'w') as f:
            f.write("PHASE 1 EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            for head_name, stats in results.items():
                f.write(f"{head_name}: {stats['accuracy']:.2f}%\n")
        
        print(f"\n💾 Results saved to: {output_file}")


if __name__ == '__main__':
    main()
