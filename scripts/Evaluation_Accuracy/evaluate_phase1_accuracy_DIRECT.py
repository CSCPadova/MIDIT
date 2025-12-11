"""
Valuta accuracy caricando DIRETTAMENTE i file .pt.
Salta completamente LakhDataset, carica raw .pt files.
"""
import torch
from pathlib import Path
from tqdm import tqdm
from midit.model.masked_transformer import MaskedTransformerREMI
from miditok import REMI, TokenizerConfig
from collections import defaultdict
import numpy as np
import random

def create_targets_dict(tokens, tokenizer):
    """
    Crea targets dict come fa il dataset durante training.
    Separa tokens per tipo e crea array con -100 per non-target positions.
    """
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    seq_len = len(tokens)
    
    # Initialize targets
    targets = {
        'pitch': torch.full((seq_len,), -100, dtype=torch.long),
        'velocity': torch.full((seq_len,), -100, dtype=torch.long),
        'duration': torch.full((seq_len,), -100, dtype=torch.long),
        'program': torch.full((seq_len,), -100, dtype=torch.long),
        'position': torch.full((seq_len,), -100, dtype=torch.long),
        'bar': torch.full((seq_len,), -100, dtype=torch.long),
        'tempo': torch.full((seq_len,), -100, dtype=torch.long),
        'timesig': torch.full((seq_len,), -100, dtype=torch.long),
        'rest': torch.full((seq_len,), -100, dtype=torch.long),
    }
    
    # Map head name to vocab offset
    head_vocab_offsets = {
        'pitch': 0,
        'velocity': 0,
        'duration': 0,
        'program': 0,
        'position': 0,
        'bar': 0,
        'tempo': 0,
        'timesig': 0,
        'rest': 0,
    }
    
    # Build reverse mapping per head
    head_vocabs = {key: {} for key in targets.keys()}
    
    for token_id, token_name in id_to_token.items():
        if token_name.startswith('Pitch_') or token_name.startswith('PitchDrum_'):
            head_vocabs['pitch'][token_id] = len(head_vocabs['pitch'])
        elif token_name.startswith('Velocity_'):
            head_vocabs['velocity'][token_id] = len(head_vocabs['velocity'])
        elif token_name.startswith('Duration_'):
            head_vocabs['duration'][token_id] = len(head_vocabs['duration'])
        elif token_name.startswith('Program_'):
            head_vocabs['program'][token_id] = len(head_vocabs['program'])
        elif token_name.startswith('Position_'):
            head_vocabs['position'][token_id] = len(head_vocabs['position'])
        elif token_name.startswith('Bar_'):
            head_vocabs['bar'][token_id] = len(head_vocabs['bar'])
        elif token_name.startswith('Tempo_'):
            head_vocabs['tempo'][token_id] = len(head_vocabs['tempo'])
        elif token_name.startswith('TimeSig_'):
            head_vocabs['timesig'][token_id] = len(head_vocabs['timesig'])
        elif token_name.startswith('Rest_'):
            head_vocabs['rest'][token_id] = len(head_vocabs['rest'])
    
    # Fill targets
    for i, token_id in enumerate(tokens.tolist()):
        if token_id not in id_to_token:
            continue
        
        token_name = id_to_token[token_id]
        
        for head_name in targets.keys():
            if token_id in head_vocabs[head_name]:
                targets[head_name][i] = head_vocabs[head_name][token_id]
                break
    
    return targets


def apply_masking(tokens, targets_dict, mask_ratio=0.15, mask_id=3):
    """
    Applica masking come durante training.
    Maschera mask_ratio% dei token TOTALI.
    """
    seq_len = len(tokens)
    num_mask = max(1, int(seq_len * mask_ratio))
    
    # Random positions to mask
    mask_positions = random.sample(range(seq_len), num_mask)
    
    # Create masked input
    input_ids = tokens.clone()
    input_ids[mask_positions] = mask_id
    
    # Mantieni targets solo per posizioni mascherate
    masked_targets = {}
    for head_name, target in targets_dict.items():
        masked_target = torch.full_like(target, -100)
        masked_target[mask_positions] = target[mask_positions]
        masked_targets[head_name] = masked_target
    
    return input_ids, masked_targets


def evaluate_reconstruction(model, data_dir, tokenizer, num_samples=100):
    """Valuta reconstruction accuracy."""
    print("=" * 80)
    print("🔬 PHASE 1 RECONSTRUCTION ACCURACY")
    print("=" * 80)
    
    model.eval()
    device = next(model.parameters()).device
    mask_id = 3
    
    # Get files
    data_path = Path(data_dir)
    all_files = sorted(data_path.glob("*.pt"))
    
    if len(all_files) < num_samples:
        print(f"⚠️ Only {len(all_files)} files available, using all")
        sample_files = all_files
    else:
        sample_files = random.sample(all_files, num_samples)
    
    print(f"\n📊 Testing {len(sample_files)} files...")
    
    # Statistics
    correct = defaultdict(int)
    total = defaultdict(int)
    unique_preds = defaultdict(set)
    unique_targets = defaultdict(set)
    
    with torch.no_grad():
        for file_path in tqdm(sample_files, desc="Evaluating"):
            try:
                # Load preprocessed file
                data = torch.load(file_path)
                tokens = data['tokens']
                
                # Skip short sequences
                if len(tokens) < 50:
                    continue
                
                # Truncate long sequences
                if len(tokens) > 512:
                    tokens = tokens[:512]
                
                # Create targets dict
                targets_dict = create_targets_dict(tokens, tokenizer)
                
                # Apply masking
                input_ids, masked_targets = apply_masking(
                    tokens, targets_dict, 
                    mask_ratio=0.15, 
                    mask_id=mask_id
                )
                
                # To device
                input_ids = input_ids.unsqueeze(0).to(device)
                
                # Forward pass
                hidden_states = model(input_ids)
                logits_dict = model.multi_head_output(hidden_states)
                
                # Evaluate each head
                for head_name, logits in logits_dict.items():
                    if head_name not in masked_targets:
                        continue
                    
                    target = masked_targets[head_name].to(device)
                    
                    # Predictions
                    pred = torch.argmax(logits, dim=-1).squeeze(0)
                    
                    # Valid positions (target != -100)
                    valid_mask = (target != -100)
                    
                    if valid_mask.sum() == 0:
                        continue
                    
                    pred_valid = pred[valid_mask]
                    target_valid = target[valid_mask]
                    
                    # Accuracy
                    correct[head_name] += (pred_valid == target_valid).sum().item()
                    total[head_name] += valid_mask.sum().item()
                    
                    # Diversity
                    unique_preds[head_name].update(pred_valid.cpu().tolist())
                    unique_targets[head_name].update(target_valid.cpu().tolist())
                    
            except Exception as e:
                print(f"\n⚠️ Error {file_path.name}: {e}")
                continue
    
    # Results
    print("\n" + "=" * 80)
    print("📊 RESULTS:")
    print("=" * 80)
    
    results = {}
    
    for head_name in sorted(total.keys()):
        if total[head_name] == 0:
            continue
        
        acc = correct[head_name] / total[head_name] * 100
        
        results[head_name] = {
            'accuracy': acc,
            'correct': correct[head_name],
            'total': total[head_name],
            'unique_pred': len(unique_preds[head_name]),
            'unique_target': len(unique_targets[head_name])
        }
        
        # Status
        if acc >= 70:
            status = "✅ GOOD"
        elif acc >= 50:
            status = "⚠️ FAIR"
        else:
            status = "❌ POOR"
        
        print(f"\n{head_name.upper():12} {status}")
        print(f"   Accuracy:  {acc:>6.2f}% ({correct[head_name]:>7,} / {total[head_name]:>7,})")
        print(f"   Diversity: {len(unique_preds[head_name]):>3} predicted (target: {len(unique_targets[head_name]):>3})")
        
        if len(unique_preds[head_name]) == 1:
            print(f"   ⚠️ MODE COLLAPSE!")
        elif len(unique_preds[head_name]) < 5:
            print(f"   ⚠️ LOW DIVERSITY")
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 SUMMARY:")
    print("=" * 80)
    
    if not results:
        print("\n❌ No results!")
        return {}
    
    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    print(f"\nAverage accuracy: {avg_acc:.2f}%")
    
    good = sum(1 for r in results.values() if r['accuracy'] >= 70)
    fair = sum(1 for r in results.values() if 50 <= r['accuracy'] < 70)
    poor = sum(1 for r in results.values() if r['accuracy'] < 50)
    
    print(f"\n✅ Good (≥70%): {good}/{len(results)}")
    print(f"⚠️ Fair (50-70%): {fair}/{len(results)}")
    print(f"❌ Poor (<50%): {poor}/{len(results)}")
    
    print("=" * 80)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()
    
    print("🔧 Loading checkpoint...")
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval()
    print("✅ Loaded\n")
    
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
    print("✅ Ready\n")
    
    # Evaluate
    results = evaluate_reconstruction(model, args.data_dir, tokenizer, args.num_samples)
    
    # Save
    if results:
        with open('phase1_accuracy_results.txt', 'w') as f:
            f.write("PHASE 1 RECONSTRUCTION ACCURACY\n")
            f.write("=" * 80 + "\n\n")
            for name, stats in sorted(results.items()):
                f.write(f"{name:12} {stats['accuracy']:>6.2f}%\n")
            f.write(f"\nAverage: {np.mean([r['accuracy'] for r in results.values()]):.2f}%\n")
        
        print("\n💾 Results saved: phase1_accuracy_results.txt")


if __name__ == '__main__':
    main()
