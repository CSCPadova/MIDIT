"""
Extended Evaluation - Based on working evaluate_phase1_accuracy_DIRECT.py
Adds: Top-5 Accuracy
"""

import torch
from pathlib import Path
from tqdm import tqdm
from midit.model.masked_transformer import MaskedTransformerREMI
from miditok import REMI, TokenizerConfig
from collections import defaultdict
import numpy as np
import random
import argparse


def create_targets_dict(tokens, tokenizer):
    """Create targets dict aligned with LakhDataset._create_targets_dict.

    Uses sorted() on token name lists to match training class indices exactly.
    The original version used insertion order of id_to_token, which differs
    from sorted() for multi-digit names (e.g. Velocity_10 < Velocity_2 alphabetically).
    """
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    vocab_keys = list(tokenizer.vocab.keys())

    # Build sorted token lists — identical to LakhDataset
    pitch_tokens     = sorted([k for k in vocab_keys if k.startswith('Pitch_')])
    pitchdrum_tokens = sorted([k for k in vocab_keys if k.startswith('PitchDrum_')])
    velocity_tokens  = sorted([k for k in vocab_keys if k.startswith('Velocity_')])
    duration_tokens  = sorted([k for k in vocab_keys if k.startswith('Duration_')])
    program_tokens   = sorted([k for k in vocab_keys if k.startswith('Program_')])
    position_tokens  = sorted([k for k in vocab_keys if k.startswith('Position_')])
    bar_tokens       = sorted([k for k in vocab_keys if k.startswith('Bar_')])
    tempo_tokens     = sorted([k for k in vocab_keys if k.startswith('Tempo_')])
    timesig_tokens   = sorted([k for k in vocab_keys if k.startswith('TimeSig_')])
    rest_tokens      = sorted([k for k in vocab_keys if k.startswith('Rest_')])

    # Build name→class_index lookups (same logic as LakhDataset)
    pitch_to_class    = {t: i for i, t in enumerate(pitch_tokens)}
    pitchdrum_to_class = {t: len(pitch_tokens) + i for i, t in enumerate(pitchdrum_tokens)}
    velocity_to_class = {t: i for i, t in enumerate(velocity_tokens)}
    duration_to_class = {t: i for i, t in enumerate(duration_tokens)}
    program_to_class  = {t: i for i, t in enumerate(program_tokens)}
    position_to_class = {t: i for i, t in enumerate(position_tokens)}
    bar_to_class      = {t: i for i, t in enumerate(bar_tokens)}
    tempo_to_class    = {t: i for i, t in enumerate(tempo_tokens)}
    timesig_to_class  = {t: i for i, t in enumerate(timesig_tokens)}
    rest_to_class     = {t: i for i, t in enumerate(rest_tokens)}

    seq_len = len(tokens)
    targets = {
        'pitch':    torch.full((seq_len,), -100, dtype=torch.long),
        'velocity': torch.full((seq_len,), -100, dtype=torch.long),
        'duration': torch.full((seq_len,), -100, dtype=torch.long),
        'program':  torch.full((seq_len,), -100, dtype=torch.long),
        'position': torch.full((seq_len,), -100, dtype=torch.long),
        'bar':      torch.full((seq_len,), -100, dtype=torch.long),
        'tempo':    torch.full((seq_len,), -100, dtype=torch.long),
        'timesig':  torch.full((seq_len,), -100, dtype=torch.long),
        'rest':     torch.full((seq_len,), -100, dtype=torch.long),
    }

    for pos, token_id in enumerate(tokens.tolist()):
        if token_id not in id_to_token:
            continue
        token_name = id_to_token[token_id]

        if token_name in pitch_to_class:
            targets['pitch'][pos] = pitch_to_class[token_name]
        elif token_name in pitchdrum_to_class:
            targets['pitch'][pos] = pitchdrum_to_class[token_name]
        elif token_name in velocity_to_class:
            targets['velocity'][pos] = velocity_to_class[token_name]
        elif token_name in duration_to_class:
            targets['duration'][pos] = duration_to_class[token_name]
        elif token_name in program_to_class:
            targets['program'][pos] = program_to_class[token_name]
        elif token_name in position_to_class:
            targets['position'][pos] = position_to_class[token_name]
        elif token_name in bar_to_class:
            targets['bar'][pos] = bar_to_class[token_name]
        elif token_name in tempo_to_class:
            targets['tempo'][pos] = tempo_to_class[token_name]
        elif token_name in timesig_to_class:
            targets['timesig'][pos] = timesig_to_class[token_name]
        elif token_name in rest_to_class:
            targets['rest'][pos] = rest_to_class[token_name]

    return targets


def apply_masking(tokens, targets_dict, mask_ratio=0.15, mask_id=3):
    """Apply masking (from working script)."""
    seq_len = len(tokens)
    num_mask = max(1, int(seq_len * mask_ratio))
    mask_positions = random.sample(range(seq_len), num_mask)
    
    input_ids = tokens.clone()
    input_ids[mask_positions] = mask_id
    
    masked_targets = {}
    for head_name, target in targets_dict.items():
        masked_target = torch.full_like(target, -100)
        masked_target[mask_positions] = target[mask_positions]
        masked_targets[head_name] = masked_target
    
    return input_ids, masked_targets


def evaluate_top1_and_top5(model, data_dir, tokenizer, num_samples=100, split='val'):
    """
    Evaluate BOTH Top-1 and Top-5 accuracy.
    Uses exact logic from working script + adds top-5.
    """

    print("=" * 80)
    print(f"🔬 PHASE 1 - TOP-1 & TOP-5 ACCURACY  [split={split}]")
    print("=" * 80)

    model.eval()
    device = next(model.parameters()).device
    mask_id = 3

    # Get files
    data_path = Path(data_dir)
    all_files = sorted(data_path.glob("*.pt"))

    # Replicate exact split logic from LakhDataset (seed=42, 80/10/10)
    _rng = np.random.RandomState(42)
    indices = np.arange(len(all_files))
    _rng.shuffle(indices)
    n_train = int(0.8 * len(all_files))
    n_val   = int(0.1 * len(all_files))
    if split == 'train':
        split_indices = indices[:n_train]
    elif split == 'val':
        split_indices = indices[n_train:n_train + n_val]
    elif split == 'test':
        split_indices = indices[n_train + n_val:]
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'")
    split_files = [all_files[i] for i in split_indices]
    print(f"\n📂 Split '{split}': {len(split_files)} files (from {len(all_files)} total)")

    if len(split_files) < num_samples:
        print(f"⚠️ Only {len(split_files)} files in split, using all")
        sample_files = split_files
    else:
        sample_files = random.sample(split_files, num_samples)
    
    print(f"\n📊 Testing {len(sample_files)} files...\n")
    
    # Statistics
    top1_correct = defaultdict(int)
    top5_correct = defaultdict(int)
    total = defaultdict(int)
    
    with torch.no_grad():
        for file_path in tqdm(sample_files, desc="Evaluating"):
            try:
                # Load file
                data = torch.load(file_path)
                tokens = data['tokens']
                
                if len(tokens) < 50:
                    continue
                if len(tokens) > 512:
                    tokens = tokens[:512]
                
                # Create targets
                targets_dict = create_targets_dict(tokens, tokenizer)
                
                # Apply masking
                input_ids, masked_targets = apply_masking(
                    tokens, targets_dict, 
                    mask_ratio=0.15, 
                    mask_id=mask_id
                )
                
                # To device
                input_ids = input_ids.unsqueeze(0).to(device)
                
                # ✅ SAME AS WORKING SCRIPT
                hidden_states = model(input_ids)
                logits_dict = model.multi_head_output(hidden_states)
                
                # Evaluate each head
                for head_name, logits in logits_dict.items():
                    if head_name not in masked_targets:
                        continue
                    
                    target = masked_targets[head_name].to(device)
                    
                    # Top-1 predictions
                    top1_pred = torch.argmax(logits, dim=-1).squeeze(0)
                    
                    # ✅ NEW: Top-5 predictions
                    vocab_size = logits.size(-1)
                    k = min(5, vocab_size)
                    top5_preds = torch.topk(logits.squeeze(0), k=k, dim=-1).indices  # [seq_len, k]
                    
                    # Valid positions
                    valid_mask = (target != -100)
                    
                    if valid_mask.sum() == 0:
                        continue
                    
                    # Get valid predictions and targets
                    top1_pred_valid = top1_pred[valid_mask]
                    top5_preds_valid = top5_preds[valid_mask]
                    target_valid = target[valid_mask]
                    
                    # Top-1 accuracy
                    top1_correct[head_name] += (top1_pred_valid == target_valid).sum().item()
                    
                    # ✅ NEW: Top-5 accuracy
                    for i in range(len(target_valid)):
                        if target_valid[i].item() in top5_preds_valid[i].cpu().tolist():
                            top5_correct[head_name] += 1
                    
                    total[head_name] += valid_mask.sum().item()
                    
            except Exception as e:
                print(f"\n⚠️ Error {file_path.name}: {e}")
                continue
    
    # Compute results
    print("\n" + "=" * 80)
    print("📊 RESULTS: TOP-1 vs TOP-5")
    print("=" * 80 + "\n")
    
    results = {}
    
    # Header
    print(f"{'Token Type':<12} {'Top-1':>10} {'Top-5':>10} {'Gain':>10} {'Status':<20}")
    print("-" * 80)
    
    for head_name in sorted(total.keys()):
        if total[head_name] == 0:
            continue
        
        top1_acc = top1_correct[head_name] / total[head_name] * 100
        top5_acc = top5_correct[head_name] / total[head_name] * 100
        gain = top5_acc - top1_acc
        
        results[head_name] = {
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'gain': gain,
            'total': total[head_name]
        }
        
        # Status
        if gain > 20:
            status = "💡 Close"
        elif gain > 10:
            status = "⚠️ Near"
        else:
            status = "❌ Far"
        
        print(f"{head_name.upper():<12} {top1_acc:>9.2f}% {top5_acc:>9.2f}% {gain:>9.2f}% {status:<20}")
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 SUMMARY:")
    print("=" * 80)
    
    if results:
        avg_top1 = np.mean([r['top1_accuracy'] for r in results.values()])
        avg_top5 = np.mean([r['top5_accuracy'] for r in results.values()])
        avg_gain = avg_top5 - avg_top1
        
        print(f"\nAverage Top-1:  {avg_top1:.2f}%")
        print(f"Average Top-5:  {avg_top5:.2f}%")
        print(f"Average Gain:   +{avg_gain:.2f}%")
        
        if avg_gain > 20:
            print(f"\n✅ Model has good representations, needs confidence calibration")
        elif avg_gain > 10:
            print(f"\n⚠️ Model has moderate understanding, needs more training")
        else:
            print(f"\n❌ Model has weak representations, fundamental issues")
    
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    print("🔧 Loading checkpoint...")
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval()
    print("✅ Loaded\n")
    
    print("🔤 Initializing tokenizer...")
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_controls=False,
        delete_equal_successive_tempo_changes=True,
        delete_equal_successive_time_sig_changes=True,
    )
    tokenizer = REMI(tokenizer_config=config)
    print("✅ Ready\n")
    
    # Evaluate
    results = evaluate_top1_and_top5(model, args.data_dir, tokenizer, args.num_samples, args.split)
    
    # Save
    if results:
        with open('phase1_extended_results.txt', 'w') as f:
            f.write("PHASE 1 - TOP-1 & TOP-5 ACCURACY\n")
            f.write("=" * 80 + "\n\n")
            
            for name, stats in sorted(results.items()):
                f.write(f"{name.upper():<12}\n")
                f.write(f"  Top-1: {stats['top1_accuracy']:>6.2f}%\n")
                f.write(f"  Top-5: {stats['top5_accuracy']:>6.2f}%\n")
                f.write(f"  Gain:  +{stats['gain']:>5.2f}%\n\n")
            
            avg_top1 = np.mean([r['top1_accuracy'] for r in results.values()])
            avg_top5 = np.mean([r['top5_accuracy'] for r in results.values()])
            f.write(f"\nAverage Top-1: {avg_top1:.2f}%\n")
            f.write(f"Average Top-5: {avg_top5:.2f}%\n")
        
        print("\n💾 Results saved: phase1_extended_results.txt")


if __name__ == '__main__':
    main()