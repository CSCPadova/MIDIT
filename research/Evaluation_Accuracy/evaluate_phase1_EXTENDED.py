"""
Extended Evaluation - Top-5 Accuracy
Based on working evaluate_phase1_accuracy_DIRECT.py
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
    """Create targets dict."""
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    seq_len = len(tokens)
    
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
    """Apply masking."""
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


def evaluate_top1_and_top5(model, data_dir, tokenizer, num_samples=100):
    """Evaluate Top-1 and Top-5 accuracy."""
    
    print("=" * 80)
    print("🔬 PHASE 1 - TOP-1 & TOP-5 ACCURACY")
    print("=" * 80)
    
    model.eval()
    device = next(model.parameters()).device
    mask_id = 3
    
    data_path = Path(data_dir)
    all_files = sorted(data_path.glob("*.pt"))
    
    if len(all_files) < num_samples:
        print(f"⚠️ Only {len(all_files)} files, using all")
        sample_files = all_files
    else:
        sample_files = random.sample(all_files, num_samples)
    
    print(f"\n📊 Testing {len(sample_files)} files...\n")
    
    top1_correct = defaultdict(int)
    top5_correct = defaultdict(int)
    total = defaultdict(int)
    
    with torch.no_grad():
        for file_path in tqdm(sample_files, desc="Evaluating"):
            try:
                data = torch.load(file_path)
                tokens = data['tokens']
                
                if len(tokens) < 50:
                    continue
                if len(tokens) > 512:
                    tokens = tokens[:512]
                
                targets_dict = create_targets_dict(tokens, tokenizer)
                input_ids, masked_targets = apply_masking(tokens, targets_dict, 0.15, mask_id)
                input_ids = input_ids.unsqueeze(0).to(device)
                
                # Forward (SAME AS WORKING SCRIPT)
                hidden_states = model(input_ids)
                logits_dict = model.multi_head_output(hidden_states)
                
                for head_name, logits in logits_dict.items():
                    if head_name not in masked_targets:
                        continue
                    
                    target = masked_targets[head_name].to(device)
                    top1_pred = torch.argmax(logits, dim=-1).squeeze(0)
                    
                    # Top-5
                    vocab_size = logits.size(-1)
                    k = min(5, vocab_size)
                    top5_preds = torch.topk(logits.squeeze(0), k=k, dim=-1).indices
                    
                    valid_mask = (target != -100)
                    if valid_mask.sum() == 0:
                        continue
                    
                    top1_pred_valid = top1_pred[valid_mask]
                    top5_preds_valid = top5_preds[valid_mask]
                    target_valid = target[valid_mask]
                    
                    # Count
                    top1_correct[head_name] += (top1_pred_valid == target_valid).sum().item()
                    
                    for i in range(len(target_valid)):
                        if target_valid[i].item() in top5_preds_valid[i].cpu().tolist():
                            top5_correct[head_name] += 1
                    
                    total[head_name] += valid_mask.sum().item()
                    
            except Exception as e:
                continue
    
    # Results
    print("\n" + "=" * 80)
    print("📊 RESULTS: TOP-1 vs TOP-5")
    print("=" * 80 + "\n")
    
    results = {}
    
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
        
        if gain > 20:
            status = "💡 Close"
        elif gain > 10:
            status = "⚠️ Near"
        else:
            status = "❌ Far"
        
        print(f"{head_name.upper():<12} {top1_acc:>9.2f}% {top5_acc:>9.2f}% {gain:>9.2f}% {status:<20}")
    
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
            print(f"\n✅ Good representations, needs calibration")
        elif avg_gain > 10:
            print(f"\n⚠️ Moderate, needs more training")
        else:
            print(f"\n❌ Weak representations")
    
    print("=" * 80)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()
    
    print("🔧 Loading checkpoint...")
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval()
    print("✅ Loaded\n")
    
    print("🔤 Tokenizer...")
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
    
    results = evaluate_top1_and_top5(model, args.data_dir, tokenizer, args.num_samples)
    
    if results:
        with open('phase1_extended_results.txt', 'w') as f:
            f.write("TOP-1 & TOP-5 ACCURACY\n")
            f.write("=" * 80 + "\n\n")
            for name, stats in sorted(results.items()):
                f.write(f"{name.upper():<12}\n")
                f.write(f"  Top-1: {stats['top1_accuracy']:>6.2f}%\n")
                f.write(f"  Top-5: {stats['top5_accuracy']:>6.2f}%\n")
                f.write(f"  Gain:  +{stats['gain']:>5.2f}%\n\n")
        
        print("\n💾 Saved: phase1_extended_results.txt")


if __name__ == '__main__':
    main()