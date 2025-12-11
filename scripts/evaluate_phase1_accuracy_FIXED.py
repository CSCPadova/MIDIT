"""
Valuta usando IL VERO DATASET di training.
Non mascherare manualmente, usa dataset che sa come fare!
"""
import torch
from pathlib import Path
from tqdm import tqdm
from midit.model.masked_transformer import MaskedTransformerREMI
from midit.data.lakh_dataset import LakhDataset
from miditok import REMI, TokenizerConfig
from collections import defaultdict
import numpy as np

def evaluate_with_real_dataset(model, dataset, num_samples=100):
    """
    Usa IL VERO dataset con masking corretto.
    """
    print("=" * 80)
    print("🔬 PHASE 1 ACCURACY - Using Real Dataset")
    print("=" * 80)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Statistics
    correct = defaultdict(int)
    total = defaultdict(int)
    unique_preds = defaultdict(set)
    unique_targets = defaultdict(set)
    
    print(f"\n📊 Testing on {num_samples} samples...\n")
    
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Evaluating"):
            try:
                # Get sample from REAL dataset
                sample = dataset[i]
                
                input_ids = sample['input_ids'].unsqueeze(0).to(device)
                targets_dict = sample['targets']
                
                # Forward pass
                hidden_states = model(input_ids)
                logits_dict = model.multi_head_output(hidden_states)
                
                # Evaluate each head
                for head_name in logits_dict.keys():
                    if head_name not in targets_dict:
                        continue
                    
                    logits = logits_dict[head_name]
                    target = targets_dict[head_name].to(device).squeeze(0)
                    
                    # Predictions
                    pred = torch.argmax(logits, dim=-1).squeeze(0)
                    
                    # Valid positions (not -100)
                    valid_mask = (target != -100)
                    
                    if valid_mask.sum() == 0:
                        continue
                    
                    pred_valid = pred[valid_mask]
                    target_valid = target[valid_mask]
                    
                    # Accuracy
                    correct[head_name] += (pred_valid == target_valid).sum().item()
                    total[head_name] += valid_mask.sum().item()
                    
                    # Track unique values
                    unique_preds[head_name].update(pred_valid.cpu().tolist())
                    unique_targets[head_name].update(target_valid.cpu().tolist())
                    
            except Exception as e:
                print(f"\n⚠️ Error sample {i}: {e}")
                continue
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 RECONSTRUCTION ACCURACY PER HEAD:")
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
        
        # Mode collapse check
        if len(unique_preds[head_name]) == 1:
            print(f"   ⚠️ MODE COLLAPSE!")
        elif len(unique_preds[head_name]) < 5:
            print(f"   ⚠️ LOW DIVERSITY")
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 SUMMARY:")
    print("=" * 80)
    
    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    print(f"\nAverage accuracy: {avg_acc:.2f}%")
    
    good = sum(1 for r in results.values() if r['accuracy'] >= 70)
    fair = sum(1 for r in results.values() if 50 <= r['accuracy'] < 70)
    poor = sum(1 for r in results.values() if r['accuracy'] < 50)
    
    print(f"\n✅ Good (≥70%): {good}/{len(results)}")
    print(f"⚠️ Fair (50-70%): {fair}/{len(results)}")
    print(f"❌ Poor (<50%): {poor}/{len(results)}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 NEXT STEPS:")
    print("=" * 80)
    
    if good >= len(results) * 0.7:
        print("\n✅ Phase 1 is GOOD! Most heads work.")
        print("   → Proceed to Phase 2")
    elif poor > 0:
        print(f"\n⚠️ {poor} heads need fixing:")
        for name, stats in results.items():
            if stats['accuracy'] < 50:
                print(f"   → {name}: {stats['accuracy']:.1f}%")
        print("\n   Recommendation: Weighted loss + re-train")
    
    print("=" * 80)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()
    
    # Load model
    print("🔧 Loading checkpoint...")
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval()
    print("✅ Loaded")
    
    # Load dataset
    print("\n📂 Loading validation dataset...")
    dataset = LakhDataset(
        data_dir=args.data_dir,
        max_seq_len=512,
        mask_ratio=0.15,
        split='val'
    )
    print(f"✅ Loaded: {len(dataset)} samples")
    
    # Evaluate
    results = evaluate_with_real_dataset(model, dataset, args.num_samples)
    
    # Save
    with open('evaluation_results_REAL.txt', 'w') as f:
        for name, stats in results.items():
            f.write(f"{name}: {stats['accuracy']:.2f}%\n")
    
    print("\n💾 Saved: evaluation_results_REAL.txt")


if __name__ == '__main__':
    main()
