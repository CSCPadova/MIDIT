"""
Per-Class Accuracy Analysis - Identify Mode Collapse
"""

import torch
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from midit.model.masked_transformer import MaskedTransformerREMI
from midit.data.lakh_dataset import LakhDataset
from miditok import REMI
from omegaconf import OmegaConf



def load_model(checkpoint_path):
    """Load trained model from checkpoint."""
    print(f"\n🔧 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load config
    cfg = OmegaConf.load('conf/config.yaml')
    
    # ✅ Override BEFORE model init
    OmegaConf.set_struct(cfg, False)
    cfg.model.vocab_sizes.pitch = 151
    cfg.model.vocab_sizes.velocity = 32
    cfg.model.vocab_sizes.duration = 32
    cfg.model.vocab_sizes.program = 129
    cfg.model.vocab_sizes.position = 96
    cfg.model.vocab_sizes.bar = 1
    cfg.model.vocab_sizes.tempo = 32
    cfg.model.vocab_sizes.timesig = 9
    cfg.model.vocab_sizes.rest = 32
    OmegaConf.set_struct(cfg, True)
    
    # Initialize model
    model = MaskedTransformerREMI(cfg)
    
    # ✅ Load with strict=False (ignore size mismatch for embeddings)
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['state_dict'], 
        strict=False  # ← KEY!
    )
    
    if missing_keys:
        print(f"⚠️ Missing keys: {missing_keys[:3]}...")
    if unexpected_keys:
        print(f"⚠️ Unexpected keys: {unexpected_keys[:3]}...")
    
    model.eval()
    print("✅ Model loaded (with size mismatches ignored)")
    return model

def compute_per_class_accuracy(model, dataset, num_samples=500):
    """Compute accuracy for each individual class."""
    
    print(f"\n{'='*70}")
    print("🎯 PER-CLASS ACCURACY ANALYSIS")
    print(f"{'='*70}")
    print(f"\n📊 Evaluating {num_samples} samples...")
    
    # Track predictions vs targets per class
    results = {
        'pitch': defaultdict(lambda: {'correct': 0, 'total': 0, 'frequency': 0}),
        'velocity': defaultdict(lambda: {'correct': 0, 'total': 0, 'frequency': 0}),
        'duration': defaultdict(lambda: {'correct': 0, 'total': 0, 'frequency': 0}),
        'program': defaultdict(lambda: {'correct': 0, 'total': 0, 'frequency': 0}),
        'position': defaultdict(lambda: {'correct': 0, 'total': 0, 'frequency': 0}),
    }
    
    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset)))):
            sample = dataset[i]
            if sample is None:
                continue
            
            input_ids = sample['input_ids'].unsqueeze(0)  # [1, seq_len]
            targets = sample['targets']
            mask_labels = sample['mask_labels']
            
            # Forward pass
            outputs = model(input_ids, mask_labels.unsqueeze(0))
            
            # Analyze each head
            for token_type in results.keys():
                logits = outputs[token_type][0]  # [seq_len, vocab_size]
                target = targets[token_type]
                
                # Only evaluate masked positions
                masked_positions = mask_labels.bool()
                
                for pos in range(len(target)):
                    if not masked_positions[pos]:
                        continue
                    
                    target_class = target[pos].item()
                    if target_class == -100:
                        continue
                    
                    pred_class = logits[pos].argmax().item()
                    
                    # Update per-class stats
                    results[token_type][target_class]['total'] += 1
                    results[token_type][target_class]['frequency'] += 1
                    
                    if pred_class == target_class:
                        results[token_type][target_class]['correct'] += 1
    
    return results


def analyze_results(results, dataset):
    """Analyze and print per-class accuracy results."""
    
    print(f"\n{'='*70}")
    print("📊 PER-CLASS ACCURACY RESULTS")
    print(f"{'='*70}")
    
    for token_type, class_stats in results.items():
        if len(class_stats) == 0:
            continue
        
        print(f"\n{'='*70}")
        print(f"📈 {token_type.upper()} HEAD")
        print(f"{'='*70}")
        
        # Compute accuracies
        class_accuracies = []
        class_frequencies = []
        class_names = []
        
        for class_id, stats in sorted(class_stats.items()):
            total = stats['total']
            if total == 0:
                continue
            
            accuracy = stats['correct'] / total * 100
            frequency = stats['frequency']
            
            # Get token name
            if class_id in dataset.id_to_token:
                token_name = dataset.id_to_token[class_id]
            else:
                # Reconstruct from type-specific lists
                if token_type == 'velocity' and class_id < len(dataset.velocity_tokens):
                    token_name = dataset.velocity_tokens[class_id]
                elif token_type == 'program' and class_id < len(dataset.program_tokens):
                    token_name = dataset.program_tokens[class_id]
                else:
                    token_name = f"{token_type}_{class_id}"
            
            class_accuracies.append(accuracy)
            class_frequencies.append(frequency)
            class_names.append((token_name, class_id))
        
        # Sort by frequency (descending)
        sorted_indices = np.argsort(class_frequencies)[::-1]
        
        # Print top 10 most common
        print("\n🔝 Top 10 Most Common Classes:")
        print(f"{'Class':<30} {'Frequency':<12} {'Accuracy':<10} Status")
        print("-" * 70)
        
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            name, class_id = class_names[idx]
            freq = class_frequencies[idx]
            acc = class_accuracies[idx]
            
            status = "✅" if acc > 50 else "⚠️" if acc > 20 else "❌"
            print(f"{name:<30} {freq:<12,} {acc:>6.2f}%    {status}")
        
        # Print bottom 10 least common
        print("\n⬇️ Bottom 10 Least Common Classes:")
        print(f"{'Class':<30} {'Frequency':<12} {'Accuracy':<10} Status")
        print("-" * 70)
        
        for i in range(max(0, len(sorted_indices) - 10), len(sorted_indices)):
            idx = sorted_indices[i]
            name, class_id = class_names[idx]
            freq = class_frequencies[idx]
            acc = class_accuracies[idx]
            
            status = "✅" if acc > 50 else "⚠️" if acc > 20 else "❌"
            print(f"{name:<30} {freq:<12,} {acc:>6.2f}%    {status}")
        
        # Statistics
        print(f"\n📊 Summary Statistics:")
        print(f"   Total classes evaluated: {len(class_accuracies)}")
        print(f"   Average accuracy: {np.mean(class_accuracies):.2f}%")
        print(f"   Std deviation: {np.std(class_accuracies):.2f}%")
        
        # Zero accuracy classes
        zero_acc_count = sum(1 for acc in class_accuracies if acc == 0)
        print(f"   Classes with 0% accuracy: {zero_acc_count}/{len(class_accuracies)} ({zero_acc_count/len(class_accuracies)*100:.1f}%)")
        
        # Correlation between frequency and accuracy
        if len(class_frequencies) > 1:
            correlation = np.corrcoef(class_frequencies, class_accuracies)[0, 1]
            print(f"   Correlation (freq ↔ accuracy): {correlation:.3f}")
            
            if correlation > 0.7:
                print(f"      ⚠️ STRONG positive correlation → MODE COLLAPSE!")
                print(f"      → Model predicts common classes well, ignores rare classes")
            elif correlation > 0.4:
                print(f"      ⚠️ MODERATE positive correlation → Bias toward common classes")
            else:
                print(f"      ✅ Weak correlation → Model generalizes to rare classes")


def plot_frequency_vs_accuracy(results, token_type='velocity'):
    """Plot frequency vs accuracy scatter plot."""
    
    if token_type not in results or len(results[token_type]) == 0:
        return
    
    class_stats = results[token_type]
    
    frequencies = []
    accuracies = []
    
    for class_id, stats in class_stats.items():
        if stats['total'] == 0:
            continue
        
        freq = stats['frequency']
        acc = stats['correct'] / stats['total'] * 100
        
        frequencies.append(freq)
        accuracies.append(acc)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(frequencies, accuracies, alpha=0.6, s=100)
    
    # Add trend line
    z = np.polyfit(frequencies, accuracies, 1)
    p = np.poly1d(z)
    plt.plot(frequencies, p(frequencies), "r--", alpha=0.8, linewidth=2)
    
    plt.xlabel('Class Frequency in Dataset', fontsize=12)
    plt.ylabel('Prediction Accuracy (%)', fontsize=12)
    plt.title(f'{token_type.upper()} - Frequency vs Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(frequencies, accuracies)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'per_class_analysis_{token_type}.png', dpi=150)
    print(f"\n💾 Plot saved: per_class_analysis_{token_type}.png")


def main():
    """Run per-class accuracy analysis."""
    
    print("="*70)
    print("🔬 PER-CLASS ACCURACY ANALYSIS")
    print("="*70)
    
    # Load model
    checkpoint_path = 'logs/checkpoints_phase1/phase1-epoch=44-val_loss=0.0982.ckpt'
    print(f"\n⚠️ Using BASELINE checkpoint (lambda hardcoded)")
    model = load_model(checkpoint_path)

    
    # Load dataset
    print("\n🔍 Loading dataset...")
    cfg = OmegaConf.load('conf/config.yaml')
    
    dataset = LakhDataset(
        split='val',
        data_dir=Path('data/preprocessed_phase1'),
        chunking_config=cfg.chunking,
        apply_masking=True,
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} files")
    
    # Compute per-class accuracy
    results = compute_per_class_accuracy(model, dataset, num_samples=500)
    
    # Analyze results
    analyze_results(results, dataset)
    
    # Plot for velocity and program (most problematic)
    print("\n📊 Generating plots...")
    plot_frequency_vs_accuracy(results, 'velocity')
    plot_frequency_vs_accuracy(results, 'program')
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()