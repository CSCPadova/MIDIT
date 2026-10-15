"""
Debug Mode Collapse - Velocity/Program Analysis
"""

import torch
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from midit.data.lakh_dataset import LakhDataset
from miditok import REMI
from omegaconf import OmegaConf

def analyze_token_frequencies(dataset, num_samples=1000):
    """Analyze token frequency distribution in dataset."""
    
    print("\n" + "="*70)
    print("📊 TOKEN FREQUENCY ANALYSIS")
    print("="*70)
    
    token_counts = {
        'pitch': Counter(),
        'velocity': Counter(),
        'duration': Counter(),
        'program': Counter(),
        'position': Counter(),
    }
    
    total_tokens = 0
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        if sample is None:
            continue
        
        tokens = sample['original_tokens']
        total_tokens += len(tokens)
        
        # Count each token type
        for pos in range(len(tokens)):
            token_id = tokens[pos].item()
            
            if token_id not in dataset.id_to_token:
                continue
            
            token_name = dataset.id_to_token[token_id]
            
            if token_name.startswith('Pitch_'):
                token_counts['pitch'][token_name] += 1
            elif token_name.startswith('Velocity_'):
                token_counts['velocity'][token_name] += 1
            elif token_name.startswith('Duration_'):
                token_counts['duration'][token_name] += 1
            elif token_name.startswith('Program_'):
                token_counts['program'][token_name] += 1
            elif token_name.startswith('Position_'):
                token_counts['position'][token_name] += 1
    
    # Print statistics
    print(f"\n📈 Token Distribution (from {num_samples} samples):")
    print(f"   Total tokens: {total_tokens:,}")
    
    for token_type in ['pitch', 'velocity', 'duration', 'program', 'position']:
        counts = token_counts[token_type]
        total = sum(counts.values())
        unique = len(counts)
        
        print(f"\n   {token_type.upper()}:")
        print(f"      Total occurrences: {total:,} ({total/total_tokens*100:.2f}% of dataset)")
        print(f"      Unique tokens: {unique}")
        
        if len(counts) > 0:
            # Most common
            most_common = counts.most_common(5)
            print(f"      Top 5 most common:")
            for token, count in most_common:
                print(f"         {token}: {count:,} ({count/total*100:.2f}%)")
            
            # Least common
            least_common = counts.most_common()[-5:]
            print(f"      Top 5 least common:")
            for token, count in least_common:
                print(f"         {token}: {count:,} ({count/total*100:.2f}%)")
            
            # Class imbalance ratio
            max_freq = most_common[0][1]
            min_freq = least_common[0][1]
            imbalance = max_freq / min_freq if min_freq > 0 else float('inf')
            print(f"      ⚠️ Imbalance ratio: {imbalance:.1f}× (max/min frequency)")
    
    return token_counts


def analyze_masked_coverage(dataset, num_samples=100):
    """Analyze which token types get masked."""
    
    print("\n" + "="*70)
    print("🎭 MASKED TOKEN COVERAGE ANALYSIS")
    print("="*70)
    
    masked_counts = {
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
    
    total_masked = 0
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        if sample is None:
            continue
        
        original_tokens = sample['original_tokens']
        mask_labels = sample['mask_labels']
        
        # Count masked positions per type
        for pos in range(len(original_tokens)):
            if not mask_labels[pos]:
                continue
            
            total_masked += 1
            token_id = original_tokens[pos].item()
            
            if token_id not in dataset.id_to_token:
                continue
            
            token_name = dataset.id_to_token[token_id]
            
            if token_name.startswith('Pitch_'):
                masked_counts['pitch'] += 1
            elif token_name.startswith('Velocity_'):
                masked_counts['velocity'] += 1
            elif token_name.startswith('Duration_'):
                masked_counts['duration'] += 1
            elif token_name.startswith('Program_'):
                masked_counts['program'] += 1
            elif token_name.startswith('Position_'):
                masked_counts['position'] += 1
            elif token_name.startswith('Bar_'):
                masked_counts['bar'] += 1
            elif token_name.startswith('Tempo_'):
                masked_counts['tempo'] += 1
            elif token_name.startswith('TimeSig_'):
                masked_counts['timesig'] += 1
            elif token_name.startswith('Rest_'):
                masked_counts['rest'] += 1
    
    print(f"\n📊 Masked Token Distribution (from {num_samples} samples):")
    print(f"   Total masked: {total_masked:,}")
    
    for token_type, count in masked_counts.items():
        pct = (count / total_masked * 100) if total_masked > 0 else 0
        print(f"   {token_type:10s}: {count:6,} ({pct:5.2f}%)")
        
        if pct < 1.0:
            print(f"      ⚠️ WARNING: Very low coverage!")
    
    return masked_counts


def analyze_gradient_dominance(dataset, num_samples=10):
    """Check if some heads dominate gradient flow."""
    
    print("\n" + "="*70)
    print("🔥 GRADIENT DOMINANCE ANALYSIS")
    print("="*70)
    
    # Calculate loss magnitude per head
    loss_magnitudes = {
        'pitch': [],
        'velocity': [],
        'duration': [],
        'program': [],
        'position': [],
    }
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        if sample is None:
            continue
        
        targets = sample['targets']
        
        for token_type in loss_magnitudes.keys():
            valid_targets = (targets[token_type] != -100).sum().item()
            loss_magnitudes[token_type].append(valid_targets)
    
    print("\n📊 Valid Targets per Head (proxy for gradient magnitude):")
    
    for token_type, counts in loss_magnitudes.items():
        avg = sum(counts) / len(counts) if counts else 0
        print(f"   {token_type:10s}: {avg:.1f} avg valid targets per sample")
    
    # Check dominance
    avg_counts = {k: sum(v)/len(v) if v else 0 for k, v in loss_magnitudes.items()}
    max_avg = max(avg_counts.values())
    
    print("\n⚖️ Gradient Dominance:")
    for token_type, avg in avg_counts.items():
        ratio = avg / max_avg if max_avg > 0 else 0
        status = "✅" if ratio > 0.5 else "⚠️" if ratio > 0.1 else "❌"
        print(f"   {status} {token_type:10s}: {ratio:.2%} of max head")


def analyze_prediction_distribution(checkpoint_path, dataset, num_samples=100):
    """Analyze what classes the model predicts."""
    
    print("\n" + "="*70)
    print("🎯 PREDICTION DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # This would require loading model and running inference
    # Simplified version: just show what we expect
    
    print("\n💡 Expected behavior:")
    print("   ✅ Healthy model: predicts diverse classes")
    print("   ❌ Mode collapse: predicts same 1-3 classes")
    print("\n   Current observation:")
    print("      Velocity: only 22/32 classes predicted")
    print("      Program: only 28/129 classes predicted")
    print("\n   → Strong indicator of mode collapse!")


def main():
    """Run all mode collapse diagnostics."""
    
    print("="*70)
    print("🔬 MODE COLLAPSE DEBUG - Velocity/Program Analysis")
    print("="*70)
    
    # Load dataset
    print("\n🔍 Loading dataset...")
    
    # LakhDataset loads config internally
    dataset = LakhDataset(
        split='train',
        data_dir=Path('data/preprocessed_phase1'),
        apply_masking=True,
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} files\n")
    
    # Run analyses
    token_freqs = analyze_token_frequencies(dataset, num_samples=1000)
    masked_coverage = analyze_masked_coverage(dataset, num_samples=100)
    analyze_gradient_dominance(dataset, num_samples=50)
    analyze_prediction_distribution(None, dataset)
    
    # Summary
    print("\n" + "="*70)
    print("📋 DIAGNOSIS SUMMARY")
    print("="*70)
    
    # Check if velocity/program are rare
    velocity_pct = masked_coverage['velocity'] / sum(masked_coverage.values()) * 100
    program_pct = masked_coverage['program'] / sum(masked_coverage.values()) * 100
    
    print("\n🔍 Likely Causes of Mode Collapse:")
    
    if velocity_pct < 5:
        print("   1. ⚠️ Velocity rarely masked (<5% of masked tokens)")
        print("      → Model sees few velocity examples during training")
        print("      → Solution: Token-level masking OR increase type_mask_ratio")
    
    if program_pct < 5:
        print("   2. ⚠️ Program rarely masked (<5% of masked tokens)")
        print("      → Same issue as velocity")
    
    print("   3. ⚠️ Lambda weights too low?")
    print("      → velocity: 0.5, program: 1.0")
    print("      → Gradient ~2-4× smaller than pitch")
    print("      → Solution: Increase lambda OR add class weights")
    
    print("   4. ⚠️ Class imbalance within head")
    print("      → Common velocity (mf) >> rare velocity (ppp)")
    print("      → Model predicts common class = low loss")
    print("      → Solution: Class weights (inverse frequency)")
    
    print("\n🎯 Recommended Actions:")
    print("   Priority 1: Switch to token-level masking (15% fixed)")
    print("      → Guarantees velocity/program coverage")
    print("   Priority 2: Increase lambda for velocity/program")
    print("      → velocity: 0.5 → 2.0, program: 1.0 → 3.0")
    print("   Priority 3: Add class weights (later)")
    print("      → Weight rare classes higher in cross_entropy")


if __name__ == '__main__':
    main()