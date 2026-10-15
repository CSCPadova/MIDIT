"""
Compute class weights for imbalanced MIDI token types.
Uses inverse frequency: weight = 1 / frequency
"""

import torch
from pathlib import Path
from collections import Counter
import json
from tqdm import tqdm
from miditok import REMI, TokenizerConfig


def compute_class_weights(data_dir, token_types=['velocity', 'program']):
    """
    Compute inverse frequency weights for specified token types.
    
    Args:
        data_dir: Directory with preprocessed .pt files
        token_types: List of token types to compute weights for
    
    Returns:
        Dictionary: {token_type: [weight_0, weight_1, ...]}
    """
    
    print("="*80)
    print("COMPUTING CLASS WEIGHTS - Inverse Frequency")
    print("="*80)
    
    # Initialize tokenizer
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
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    
    # Build token lists per type
    token_lists = {}
    for ttype in token_types:
        token_lists[ttype] = sorted([
            k for k in tokenizer.vocab.keys() 
            if k.lower().startswith(ttype.lower())
        ])
    
    print(f"\nToken vocabularies:")
    for ttype, tokens in token_lists.items():
        print(f"  {ttype}: {len(tokens)} tokens")
    
    # Count token occurrences
    counters = {ttype: Counter() for ttype in token_types}
    
    files = list(Path(data_dir).glob("*.pt"))
    print(f"\nProcessing {len(files)} files...\n")
    
    for file in tqdm(files, desc="Counting tokens"):
        try:
            data = torch.load(file)
            tokens = data['tokens']
            
            for token_id in tokens.tolist():
                if token_id not in id_to_token:
                    continue
                
                token_name = id_to_token[token_id]
                
                # Check which type this token belongs to
                for ttype in token_types:
                    if token_name in token_lists[ttype]:
                        counters[ttype][token_name] += 1
                        break
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
            continue
    
    # Compute inverse frequency weights
    weights = {}
    
    for ttype in token_types:
        counter = counters[ttype]
        token_list = token_lists[ttype]
        vocab_size = len(token_list)
        total = sum(counter.values())
        
        if total == 0:
            print(f"\n⚠️ No tokens found for {ttype}, skipping")
            continue
        
        # Compute weight for each token in vocabulary
        type_weights = []
        for token_name in token_list:
            freq = counter.get(token_name, 1)  # Avoid division by zero
            # Inverse frequency, normalized
            weight = total / (freq * vocab_size)
            type_weights.append(weight)
        
        weights[ttype] = type_weights
        
        # Statistics
        print(f"\n{ttype.upper()}:")
        print(f"  Total occurrences: {total:,}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Weight range: [{min(type_weights):.4f}, {max(type_weights):.4f}]")
        print(f"  Imbalance (max/min): {max(type_weights)/min(type_weights):.1f}×")
        
        # Show top 5 most/least common
        sorted_tokens = sorted(
            [(token_name, counter[token_name]) for token_name in token_list],
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"  Most common:")
        for token, count in sorted_tokens[:5]:
            pct = count / total * 100
            print(f"    {token}: {count:,} ({pct:.2f}%)")
        
        print(f"  Least common:")
        for token, count in sorted_tokens[-5:]:
            pct = count / total * 100
            print(f"    {token}: {count:,} ({pct:.2f}%)")
    
    return weights


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute inverse frequency class weights for MIDI tokens'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/preprocessed_phase1',
        help='Directory containing .pt files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='class_weights.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--token_types',
        nargs='+',
        default=['velocity', 'program'],
        help='Token types to compute weights for'
    )
    
    args = parser.parse_args()
    
    # Compute weights
    weights = compute_class_weights(args.data_dir, args.token_types)
    
    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Weights saved to: {output_path.absolute()}")
    print(f"{'='*80}\n")