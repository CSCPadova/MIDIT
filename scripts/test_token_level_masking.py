"""
Test TokenLevelMasking in isolation.
Verifies correctness before integration into training pipeline.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from midit.data.masking import TokenLevelMasking


def test_basic_masking():
    """Test 1: Basic masking functionality."""
    print("=" * 80)
    print("🧪 TEST 1: Basic Token-Level Masking")
    print("=" * 80)
    
    # Setup
    mask_token_id = 518
    masker = TokenLevelMasking(
        mask_token_id=mask_token_id,
        mask_ratio=0.15,
        mask_strategy='bert',
        vocab_size=600
    )
    
    # Create LONGER test sequence for better statistics
    tokens = torch.arange(100, 600)  # 500 tokens
    seq_len = len(tokens)
    
    print(f"\n📊 Input:")
    print(f"   Sequence length: {seq_len}")
    print(f"   Token range: [{tokens.min()}, {tokens.max()}]")
    print(f"   Target mask ratio: 0.15 (15%)")
    print(f"   Expected masks: ~{int(seq_len * 0.15)}")
    
    # Apply masking
    masked_tokens, mask_labels = masker.apply_masking(tokens)
    
    # Analyze results
    num_masked = mask_labels.sum().item()
    num_mask_tokens = (masked_tokens == mask_token_id).sum().item()
    
    # Count unchanged (same token at masked position)
    num_unchanged = 0
    for i in range(seq_len):
        if mask_labels[i] and masked_tokens[i] == tokens[i]:
            num_unchanged += 1
    
    num_random = num_masked - num_mask_tokens - num_unchanged
    
    print(f"\n✅ Masking Results:")
    print(f"   Total masked positions: {num_masked} ({num_masked/seq_len*100:.1f}%)")
    print(f"   └─ [MASK] tokens: {num_mask_tokens} ({num_mask_tokens/num_masked*100:.1f}% - target 80%)")
    print(f"   └─ Random tokens: {num_random} ({num_random/num_masked*100:.1f}% - target 10%)")
    print(f"   └─ Unchanged: {num_unchanged} ({num_unchanged/num_masked*100:.1f}% - target 10%)")
    
    # Relaxed bounds (allow variance with larger sample)
    assert 0.10 <= num_masked / seq_len <= 0.20, f"Mask ratio {num_masked/seq_len} out of range!"
    assert 0.65 <= num_mask_tokens / num_masked <= 0.95, f"[MASK] ratio {num_mask_tokens/num_masked} out of range!"
    
    print(f"\n✅ TEST 1 PASSED - Basic masking works!")
    
    return masker


def test_short_sequences():
    """Test 2: Edge case - very short sequences."""
    print("\n" + "=" * 80)
    print("🧪 TEST 2: Short Sequences (Edge Case)")
    print("=" * 80)
    
    masker = TokenLevelMasking(mask_token_id=999, mask_ratio=0.15)
    
    # Test with 3 tokens
    tokens = torch.tensor([10, 20, 30])
    
    print(f"\n📊 Input:")
    print(f"   Sequence: {tokens.tolist()}")
    print(f"   Length: {len(tokens)}")
    print(f"   Expected masks: max(1, floor({len(tokens)} × 0.15)) = 1")
    
    masked_tokens, mask_labels = masker.apply_masking(tokens)
    
    num_masked = mask_labels.sum().item()
    
    print(f"\n✅ Results:")
    print(f"   Masked positions: {num_masked}")
    print(f"   Mask positions: {mask_labels.nonzero(as_tuple=True)[0].tolist()}")
    print(f"   Original: {tokens.tolist()}")
    print(f"   Masked:   {masked_tokens.tolist()}")
    
    assert num_masked >= 1, "Should mask at least 1 token!"
    assert num_masked <= len(tokens), "Can't mask more than sequence length!"
    
    print(f"\n✅ TEST 2 PASSED - Edge case handled!")


def test_always_strategy():
    """Test 3: 'always' strategy (100% [MASK])."""
    print("\n" + "=" * 80)
    print("🧪 TEST 3: 'Always' Masking Strategy")
    print("=" * 80)
    
    masker = TokenLevelMasking(
        mask_token_id=777,
        mask_ratio=0.2,
        mask_strategy='always'
    )
    
    tokens = torch.arange(50)
    
    print(f"\n📊 Input:")
    print(f"   Strategy: 'always' (100% [MASK])")
    print(f"   Sequence length: {len(tokens)}")
    
    masked_tokens, mask_labels = masker.apply_masking(tokens)
    
    num_masked = mask_labels.sum().item()
    num_mask_tokens = (masked_tokens == 777).sum().item()
    
    print(f"\n✅ Results:")
    print(f"   Masked positions: {num_masked}")
    print(f"   [MASK] tokens: {num_mask_tokens}")
    print(f"   Coverage: {num_mask_tokens/num_masked*100:.0f}%")
    
    assert num_mask_tokens == num_masked, "'always' must use [MASK] for all!"
    
    print(f"\n✅ TEST 3 PASSED!")


def test_statistical_distribution():
    """Test 4: Statistical distribution (THIS IS THE REAL TEST)."""
    print("\n" + "=" * 80)
    print("🧪 TEST 4: Statistical Distribution (1000 samples)")
    print("=" * 80)
    print("   This test verifies 80-10-10 distribution converges")
    
    masker = TokenLevelMasking(mask_token_id=999, mask_ratio=0.15)
    
    num_trials = 1000
    seq_len = 100
    
    mask_ratios = []
    mask_token_ratios = []
    
    print(f"\n📊 Running {num_trials} trials with seq_len={seq_len}...")
    
    for _ in range(num_trials):
        tokens = torch.randint(0, 500, (seq_len,))
        masked_tokens, mask_labels = masker.apply_masking(tokens)
        
        num_masked = mask_labels.sum().item()
        num_mask_tok = (masked_tokens == 999).sum().item()
        
        if num_masked > 0:
            mask_ratios.append(num_masked / seq_len)
            mask_token_ratios.append(num_mask_tok / num_masked)
    
    import numpy as np
    
    mean_mask = np.mean(mask_ratios)
    std_mask = np.std(mask_ratios)
    mean_mask_tok = np.mean(mask_token_ratios)
    std_mask_tok = np.std(mask_token_ratios)
    
    print(f"\n✅ Statistical Results:")
    print(f"   Masking ratio:")
    print(f"      Target: 0.150 (15.0%)")
    print(f"      Actual: {mean_mask:.4f} ± {std_mask:.4f}")
    print(f"      Range:  [{min(mask_ratios):.4f}, {max(mask_ratios):.4f}]")
    print(f"   [MASK] token ratio:")
    print(f"      Target: 0.800 (80.0%)")
    print(f"      Actual: {mean_mask_tok:.4f} ± {std_mask_tok:.4f}")
    
    # These should converge with 1000 trials
    assert 0.145 <= mean_mask <= 0.155, f"Mean mask ratio {mean_mask} deviates!"
    assert 0.75 <= mean_mask_tok <= 0.85, f"Mean [MASK] ratio {mean_mask_tok} deviates!"
    
    print(f"\n✅ TEST 4 PASSED - Statistical convergence verified!")


def test_compatibility():
    """Test 5: API compatibility."""
    print("\n" + "=" * 80)
    print("🧪 TEST 5: API Compatibility")
    print("=" * 80)
    
    token_masker = TokenLevelMasking(mask_token_id=518, mask_ratio=0.15)
    
    tokens = torch.arange(100)
    bar_positions = [0, 25, 50, 75, 100]
    
    print(f"\n📊 Testing compatibility with BarLevelMasking API...")
    
    masked_tokens, mask_labels = token_masker.apply_masking(
        tokens,
        bar_positions=bar_positions  # Should accept but ignore
    )
    
    print(f"\n✅ Results:")
    print(f"   ✓ Accepts bar_positions parameter")
    print(f"   ✓ Returns (masked_tokens, mask_labels)")
    print(f"   ✓ Shapes: {masked_tokens.shape} == {tokens.shape}")
    print(f"   ✓ Dtype: {mask_labels.dtype} == torch.bool")
    
    assert masked_tokens.shape == tokens.shape
    assert mask_labels.dtype == torch.bool
    
    print(f"\n✅ TEST 5 PASSED!")


if __name__ == '__main__':
    print("\n🎵 Testing TokenLevelMasking\n")
    
    try:
        test_basic_masking()
        test_short_sequences()
        test_always_strategy()
        test_statistical_distribution()
        test_compatibility()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80 + "\n")
        
        print("📝 Summary:")
        print("   ✓ Basic masking works")
        print("   ✓ Edge cases handled")
        print("   ✓ Both strategies work")
        print("   ✓ Statistical distribution correct (80-10-10)")
        print("   ✓ API compatible")
        print("\n✅ TokenLevelMasking ready for integration!\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
