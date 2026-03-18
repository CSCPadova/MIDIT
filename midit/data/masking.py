"""
Bar-Level Masking Strategy for MIDI Sequences
Following MusicBERT approach: mask all tokens of same type within a bar
"""

import torch
import random
from typing import Dict, List, Tuple, Optional


class BarLevelMasking:
    """Bar-level masking strategy for MLM pre-training.
    
    Following MusicBERT: mask all tokens of the same type within a bar
    to prevent information leakage.
    """
    
    def __init__(
        self,
        tokenizer_vocab: Dict[str, int],
        mask_token_id: int,
        bar_mask_prob: float = 0.15,
        type_mask_ratio: float = 0.3,
    ):
        """Initialize masking strategy.
        
        Args:
            tokenizer_vocab: Tokenizer vocabulary (token_name -> token_id)
            mask_token_id: ID of [MASK] token
            bar_mask_prob: Probability of masking a bar
            type_mask_ratio: Ratio of token types to mask within a bar
        """
        self.vocab = tokenizer_vocab
        self.mask_token_id = mask_token_id
        self.bar_mask_prob = bar_mask_prob
        self.type_mask_ratio = type_mask_ratio
        
        # Create reverse vocab (token_id -> token_name)
        self.id_to_token = {v: k for k, v in tokenizer_vocab.items()}
        
        # Token type categories
        self.token_types = {
            'pitch': ['Pitch_'],
            'velocity': ['Velocity_'],
            'duration': ['Duration_'],
            'program': ['Program_'],
            'position': ['Position_'],
            'bar': ['Bar_'],
            'tempo': ['Tempo_'],
            'timesig': ['TimeSig_'],
            'rest': ['Rest_'],
            'pitchdrum': ['PitchDrum_'],
        }
        
        print(f"✅ BarLevelMasking initialized")
        print(f"   Bar mask prob: {bar_mask_prob}")
        print(f"   Type mask ratio: {type_mask_ratio}")
        print(f"   Mask token ID: {mask_token_id}\n")
    
    def get_token_type(self, token_id: int) -> Optional[str]:
        """Get token type from token ID.
        
        Args:
            token_id: Token ID
        
        Returns:
            Token type string or None
        """
        if token_id not in self.id_to_token:
            return None
        
        token_name = self.id_to_token[token_id]
        
        # Find which type this token belongs to
        for type_name, prefixes in self.token_types.items():
            for prefix in prefixes:
                if token_name.startswith(prefix):
                    return type_name
        
        return None
    
    def apply_masking(
        self,
        tokens: torch.Tensor,
        bar_positions: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply bar-level masking to token sequence.
        
        Args:
            tokens: Token IDs [seq_len]
            bar_positions: Bar boundary positions
        
        Returns:
            Tuple of (masked_tokens, mask_labels):
                - masked_tokens: Tokens with [MASK] applied [seq_len]
                - mask_labels: 1 where masked, 0 otherwise [seq_len]
        """
        masked_tokens = tokens.clone()
        mask_labels = torch.zeros_like(tokens, dtype=torch.bool)
        
        num_bars = len(bar_positions) - 1
        
        # Iterate through bars
        for bar_idx in range(num_bars):
            # Decide if we mask this bar
            if random.random() > self.bar_mask_prob:
                continue
            
            start_pos = bar_positions[bar_idx]
            end_pos = bar_positions[bar_idx + 1]
            
            # Get tokens in this bar
            bar_tokens = tokens[start_pos:end_pos]
            
            # Find unique token types in this bar
            types_in_bar = set()
            for pos in range(len(bar_tokens)):
                token_type = self.get_token_type(bar_tokens[pos].item())
                if token_type is not None:
                    types_in_bar.add(token_type)
            
            if len(types_in_bar) == 0:
                continue
            
            # Select token types to mask
            num_types_to_mask = max(1, int(len(types_in_bar) * self.type_mask_ratio))
            types_to_mask = random.sample(list(types_in_bar), k=num_types_to_mask)
            
            # Mask all tokens of selected types in this bar
            for local_pos in range(len(bar_tokens)):
                global_pos = start_pos + local_pos
                token_type = self.get_token_type(tokens[global_pos].item())
                
                if token_type in types_to_mask:
                    masked_tokens[global_pos] = self.mask_token_id
                    mask_labels[global_pos] = 1
        
        return masked_tokens, mask_labels
    
    def get_masking_stats(
        self,
        mask_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Get statistics about masking.
        
        Args:
            mask_labels: Mask labels [seq_len]
        
        Returns:
            Dictionary with masking statistics
        """
        total_tokens = len(mask_labels)
        masked_tokens = mask_labels.sum().item()
        mask_ratio = masked_tokens / total_tokens if total_tokens > 0 else 0
        
        return {
            'total_tokens': total_tokens,
            'masked_tokens': masked_tokens,
            'mask_ratio': mask_ratio,
        }

# ============================================================================
# Token-Level Masking (NEW - Optimized for Rare Tokens)
# ============================================================================

class TokenLevelMasking:
    """
    Random token-level masking strategy (BERT-style).
    
    MOTIVATION:
    -----------
    Bar-level masking has a critical issue with rare tokens:
    
    Dataset token frequencies (from analysis):
    - Tempo:   0.14% of dataset (7,668 / 5.5M tokens)
    - TimeSig: 1.7% of dataset (95,826 / 5.5M tokens)
    
    With bar-level masking (15% bars masked):
    - Tempo/TimeSig in 15% of bars → masked
    - Model sees Tempo/TimeSig in only 85% × 0.14% = 0.12% of training
    - Result: Tempo accuracy = 0%, TimeSig accuracy = 0%
    
    With token-level masking (15% tokens masked):
    - ALL tokens masked uniformly with 15% probability
    - Model sees Tempo/TimeSig in 85% of training steps
    - Result: Much better learning for rare tokens
    
    TRADE-OFF:
    ----------
    - Bar-level: Preserves musical structure, but hurts rare token learning
    - Token-level: Better rare token learning, may break local coherence
    - For Phase 1 pre-training: token-level is better (learn representations)
    - For Phase 2 fine-tuning: bar-level might be better (musical coherence)
    
    IMPLEMENTATION:
    ---------------
    Follows BERT masking strategy (Devlin et al., 2018):
    - 80%: Replace with [MASK] token
    - 10%: Replace with random token  
    - 10%: Keep original token
    
    Why 80-10-10 split?
    - 80% [MASK]: Standard masked language modeling
    - 10% random: Prevents model from relying only on [MASK] signal
    - 10% unchanged: Helps with fine-tuning (no [MASK] in inference)
    """
    
    def __init__(
        self,
        mask_token_id: int,
        mask_ratio: float = 0.15,
        mask_strategy: str = 'bert',
        vocab_size: Optional[int] = None,
    ):
        """
        Initialize token-level masking.
        
        Args:
            mask_token_id: ID of [MASK] special token
            mask_ratio: Fraction of tokens to mask (default: 0.15)
                       Standard BERT value, balances learning vs. context
            mask_strategy: Masking distribution
                - 'bert': 80% [MASK], 10% random, 10% unchanged (default)
                - 'always': 100% [MASK] (simpler, less robust)
            vocab_size: Total vocabulary size (for random token sampling)
                       If None, inferred from token range during masking
        
        Raises:
            ValueError: If mask_ratio not in (0, 1) or invalid strategy
        
        Example:
            >>> masker = TokenLevelMasking(mask_token_id=518, mask_ratio=0.15)
            >>> tokens = torch.tensor([10, 20, 30, 40, 50])
            >>> masked, mask_labels = masker.apply_masking(tokens)
        """
        # Validate inputs
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(
                f"mask_ratio must be in (0, 1), got {mask_ratio}. "
                f"Standard BERT value is 0.15 (15% masking)."
            )
        
        if mask_strategy not in ['bert', 'always']:
            raise ValueError(
                f"mask_strategy must be 'bert' or 'always', got {mask_strategy}"
            )
        
        self.mask_token_id = mask_token_id
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        self.vocab_size = vocab_size
        
        print(f"[OK] TokenLevelMasking initialized")
        print(f"   Mask ratio: {mask_ratio} ({mask_ratio*100:.1f}%)")
        print(f"   Strategy: {mask_strategy}")
        print(f"   Mask token ID: {mask_token_id}")
        if vocab_size is not None:
            print(f"   Vocab size: {vocab_size}")
        print()
    
    def apply_masking(
        self,
        tokens: torch.Tensor,
        bar_positions: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random token-level masking.
        
        Algorithm:
        1. Compute num_mask = ceil(seq_len × mask_ratio)
        2. Random sample `num_mask` positions without replacement
        3. For each masked position, apply strategy:
           - BERT: 80% [MASK], 10% random, 10% unchanged
           - Always: 100% [MASK]
        4. Create boolean mask for loss computation
        
        Args:
            tokens: Input token IDs [seq_len]
            bar_positions: Ignored (kept for API compatibility with BarLevelMasking)
        
        Returns:
            Tuple of (masked_tokens, mask_labels):
                - masked_tokens: Tokens with masking applied [seq_len]
                - mask_labels: Boolean mask (True = masked position) [seq_len]
        
        Notes:
            - Masking is applied in-place on a clone (original unchanged)
            - mask_labels marks ALL masked positions (even unchanged ones)
            - This ensures loss is computed only on masked positions
        
        Edge cases handled:
            - Very short sequences: masks at least 1 token
            - num_mask > seq_len: clamps to seq_len
            - vocab_size=None: infers from max(tokens) + 1
        
        Example:
            Input:  [10, 20, 30, 40, 50]  (5 tokens, mask_ratio=0.4)
            Masks:  positions [1, 3]      (40% = 2 tokens)
            
            With 'bert' strategy:
            - 80% chance: [10, [MASK], 30, [MASK], 50]
            - 10% chance: [10, 234, 30, [MASK], 50]  (234 = random)
            - 10% chance: [10, 20, 30, [MASK], 50]   (20 unchanged)
            
            mask_labels: [False, True, False, True, False]
        """
        seq_len = len(tokens)
        device = tokens.device
        
        # Compute number of tokens to mask
        num_mask = max(1, int(seq_len * self.mask_ratio))
        
        # Edge case: can't mask more than sequence length
        if num_mask > seq_len:
            num_mask = seq_len
        
        # Random sample positions (without replacement)
        # Uses Python random for compatibility with training seed
        mask_positions = random.sample(range(seq_len), num_mask)
        
        # Initialize outputs
        masked_tokens = tokens.clone()
        mask_labels = torch.zeros(seq_len, dtype=torch.bool, device=device)
        
        # Infer vocab size if not provided
        vocab_size = self.vocab_size
        if vocab_size is None:
            # Use max token ID + 1 as approximation
            vocab_size = tokens.max().item() + 1
        
        # Apply masking to selected positions
        for pos in mask_positions:
            # Mark as masked (for loss computation)
            mask_labels[pos] = True
            
            # Apply masking strategy
            if self.mask_strategy == 'always':
                # Simple: always replace with [MASK]
                masked_tokens[pos] = self.mask_token_id
                
            elif self.mask_strategy == 'bert':
                # BERT-style: 80-10-10 distribution
                rand = random.random()
                
                if rand < 0.8:
                    # 80%: Replace with [MASK] token
                    masked_tokens[pos] = self.mask_token_id
                    
                elif rand < 0.9:
                    # 10%: Replace with random token from vocabulary
                    # Ensures model can't rely solely on [MASK] signal
                    random_token = random.randint(0, vocab_size - 1)
                    masked_tokens[pos] = random_token
                
                # else: 10%: Keep original token unchanged
                # Helps model learn bidirectional representations
                # Important for fine-tuning (no [MASK] in inference)
        
        return masked_tokens, mask_labels
    
    def get_masking_stats(
        self,
        mask_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Get statistics about masking operation.
        
        Args:
            mask_labels: Boolean mask from apply_masking() [seq_len]
        
        Returns:
            Dictionary with:
                - total_tokens: Total sequence length
                - masked_tokens: Number of masked positions
                - mask_ratio: Actual masking ratio (should ≈ self.mask_ratio)
        
        Example:
            >>> stats = masker.get_masking_stats(mask_labels)
            >>> print(f"Masked {stats['mask_ratio']:.1%} of tokens")
        """
        total_tokens = len(mask_labels)
        masked_tokens = mask_labels.sum().item()
        mask_ratio = masked_tokens / total_tokens if total_tokens > 0 else 0.0
        
        return {
            'total_tokens': total_tokens,
            'masked_tokens': masked_tokens,
            'mask_ratio': mask_ratio,
        }
