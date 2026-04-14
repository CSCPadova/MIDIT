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


# ============================================================================
# Bar-Fill Masking (Fine-tuning / Infilling)
# ============================================================================

class BarFillMasking:
    """Bar-infilling masking for fine-tuning readiness.

    Selects a contiguous run of 1–max_infill_bars bars and masks only the
    note-content tokens within them (Pitch, PitchDrum, Velocity, Duration,
    Program).  Structural tokens (Bar, Position, Tempo, TimeSig, Rest) are
    preserved — identical to infer.py apply_bar_mask() / MusicBERT bar infilling.

    Falls back to TokenLevelMasking when bar_positions is None or the sequence
    contains no note tokens.

    Interface matches BarLevelMasking and TokenLevelMasking:
        apply_masking(tokens, bar_positions) → (masked_tokens, mask_labels)
    """

    _NOTE_PREFIXES = ('Pitch_', 'PitchDrum_', 'Velocity_', 'Duration_', 'Program_')

    def __init__(
        self,
        tokenizer_vocab: Dict[str, int],
        mask_token_id: int = 3,
        max_infill_bars: int = 3,
        mask_strategy: str = 'always',
        vocab_size: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            tokenizer_vocab: Full tokenizer vocabulary (token_name → token_id).
                             Used to build the reverse lookup for note-token filtering.
            mask_token_id:   ID of [MASK] token (default 3 for V2 special tokens).
            max_infill_bars: Max contiguous bars to mask per sample (default 3).
            mask_strategy:   'always' (100% MASK, default) or 'bert' (80/10/10).
                             'always' is preferred for infilling fine-tuning: it matches
                             inference behavior exactly and avoids musically nonsensical
                             random token substitutions.
            vocab_size:      Total vocab size for random token sampling in bert mode.
                             If None, inferred as max(token_id)+1 per sample.
            seed:            RNG seed for reproducibility. None = non-deterministic.
        """
        if max_infill_bars < 1:
            raise ValueError(f"max_infill_bars must be >= 1, got {max_infill_bars}")
        if mask_strategy not in ('bert', 'always'):
            raise ValueError(f"mask_strategy must be 'bert' or 'always', got {mask_strategy}")

        self.mask_token_id   = mask_token_id
        self.max_infill_bars = max_infill_bars
        self.mask_strategy   = mask_strategy
        self.vocab_size      = vocab_size
        self._rng            = random.Random(seed)

        # Build reverse vocab for note-token classification
        self._id_to_token: Dict[int, str] = {v: k for k, v in tokenizer_vocab.items()}

        # Fallback: token-level masking when bar_positions unavailable
        self._fallback = TokenLevelMasking(
            mask_token_id=mask_token_id,
            mask_ratio=0.15,
            mask_strategy=mask_strategy if mask_strategy == 'bert' else 'always',
            vocab_size=vocab_size if vocab_size is not None else len(tokenizer_vocab),
        )

        print(f"[OK] BarFillMasking initialized")
        print(f"   Max infill bars: {max_infill_bars}")
        print(f"   Mask strategy:   {mask_strategy}")
        print(f"   Mask token ID:   {mask_token_id}")
        print(f"   Vocab size:      {vocab_size if vocab_size is not None else len(tokenizer_vocab)}")
        print(f"   Seed:            {seed}\n")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_note_token(self, token_name: str) -> bool:
        return token_name.startswith(self._NOTE_PREFIXES)

    def _bar_has_notes(
        self,
        tokens: torch.Tensor,
        start: int,
        end: int,
    ) -> bool:
        """True if [start, end) contains at least one note token."""
        for pos in range(start, end):
            name = self._id_to_token.get(tokens[pos].item(), '')
            if self._is_note_token(name):
                return True
        return False

    def _select_bars(
        self,
        tokens: torch.Tensor,
        bar_positions: List[int],
    ) -> Optional[Tuple[int, int]]:
        """Select a contiguous bar window that contains at least one note token.

        Tries up to 10 random samples, then falls back to a linear scan for
        the first valid window.  Returns (start_bar, end_bar) as exclusive-end
        bar indices, or None if no note tokens exist in the sequence.
        """
        num_bars = len(bar_positions) - 1

        for _ in range(10):
            num_infill = self._rng.randint(1, min(self.max_infill_bars, num_bars))
            start_bar  = self._rng.randint(0, num_bars - num_infill)
            end_bar    = start_bar + num_infill

            tok_start = bar_positions[start_bar]
            tok_end   = bar_positions[end_bar]

            if tok_end > len(tokens) or tok_start >= tok_end:
                continue
            if self._bar_has_notes(tokens, tok_start, tok_end):
                return (start_bar, end_bar)

        # Linear scan fallback: find first window of any valid size
        for start_bar in range(num_bars):
            for n in range(1, min(self.max_infill_bars, num_bars - start_bar) + 1):
                end_bar   = start_bar + n
                tok_start = bar_positions[start_bar]
                tok_end   = bar_positions[end_bar]
                if tok_start < tok_end <= len(tokens):
                    if self._bar_has_notes(tokens, tok_start, tok_end):
                        return (start_bar, end_bar)

        return None  # sequence has no note tokens

    # ── Public interface ──────────────────────────────────────────────────────

    def apply_masking(
        self,
        tokens: torch.Tensor,
        bar_positions: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply bar-infilling masking.

        Args:
            tokens:        Original token IDs [seq_len].
            bar_positions: Bar boundary positions (len = num_bars + 1).
                           If None or fewer than 2 entries, falls back to
                           TokenLevelMasking at 15% rate.

        Returns:
            masked_tokens: Clone of tokens with note content replaced in the
                           selected bar window [seq_len].
            mask_labels:   Boolean tensor, True at positions that were masked
                           and should contribute to loss [seq_len].
        """
        if bar_positions is None or len(bar_positions) < 2:
            return self._fallback.apply_masking(tokens, bar_positions)

        selected = self._select_bars(tokens, bar_positions)

        if selected is None:
            # No note tokens found — return sequence unmasked
            return tokens.clone(), torch.zeros(len(tokens), dtype=torch.bool)

        start_bar, end_bar = selected
        tok_start = bar_positions[start_bar]
        tok_end   = bar_positions[end_bar]

        masked_tokens = tokens.clone()
        mask_labels   = torch.zeros(len(tokens), dtype=torch.bool)
        vocab_size    = self.vocab_size if self.vocab_size is not None else (tokens.max().item() + 1)

        for pos in range(tok_start, tok_end):
            token_id = tokens[pos].item()
            name = self._id_to_token.get(token_id, '')

            # Preserve structural tokens unconditionally
            if name and not self._is_note_token(name):
                continue

            mask_labels[pos] = True

            if self.mask_strategy == 'always':
                masked_tokens[pos] = self.mask_token_id
            else:  # bert 80/10/10
                r = self._rng.random()
                if r < 0.8:
                    masked_tokens[pos] = self.mask_token_id
                elif r < 0.9:
                    masked_tokens[pos] = self._rng.randint(0, vocab_size - 1)
                # else 10%: keep original unchanged

        return masked_tokens, mask_labels

    def get_masking_stats(self, mask_labels: torch.Tensor) -> Dict[str, float]:
        """Return masking statistics (same interface as other maskers)."""
        total  = len(mask_labels)
        masked = mask_labels.sum().item()
        return {
            'total_tokens':  total,
            'masked_tokens': masked,
            'mask_ratio':    masked / total if total > 0 else 0.0,
        }
