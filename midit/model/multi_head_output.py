"""Multi-head output layer for REMI+ token type prediction.

This module implements 9 separate classification heads, one for each REMI+ token type:
- Pitch, Velocity, Duration, Program, Position, Bar, Tempo, TimeSig, Rest

Each head predicts its token type independently from shared hidden representations.
"""

import torch
from torch import nn
from typing import Dict, Optional


class MultiHeadOutput(nn.Module):
    """Multi-head output layer for REMI+ tokenization.
    
    Implements separate classification heads for each token type, enabling
    the model to learn token-specific representations while sharing the
    underlying transformer encoder.
    
    Architecture:
        hidden_states (batch, seq, d_model) 
        → 9 independent Linear layers
        → {token_type: logits (batch, seq, vocab_size)}
    """
    
    def __init__(
        self,
        d_model: int = 512,
        vocab_sizes: Optional[Dict[str, int]] = None,
        dropout: float = 0.1,
    ):
        """Initialize multi-head output layer.
        
        Args:
            d_model: Hidden dimension from transformer encoder
            vocab_sizes: Dictionary mapping token types to vocabulary sizes
                Required keys: pitch, velocity, duration, program, position,
                              bar, tempo, timesig, rest
            dropout: Dropout rate applied before each head
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Default vocab sizes (from analysis)
        if vocab_sizes is None:
            raise ValueError(
                "vocab_sizes MUST be provided from LakhDataset.actual_vocab_sizes. "
                "Never use default values as they don't match actual tokenizer output."
            )
        
        self.vocab_sizes = vocab_sizes
        
        # Dropout layer (shared across heads)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization before projection
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Classification heads (one per token type)
        self.head_pitch = nn.Linear(d_model, vocab_sizes['pitch'])
        self.head_velocity = nn.Linear(d_model, vocab_sizes['velocity'])
        self.head_duration = nn.Linear(d_model, vocab_sizes['duration'])
        self.head_program = nn.Linear(d_model, vocab_sizes['program'])
        self.head_position = nn.Linear(d_model, vocab_sizes['position'])
        self.head_bar = nn.Linear(d_model, vocab_sizes['bar'])
        self.head_tempo = nn.Linear(d_model, vocab_sizes['tempo'])
        self.head_timesig = nn.Linear(d_model, vocab_sizes['timesig'])
        self.head_rest = nn.Linear(d_model, vocab_sizes['rest'])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for head in [
            self.head_pitch, self.head_velocity, self.head_duration,
            self.head_program, self.head_position, self.head_bar,
            self.head_tempo, self.head_timesig, self.head_rest
        ]:
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all classification heads.
        
        Args:
            hidden_states: Hidden representations from transformer encoder
                          Shape: (batch_size, seq_length, d_model)
        
        Returns:
            Dictionary mapping token types to logits:
                {
                    'pitch': (batch, seq, vocab_size_pitch),
                    'velocity': (batch, seq, vocab_size_velocity),
                    ...
                }
        """
        # Normalize and apply dropout
        hidden = self.layer_norm(hidden_states)
        hidden = self.dropout(hidden)
        
        # Compute logits for each token type
        logits = {
            'pitch': self.head_pitch(hidden),
            'velocity': self.head_velocity(hidden),
            'duration': self.head_duration(hidden),
            'program': self.head_program(hidden),
            'position': self.head_position(hidden),
            'bar': self.head_bar(hidden),
            'tempo': self.head_tempo(hidden),
            'timesig': self.head_timesig(hidden),
            'rest': self.head_rest(hidden),
        }
        
        return logits
    
    def get_predictions(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Get token predictions from hidden states.
        
        Args:
            hidden_states: Hidden representations (batch, seq, d_model)
            temperature: Softmax temperature for sampling (default: 1.0)
        
        Returns:
            Dictionary of predicted token IDs for each type
        """
        logits = self.forward(hidden_states)
        
        predictions = {}
        for token_type, logits_tensor in logits.items():
            # Apply temperature and softmax
            probs = torch.softmax(logits_tensor / temperature, dim=-1)
            # Greedy decoding (argmax)
            predictions[token_type] = torch.argmax(probs, dim=-1)
        
        return predictions
    
    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        lambda_weights: Dict[str, float],
        mask: Optional[torch.Tensor] = None,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,  # ← NEW!
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted multi-head loss with optional class weights.
        
        Args:
            hidden_states: Hidden representations (batch, seq, d_model)
            targets: Dictionary of target token IDs for each type
                    {token_type: (batch, seq)}
                    Non-supervised positions should have value -100
            lambda_weights: Dictionary of loss weights for each type
            mask: Optional attention mask (batch, seq) - 1 for real tokens, 0 for padding
            class_weights: Optional dictionary of class weight tensors per token type
                          {token_type: tensor of shape (vocab_size,)}
                          Used to handle class imbalance (e.g., velocity, program)
        
        Returns:
            Dictionary containing:
                - 'total': Weighted sum of all losses (tensor for backprop)
                - '{token_type}': Unweighted loss for this type (tensor)
                - '{token_type}_weighted': Lambda-weighted loss (tensor)
                - '{token_type}_has_class_weights': 1.0 if class weights applied (tensor)
        """
        logits = self.forward(hidden_states)
        
        losses = {}
        total_loss = 0.0
        
        for token_type in self.vocab_sizes.keys():
            if token_type not in targets:
                continue
            
            # Get class weights for this token type (if available)
            weight = None
            if class_weights is not None and token_type in class_weights:
                weight = class_weights[token_type]
            
            # Compute cross-entropy loss
            # reduction='none' to apply mask manually afterwards
            loss = nn.functional.cross_entropy(
                logits[token_type].reshape(-1, self.vocab_sizes[token_type]),
                targets[token_type].reshape(-1),
                weight=weight,  # ← CLASS WEIGHTS APPLIED HERE!
                reduction='none',
                ignore_index=-100  # ← CRITICAL: Ignores non-supervised positions
            )
            
            # Reshape back to (batch, seq)
            loss = loss.view(targets[token_type].shape)
            
            # Apply mask if provided
            if mask is not None:
                # Normalize by masked tokens of THIS type only (not all masked tokens)
                # Avoids diluting rare types (Tempo, TimeSig) whose freq << avg
                valid_mask = mask & (targets[token_type] != -100)
                loss_mean = (loss * mask).sum() / valid_mask.sum().clamp(min=1)
            else:
                loss_mean = loss.mean()
            
            # Apply lambda weight
            weighted_loss = lambda_weights.get(token_type, 1.0) * loss_mean
            
            # Store losses (unweighted, weighted, and flag)
            losses[token_type] = loss_mean  # Tensor (for Lightning logging)
            losses[f'{token_type}_weighted'] = weighted_loss  # Tensor
            
            # Flag if class weights were applied
            if weight is not None:
                losses[f'{token_type}_has_class_weights'] = torch.tensor(
                    1.0, 
                    device=loss_mean.device,
                    dtype=loss_mean.dtype
                )
            
            # Accumulate for total loss
            total_loss = total_loss + weighted_loss
        
        losses['total'] = total_loss
        
        return losses


# ============================================================================
# Testing utilities
# ============================================================================

def test_multi_head_output():
    """Test the multi-head output module."""
    print("\n" + "="*70)
    print("🧪 TESTING MultiHeadOutput")
    print("="*70 + "\n")
    
    # Config
    batch_size = 4
    seq_length = 128
    d_model = 512
    
    # Vocab sizes (must match dataset!)
    vocab_sizes = {
        'pitch': 151,      # 89 + 62 pitchdrum
        'velocity': 32,
        'duration': 32,
        'program': 129,
        'position': 96,
        'bar': 1,
        'tempo': 32,
        'timesig': 9,
        'rest': 32,
    }
    
    print(f"📊 Test configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Hidden dimension: {d_model}\n")
    
    # Create module
    print("[TEST 1] Creating MultiHeadOutput module...")
    multi_head = MultiHeadOutput(d_model=d_model, vocab_sizes=vocab_sizes)
    
    total_params = sum(p.numel() for p in multi_head.parameters())
    print(f"✅ Module created")
    print(f"   Total parameters: {total_params:,}\n")
    
    # Print vocab sizes
    print(f"📚 Vocabulary sizes:")
    for token_type, vocab_size in multi_head.vocab_sizes.items():
        print(f"   {token_type:10s}: {vocab_size:>4d}")
    print()
    
    # Test forward pass
    print("[TEST 2] Testing forward pass...")
    hidden_states = torch.randn(batch_size, seq_length, d_model)
    
    logits = multi_head(hidden_states)
    
    print(f"✅ Forward pass successful")
    print(f"   Output logits shapes:")
    for token_type, logits_tensor in logits.items():
        print(f"   {token_type:10s}: {tuple(logits_tensor.shape)}")
    print()
    
    # Test predictions
    print("[TEST 3] Testing get_predictions...")
    predictions = multi_head.get_predictions(hidden_states)
    
    print(f"✅ Predictions generated")
    print(f"   Prediction shapes:")
    for token_type, pred_tensor in predictions.items():
        print(f"   {token_type:10s}: {tuple(pred_tensor.shape)}")
    print()
    
    # Test loss computation WITHOUT class weights
    print("[TEST 4] Testing compute_loss (no class weights)...")
    
    # Create dummy targets
    targets = {
        'pitch': torch.randint(0, vocab_sizes['pitch'], (batch_size, seq_length)),
        'velocity': torch.randint(0, vocab_sizes['velocity'], (batch_size, seq_length)),
        'duration': torch.randint(0, vocab_sizes['duration'], (batch_size, seq_length)),
        'program': torch.randint(0, vocab_sizes['program'], (batch_size, seq_length)),
        'position': torch.randint(0, vocab_sizes['position'], (batch_size, seq_length)),
        'bar': torch.randint(0, vocab_sizes['bar'], (batch_size, seq_length)),
        'tempo': torch.randint(0, vocab_sizes['tempo'], (batch_size, seq_length)),
        'timesig': torch.randint(0, vocab_sizes['timesig'], (batch_size, seq_length)),
        'rest': torch.randint(0, vocab_sizes['rest'], (batch_size, seq_length)),
    }
    
    # Lambda weights from config
    lambda_weights = {
        'pitch': 1.0,
        'velocity': 1.0,
        'duration': 1.0,
        'program': 1.0,
        'position': 4.5,
        'rest': 10.0,
        'bar': 10.0,
        'timesig': 10.0,
        'tempo': 20.0,
    }
    
    # Create mask (all ones = no masking)
    mask = torch.ones(batch_size, seq_length)
    
    losses = multi_head.compute_loss(hidden_states, targets, lambda_weights, mask)
    
    print(f"✅ Loss computation successful")
    print(f"   Individual losses:")
    for token_type in vocab_sizes.keys():
        if token_type in losses:
            unweighted = losses[token_type].item()
            weighted_key = f'{token_type}_weighted'
            weighted = losses[weighted_key].item() if weighted_key in losses else unweighted
            print(f"   {token_type:10s}: {unweighted:>8.4f} (weighted: {weighted:>8.4f})")
    print(f"\n   Total weighted loss: {losses['total'].item():.4f}\n")
    
    # Test loss computation WITH class weights
    print("[TEST 5] Testing compute_loss (WITH class weights)...")
    
    # Create dummy class weights (sqrt scaled for velocity/program)
    class_weights_dict = {
        'velocity': torch.ones(vocab_sizes['velocity']) * 2.0,  # Dummy: all 2×
        'program': torch.ones(vocab_sizes['program']) * 3.0,    # Dummy: all 3×
    }
    
    losses_with_weights = multi_head.compute_loss(
        hidden_states, 
        targets, 
        lambda_weights, 
        mask,
        class_weights=class_weights_dict
    )
    
    print(f"✅ Loss computation with class weights successful")
    print(f"   Checking flags:")
    for token_type in vocab_sizes.keys():
        flag_key = f'{token_type}_has_class_weights'
        if flag_key in losses_with_weights:
            print(f"   {token_type:10s}: ✅ Class weights applied")
    print(f"\n   Total weighted loss: {losses_with_weights['total'].item():.4f}\n")
    
    print("="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70 + "\n")
    
    return multi_head


if __name__ == "__main__":
    # Run tests
    test_multi_head_output()