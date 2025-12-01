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
            vocab_sizes = {
                'pitch': 89,      # MIDI 21-109
                'velocity': 32,   # 32 bins
                'duration': 128,  # REMI duration tokens
                'program': 129,   # 128 GM + Program -1 (drums)
                'position': 32,   # 8th note resolution × 4 beats
                'bar': 2,         # Bar_None, Bar_End
                'tempo': 20,      # Tempo change tokens
                'timesig': 10,    # Common time signatures
                'rest': 50,       # Rest duration tokens
            }
        
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
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted multi-head loss.
        
        Args:
            hidden_states: Hidden representations (batch, seq, d_model)
            targets: Dictionary of target token IDs for each type
                    {token_type: (batch, seq)}
            lambda_weights: Dictionary of loss weights for each type
            mask: Optional attention mask (batch, seq) - 1 for real tokens, 0 for padding
        
        Returns:
            Dictionary containing:
                - 'total': Weighted sum of all losses
                - 'pitch': Pitch classification loss
                - 'velocity': Velocity classification loss
                - ... (one per token type)
        """
        logits = self.forward(hidden_states)
        
        losses = {}
        total_loss = 0.0
        
        for token_type in self.vocab_sizes.keys():
            if token_type not in targets:
                continue
            
            # Compute cross-entropy loss
            loss = nn.functional.cross_entropy(
                logits[token_type].reshape(-1, self.vocab_sizes[token_type]),
                targets[token_type].reshape(-1),
                reduction='none'
            )
            
            # Reshape back to (batch, seq)
            loss = loss.view(targets[token_type].shape)
            
            # Apply mask if provided
            if mask is not None:
                loss = loss * mask
                loss = loss.sum() / mask.sum()
            else:
                loss = loss.mean()
            
            # Apply lambda weight
            weighted_loss = lambda_weights.get(token_type, 1.0) * loss
            
            losses[token_type] = loss
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
    
    print(f"📊 Test configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Hidden dimension: {d_model}\n")
    
    # Create module
    print("[TEST 1] Creating MultiHeadOutput module...")
    multi_head = MultiHeadOutput(d_model=d_model)
    
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
    
    # Test loss computation
    print("[TEST 4] Testing compute_loss...")
    
    # Create dummy targets
    targets = {
        'pitch': torch.randint(0, 89, (batch_size, seq_length)),
        'velocity': torch.randint(0, 32, (batch_size, seq_length)),
        'duration': torch.randint(0, 128, (batch_size, seq_length)),
        'program': torch.randint(0, 129, (batch_size, seq_length)),
        'position': torch.randint(0, 32, (batch_size, seq_length)),
        'bar': torch.randint(0, 2, (batch_size, seq_length)),
        'tempo': torch.randint(0, 20, (batch_size, seq_length)),
        'timesig': torch.randint(0, 10, (batch_size, seq_length)),
        'rest': torch.randint(0, 50, (batch_size, seq_length)),
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
    for token_type, loss_value in losses.items():
        if token_type != 'total':
            weighted = loss_value.item() * lambda_weights.get(token_type, 1.0)
            print(f"   {token_type:10s}: {loss_value.item():>8.4f} (weighted: {weighted:>8.4f})")
    print(f"\n   Total weighted loss: {losses['total'].item():.4f}\n")
    
    print("="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70 + "\n")
    
    return multi_head


if __name__ == "__main__":
    # Run tests
    test_multi_head_output()