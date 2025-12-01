"""Multi-head Masked Language Modeling loss for Phase 1 pre-training.

This module implements the loss function for training a masked transformer
on MIDI sequences with 9 token type heads (pitch, velocity, duration, etc.).
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict


class MultiHeadMLMLoss(nn.Module):
    """Multi-head loss for Masked Language Modeling.
    
    Computes separate cross-entropy losses for each token type head,
    then combines them with configurable weights.
    
    This is used for Phase 1 pre-training on MIDI data without text conditioning.
    """
    
    def __init__(
        self,
        head_weights: Dict[str, float] = None,
        ignore_index: int = -100,
    ):
        """Initialize the multi-head MLM loss.
        
        Args:
            head_weights: Dictionary of weights for each head type.
                         If None, uses equal weights for all heads.
            ignore_index: Index to ignore in loss calculation (for padding/non-masked)
        """
        super().__init__()
        
        self.ignore_index = ignore_index
        
        # Default equal weights if not provided
        if head_weights is None:
            self.head_weights = {
                'pitch': 1.0,
                'velocity': 1.0,
                'duration': 1.0,
                'program': 1.0,
                'position': 1.0,
                'bar': 1.0,
                'tempo': 1.0,
                'timesig': 1.0,
                'rest': 1.0,
            }
        else:
            self.head_weights = head_weights
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        return_individual: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """Calculate multi-head MLM loss.
        
        Args:
            predictions: Dictionary of predicted logits for each head.
                        Each tensor has shape (batch, seq_len, num_classes)
            targets: Dictionary of target values for each head.
                    Each tensor has shape (batch, seq_len)
            return_individual: If True, returns dict of individual losses per head
        
        Returns:
            If return_individual=False: Scalar total loss
            If return_individual=True: Dict of losses per head + 'total'
        """
        losses = {}
        total_loss = 0.0
        
        # Compute loss for each head
        for head_name in predictions.keys():
            if head_name not in targets:
                continue
            
            pred = predictions[head_name]  # (batch, seq_len, num_classes)
            target = targets[head_name]    # (batch, seq_len)
            
            # Reshape for cross-entropy
            batch_size, seq_len, num_classes = pred.shape
            pred_flat = pred.view(-1, num_classes)
            target_flat = target.view(-1)
            
            # Calculate cross-entropy loss
            loss = F.cross_entropy(
                pred_flat,
                target_flat,
                ignore_index=self.ignore_index,
                reduction='mean',
            )
            
            # Apply head weight
            weight = self.head_weights.get(head_name, 1.0)
            weighted_loss = weight * loss
            
            losses[head_name] = loss.item()  # Store unweighted for logging
            total_loss += weighted_loss
        
        # Add total loss
        losses['total'] = total_loss.item()
        
        if return_individual:
            return losses
        else:
            return total_loss


class BarLevelMLMLoss(MultiHeadMLMLoss):
    """Extension of MultiHeadMLMLoss with bar-level masking awareness.
    
    This is a wrapper that works identically to MultiHeadMLMLoss but is
    semantically named to indicate it's designed for bar-level masked training.
    """
    
    def __init__(self, head_weights: Dict[str, float] = None, ignore_index: int = -100):
        """Initialize bar-level MLM loss.
        
        Args:
            head_weights: Dictionary of weights for each head
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__(head_weights=head_weights, ignore_index=ignore_index)