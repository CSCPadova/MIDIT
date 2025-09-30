"""Diffusion loss functions for training the model."""

import torch
from torch import nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """Loss function for diffusion models.
    
    This implements various loss functions for training diffusion models,
    including simple MSE, cross-entropy, and more sophisticated variants.
    """
    
    def __init__(self, loss_type: str = "cross_entropy", ignore_index: int = -100):
        """Initialize the diffusion loss.
        
        Args:
            loss_type: Type of loss ('mse', 'cross_entropy', 'hybrid')
            ignore_index: Index to ignore in loss calculation (for padding)
        """
        super().__init__()
        self.loss_type = loss_type
        self.ignore_index = ignore_index
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        noise: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculate the diffusion loss.
        
        Args:
            predictions: Predicted values (logits or noise) of shape (batch, seq_len, vocab_size) or (batch, seq_len)
            targets: Target values of shape (batch, seq_len)
            noise: Optional noise tensor for MSE loss
            mask: Optional mask for valid positions
        
        Returns:
            Scalar loss value
        """
        if self.loss_type == "cross_entropy":
            return self._cross_entropy_loss(predictions, targets, mask)
        elif self.loss_type == "mse":
            return self._mse_loss(predictions, noise, mask)
        elif self.loss_type == "hybrid":
            ce_loss = self._cross_entropy_loss(predictions, targets, mask)
            if noise is not None:
                mse_loss = self._mse_loss(predictions, noise, mask)
                return ce_loss + 0.1 * mse_loss
            return ce_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _cross_entropy_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculate cross-entropy loss.
        
        Args:
            predictions: Predicted logits of shape (batch, seq_len, vocab_size)
            targets: Target token IDs of shape (batch, seq_len)
            mask: Optional mask of shape (batch, seq_len)
        
        Returns:
            Scalar loss value
        """
        batch_size, seq_len, vocab_size = predictions.shape
        
        # Reshape for cross-entropy
        predictions_flat = predictions.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Calculate loss
        loss = F.cross_entropy(
            predictions_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            loss = loss * mask_flat
            loss = loss.sum() / mask_flat.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return loss
    
    def _mse_loss(
        self,
        predictions: torch.Tensor,
        noise: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculate MSE loss for noise prediction.
        
        Args:
            predictions: Predicted noise of shape (batch, seq_len) or (batch, seq_len, dim)
            noise: Target noise of shape (batch, seq_len) or (batch, seq_len, dim)
            mask: Optional mask of shape (batch, seq_len)
        
        Returns:
            Scalar loss value
        """
        loss = F.mse_loss(predictions, noise, reduction="none")
        
        # Apply mask if provided
        if mask is not None:
            if loss.dim() == 3:
                mask = mask.unsqueeze(-1)
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return loss


class MaskedDiffusionLoss(DiffusionLoss):
    """Extension of DiffusionLoss with masking support for variable-length sequences."""
    
    def __init__(self, loss_type: str = "cross_entropy", ignore_index: int = -100, mask_value: float = 0.0):
        """Initialize the masked diffusion loss.
        
        Args:
            loss_type: Type of loss ('mse', 'cross_entropy', 'hybrid')
            ignore_index: Index to ignore in loss calculation
            mask_value: Value to use for masked positions
        """
        super().__init__(loss_type=loss_type, ignore_index=ignore_index)
        self.mask_value = mask_value
    
    def apply_masking(
        self,
        sequence: torch.Tensor,
        mask_ratio: float = 0.15,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking to a sequence.
        
        Args:
            sequence: Input sequence of shape (batch, seq_len)
            mask_ratio: Ratio of tokens to mask
        
        Returns:
            Tuple of (masked_sequence, mask) where mask indicates masked positions
        """
        batch_size, seq_len = sequence.shape
        
        # Create random mask
        mask = torch.rand(batch_size, seq_len, device=sequence.device) < mask_ratio
        
        # Apply mask
        masked_sequence = sequence.clone()
        masked_sequence[mask] = self.mask_value
        
        return masked_sequence, mask
