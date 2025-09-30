"""Text encoder for conditioning the diffusion model."""

import torch
from torch import nn


class TextEncoder(nn.Module):
    """Simple text encoder using embeddings and transformer layers."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_length: int = 256,
    ):
        """Initialize the text encoder.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of embeddings
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, text_tokens: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode text tokens into embeddings.
        
        Args:
            text_tokens: Input token IDs of shape (batch_size, seq_length)
            attention_mask: Optional attention mask of shape (batch_size, seq_length)
        
        Returns:
            Text embeddings of shape (batch_size, embed_dim)
        """
        batch_size, seq_length = text_tokens.shape
        
        # Create position indices
        positions = torch.arange(seq_length, device=text_tokens.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embedding(text_tokens)
        pos_embeds = self.position_embedding(positions)
        embeddings = token_embeds + pos_embeds
        
        # Apply transformer
        if attention_mask is not None:
            # Convert attention mask to the format expected by transformer (True = masked)
            attention_mask = ~attention_mask.bool()
        
        encoded = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        # Pool the output (use the first token as representation, like BERT's [CLS])
        pooled = encoded[:, 0, :]
        pooled = self.layer_norm(pooled)
        
        return pooled
