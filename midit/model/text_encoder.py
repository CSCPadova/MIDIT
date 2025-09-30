"""Text encoder for conditioning the diffusion model."""

import torch
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer


class FlanT5TextEncoder(nn.Module):
    """FLAN-T5 text encoder wrapper for conditioning the diffusion model.
    
    This encoder uses a pretrained FLAN-T5 model from HuggingFace and keeps it frozen
    during training to leverage its pretrained semantic knowledge.
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        embed_dim: int = 512,
        freeze: bool = True,
    ):
        """Initialize the FLAN-T5 text encoder.
        
        Args:
            model_name: Name of the FLAN-T5 model from HuggingFace (e.g., 'google/flan-t5-small', 'google/flan-t5-base')
            embed_dim: Target embedding dimension for the output
            freeze: Whether to freeze the FLAN-T5 encoder weights
        """
        super().__init__()
        
        # Load pretrained FLAN-T5 encoder
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Freeze encoder if specified
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get the hidden size from the encoder config
        self.hidden_size = self.encoder.config.d_model
        self.embed_dim = embed_dim
        
        # Projection layer to match target embedding dimension
        if self.hidden_size != embed_dim:
            self.projection = nn.Linear(self.hidden_size, embed_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, text_tokens: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode text tokens into embeddings.
        
        Args:
            text_tokens: Input token IDs of shape (batch_size, seq_length)
            attention_mask: Optional attention mask of shape (batch_size, seq_length)
        
        Returns:
            Text embeddings of shape (batch_size, embed_dim)
        """
        # Forward pass through FLAN-T5 encoder
        encoder_outputs = self.encoder(
            input_ids=text_tokens,
            attention_mask=attention_mask,
        )
        
        # Get the last hidden states
        hidden_states = encoder_outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        
        # Pool the output (use mean pooling across sequence length)
        if attention_mask is not None:
            # Expand attention mask for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Project to target dimension
        pooled = self.projection(pooled)
        
        return pooled


# Keep the old TextEncoder for backward compatibility (but mark as deprecated)
class TextEncoder(nn.Module):
    """Simple text encoder using embeddings and transformer layers.
    
    DEPRECATED: This encoder is kept for backward compatibility. 
    Use FlanT5TextEncoder instead for better performance.
    """
    
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
