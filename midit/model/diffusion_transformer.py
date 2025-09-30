"""Masked Diffusion Transformer for text-to-MIDI generation."""

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import math

from .text_encoder import FlanT5TextEncoder


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal embeddings for timesteps.
        
        Args:
            time: Timesteps of shape (batch_size,)
        
        Returns:
            Embeddings of shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MaskedDiffusionTransformer(L.LightningModule):
    """Masked Diffusion Transformer for text-to-MIDI generation.
    
    This model implements a diffusion process for generating MIDI sequences
    conditioned on text descriptions. It uses masking to handle variable-length
    sequences and transformer architecture for modeling.
    """
    
    def __init__(
        self,
        midi_vocab_size: int = 512,
        text_vocab_size: int = 10000,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 1024,
        timesteps: int = 1000,
        lr: float = 1e-4,
        beta_schedule: str = "linear",
        text_encoder_model: str = "google/flan-t5-base",
        text_encoder_freeze: bool = True,
    ):
        """Initialize the Masked Diffusion Transformer.
        
        Args:
            midi_vocab_size: Size of MIDI token vocabulary
            text_vocab_size: Size of text token vocabulary (kept for backward compatibility but not used with FLAN-T5)
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            timesteps: Number of diffusion timesteps
            lr: Learning rate
            beta_schedule: Type of noise schedule ('linear', 'cosine')
            text_encoder_model: Name of the FLAN-T5 model to use (e.g., 'google/flan-t5-small', 'google/flan-t5-base')
            text_encoder_freeze: Whether to freeze the FLAN-T5 encoder weights
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.midi_vocab_size = midi_vocab_size
        self.d_model = d_model
        self.timesteps = timesteps
        self.lr = lr
        
        # Text encoder for conditioning - using pretrained FLAN-T5
        self.text_encoder = FlanT5TextEncoder(
            model_name=text_encoder_model,
            embed_dim=d_model,
            freeze=text_encoder_freeze,
        )
        
        # MIDI token embeddings
        self.midi_embedding = nn.Embedding(midi_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Timestep embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Transformer architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, midi_vocab_size)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize noise schedule
        self.register_buffer("betas", self._get_beta_schedule(beta_schedule, timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
    
    def _get_beta_schedule(self, schedule: str, timesteps: int) -> torch.Tensor:
        """Get noise schedule for diffusion process.
        
        Args:
            schedule: Type of schedule ('linear', 'cosine')
            timesteps: Number of timesteps
        
        Returns:
            Beta values for each timestep
        """
        if schedule == "linear":
            beta_start = 0.0001
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def forward(
        self,
        midi_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        timestep: torch.Tensor,
        text_attention_mask: torch.Tensor = None,
        midi_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            midi_tokens: MIDI token IDs of shape (batch_size, seq_length)
            text_tokens: Text token IDs of shape (batch_size, text_length)
            timestep: Diffusion timestep of shape (batch_size,)
            text_attention_mask: Attention mask for text (batch_size, text_length)
            midi_attention_mask: Attention mask for MIDI (batch_size, seq_length)
        
        Returns:
            Predicted noise or denoised tokens of shape (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = midi_tokens.shape
        
        # Encode text conditioning
        text_embedding = self.text_encoder(text_tokens, text_attention_mask)  # (batch_size, d_model)
        
        # Get MIDI embeddings
        positions = torch.arange(seq_length, device=midi_tokens.device).unsqueeze(0).expand(batch_size, -1)
        midi_embeds = self.midi_embedding(midi_tokens)
        pos_embeds = self.position_embedding(positions)
        
        # Add timestep embeddings
        time_embeds = self.time_mlp(timestep)  # (batch_size, d_model)
        
        # Combine embeddings
        midi_embeds = midi_embeds + pos_embeds + time_embeds.unsqueeze(1)
        midi_embeds = self.layer_norm(midi_embeds)
        
        # Expand text embedding to match MIDI sequence
        text_memory = text_embedding.unsqueeze(1).expand(-1, seq_length, -1)
        
        # Apply transformer encoder to text
        text_encoded = self.transformer_encoder(text_memory)
        
        # Apply transformer decoder with text conditioning
        if midi_attention_mask is not None:
            midi_attention_mask = ~midi_attention_mask.bool()
        
        decoded = self.transformer_decoder(
            tgt=midi_embeds,
            memory=text_encoded,
            tgt_key_padding_mask=midi_attention_mask,
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoded)
        
        return logits
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Add noise to the input according to the diffusion schedule.
        
        Args:
            x: Clean input of shape (batch_size, seq_length)
            t: Timestep of shape (batch_size,)
            noise: Optional noise tensor, if None will be sampled
        
        Returns:
            Noised input
        """
        if noise is None:
            noise = torch.randn_like(x.float())
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alpha_prod * x.float() + sqrt_one_minus_alpha_prod * noise
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step for the model.
        
        Args:
            batch: Dictionary containing 'midi_tokens', 'text_tokens', and optional masks
            batch_idx: Batch index
        
        Returns:
            Loss value
        """
        midi_tokens = batch["midi_tokens"]
        text_tokens = batch["text_tokens"]
        text_attention_mask = batch.get("text_attention_mask", None)
        midi_attention_mask = batch.get("midi_attention_mask", None)
        
        batch_size = midi_tokens.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(midi_tokens.float())
        
        # Add noise to MIDI tokens
        noisy_midi = self.add_noise(midi_tokens, t, noise)
        
        # Convert back to token space (simplified for now)
        noisy_midi_tokens = noisy_midi.round().long().clamp(0, self.midi_vocab_size - 1)
        
        # Predict the noise
        predicted_logits = self(
            noisy_midi_tokens,
            text_tokens,
            t,
            text_attention_mask,
            midi_attention_mask,
        )
        
        # Calculate loss (cross-entropy with original tokens)
        loss = F.cross_entropy(
            predicted_logits.view(-1, self.midi_vocab_size),
            midi_tokens.view(-1),
            ignore_index=-100,  # Ignore padding tokens
        )
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validation step for the model.
        
        Args:
            batch: Dictionary containing 'midi_tokens', 'text_tokens', and optional masks
            batch_idx: Batch index
        
        Returns:
            Loss value
        """
        midi_tokens = batch["midi_tokens"]
        text_tokens = batch["text_tokens"]
        text_attention_mask = batch.get("text_attention_mask", None)
        midi_attention_mask = batch.get("midi_attention_mask", None)
        
        batch_size = midi_tokens.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(midi_tokens.float())
        
        # Add noise to MIDI tokens
        noisy_midi = self.add_noise(midi_tokens, t, noise)
        
        # Convert back to token space
        noisy_midi_tokens = noisy_midi.round().long().clamp(0, self.midi_vocab_size - 1)
        
        # Predict the noise
        predicted_logits = self(
            noisy_midi_tokens,
            text_tokens,
            t,
            text_attention_mask,
            midi_attention_mask,
        )
        
        # Calculate loss
        loss = F.cross_entropy(
            predicted_logits.view(-1, self.midi_vocab_size),
            midi_tokens.view(-1),
            ignore_index=-100,
        )
        
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    @torch.no_grad()
    def generate(
        self,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor = None,
        max_length: int = 512,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """Generate MIDI sequence from text description.
        
        Args:
            text_tokens: Text token IDs of shape (batch_size, text_length)
            text_attention_mask: Attention mask for text
            max_length: Maximum length of generated sequence
            num_inference_steps: Number of denoising steps
        
        Returns:
            Generated MIDI tokens of shape (batch_size, max_length)
        """
        batch_size = text_tokens.shape[0]
        
        # Start with random noise
        midi_tokens = torch.randint(
            0, self.midi_vocab_size, (batch_size, max_length), device=self.device
        )
        
        # Denoising loop
        timesteps = torch.linspace(self.timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)
        
        for t in timesteps:
            t_batch = t.repeat(batch_size)
            
            # Predict
            predicted_logits = self(
                midi_tokens,
                text_tokens,
                t_batch,
                text_attention_mask,
            )
            
            # Sample from predicted distribution
            midi_tokens = torch.argmax(predicted_logits, dim=-1)
        
        return midi_tokens
