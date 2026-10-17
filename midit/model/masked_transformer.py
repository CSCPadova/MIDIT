"""Masked Transformer for text-to-MIDI generation with REMI+ tokenization.
Phase 1: Pre-training on Lakh dataset (MIDI-only, no text conditioning)
Phase 2: Fine-tuning on MidiCaps (text-to-MIDI with FLAN-T5 conditioning)

This implementation supports both phases with a single model architecture.
"""

import sys
from pathlib import Path

# Add parent directory to path for testing
if __name__ == "__main__":
    parent_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(parent_dir))
    
    # Now use absolute imports for testing
    from midit.model.text_encoder import FlanT5TextEncoder
    from midit.model.multi_head_output import MultiHeadOutput
else:
    # Normal relative imports when used as package
    from .text_encoder import FlanT5TextEncoder
    from .multi_head_output import MultiHeadOutput    

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from typing import Dict, Optional


class MaskedTransformerREMI(L.LightningModule):
    """Masked Transformer for REMI+ MIDI generation.
    
    Supports two training phases:
    - Phase 1: Pre-training (MIDI-only, learns musical structure)
    - Phase 2: Fine-tuning (text-conditioned, learns text→MIDI mapping)
    
    Architecture:
        Phase 1: MIDI Tokens (masked) → Transformer → Multi-Head Output
        Phase 2: [Text Embeddings | MIDI Tokens] → Transformer → Multi-Head Output
    """
    
    def __init__(
        self,
        # Phase config
        phase: int = 1,  # 1=pre-training (no text), 2=fine-tuning (with text)
        
        # Tokenizer config
        vocab_sizes: Optional[Dict[str, int]] = None,
        special_token_ids: Optional[Dict[str, int]] = None,
        
        # Model config
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 8192,
        
        # Text encoder config (only Phase 2)
        text_encoder_config: Optional[Dict] = None,
        
        # Masking config
        mask_prob: float = 0.15,
        bar_level_masking: bool = True,
        
        # Loss config
        lambda_weights: Optional[Dict[str, float]] = None,
        class_weights: Optional[Dict] = None,  # ← NEW!
        
        # Optimizer config
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        num_training_steps: int = 0,   # set explicitly to fix estimated_stepping_batches=inf
    ):
        """Initialize the Masked Transformer.
        
        Args:
            phase: Training phase (1=pre-training, 2=fine-tuning)
            vocab_sizes: Dictionary of vocabulary sizes for each token type
            special_token_ids: Special tokens (PAD, BOS, EOS, MASK)
            d_model: Hidden dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            text_encoder_config: Text encoder settings (only Phase 2)
            mask_prob: Masking probability
            bar_level_masking: Use bar-level masking
            lambda_weights: Loss weights per token type
            class_weights: Class weights config for imbalanced tokens (NEW!)
            lr: Learning rate
            weight_decay: Weight decay
            warmup_steps: Warmup steps for scheduler
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.phase = phase
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob
        self.bar_level_masking = bar_level_masking
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps

        print(f"\n{'='*70}")
        print(f"🎵 Initializing MaskedTransformerREMI - Phase {phase}")
        print(f"{'='*70}\n")
        
        # Vocab sizes
        if vocab_sizes is None:
            vocab_sizes = {
                'pitch': 89,
                'velocity': 32,
                'duration': 128,
                'program': 129,
                'position': 32,
                'bar': 2,
                'tempo': 20,
                'timesig': 10,
                'rest': 50,
            }
        self.vocab_sizes = vocab_sizes
        
        # Special tokens
        if special_token_ids is None:
            special_token_ids = {
                'pad': 515,
                'bos': 516,
                'eos': 517,
                'mask': 518,
            }
        self.special_token_ids = special_token_ids
        
        # Lambda weights
        if lambda_weights is None:
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
        self.lambda_weights = lambda_weights
        
        # ====================================================================
        # CLASS WEIGHTS (NEW!) - Backward compatible
        # ====================================================================
        self.class_weights_config = class_weights
        
        if class_weights is not None and class_weights.get('enabled', False):
            print(f"\n{'='*70}")
            print("⚖️  CLASS WEIGHTS ENABLED")
            print(f"{'='*70}")
            
            scaling = class_weights.get('scaling', 'raw')
            max_weight = class_weights.get('max_weight', None)
            
            import json
            weights_file = Path(class_weights.get('weights_file', 'class_weights.json'))
            
            if weights_file.exists():
                with open(weights_file, 'r') as f:
                    raw_weights = json.load(f)
                
                for token_type in class_weights.get('token_types', []):
                    if token_type in raw_weights:
                        weights = raw_weights[token_type]
                        
                        # Apply scaling
                        if scaling == 'sqrt':
                            import math
                            weights = [math.sqrt(w) for w in weights]
                        elif scaling == 'log':
                            import math
                            weights = [math.log(1 + w) for w in weights]
                        
                        # Apply max cap
                        if max_weight is not None:
                            weights = [min(w, max_weight) for w in weights]
                        
                        # Register buffer (Lightning-safe!)
                        weight_tensor = torch.tensor(weights, dtype=torch.float32)
                        self.register_buffer(
                            f"class_weight_{token_type}",
                            weight_tensor,
                            persistent=False
                        )
                        
                        print(f"  {token_type:10s}: {len(weights):3d} weights")
                        print(f"    Scaling: {scaling}")
                        print(f"    Range: [{min(weights):.2f}, {max(weights):.2f}]")
                        if max_weight:
                            print(f"    Max cap: {max_weight}")
            else:
                print(f"  ⚠️  Weights file not found: {weights_file}")
                print(f"  Continuing without class weights")
            
            print(f"{'='*70}\n")
        
        # ====================================================================
        # TEXT ENCODER (PHASE 2 ONLY)
        # ====================================================================
        if phase == 2 and text_encoder_config is not None:
            print("📝 Phase 2: Initializing FLAN-T5 text encoder...")
            self.text_encoder = FlanT5TextEncoder(
                model_name=text_encoder_config.get('model_name', 'google/flan-t5-base'),
                embed_dim=d_model,
                freeze=text_encoder_config.get('freeze', True),
            )
            self.use_text_conditioning = True
            print("✅ Text encoder initialized (frozen)\n")
        else:
            print("📝 Phase 1: No text conditioning (MIDI-only training)\n")
            self.text_encoder = None
            self.use_text_conditioning = False
        
        # ====================================================================
        # MIDI EMBEDDINGS
        # ====================================================================
        total_vocab_size = sum(vocab_sizes.values()) + 4  # + special tokens
        self.token_embedding = nn.Embedding(total_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        print(f"🎼 MIDI embeddings:")
        print(f"   Total vocab size: {total_vocab_size:,}")
        print(f"   Embedding dim: {d_model}\n")
        
        # ====================================================================
        # TRANSFORMER ENCODER
        # ====================================================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        print(f"🔄 Transformer encoder:")
        print(f"   Layers: {num_encoder_layers}")
        print(f"   Heads: {nhead}")
        print(f"   Hidden dim: {d_model}")
        print(f"   Feedforward dim: {dim_feedforward}\n")
        
        # ====================================================================
        # MULTI-HEAD OUTPUT
        # ====================================================================
        self.multi_head_output = MultiHeadOutput(
            d_model=d_model,
            vocab_sizes=vocab_sizes,
            dropout=dropout,
        )
        
        print(f"🎯 Multi-head output: 9 token types\n")
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"📊 Model statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"{'='*70}\n")
    
    def get_class_weights_dict(self) -> Dict[str, torch.Tensor]:
        """Get class weights as dictionary for passing to loss computation.
        
        Returns:
            Dictionary {token_type: weight_tensor} or empty dict if no weights
        """
        class_weights_dict = {}
        
        if self.class_weights_config is not None and self.class_weights_config.get('enabled', False):
            for token_type in self.class_weights_config.get('token_types', []):
                weight = getattr(self, f"class_weight_{token_type}", None)
                if weight is not None:
                    class_weights_dict[token_type] = weight
        
        return class_weights_dict
    
    def apply_bar_masking(
            self,
            tokens: torch.Tensor,
            bar_positions: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Apply bar-level masking.
            
            Args:
                tokens: Token IDs (batch, seq)
                bar_positions: Bar positions (batch, num_bars) - optional
            
            Returns:
                masked_tokens: Tokens with masking applied
                mask_indices: Boolean mask of masked positions
            """
            batch_size, seq_length = tokens.shape
            device = tokens.device
            
            # Create mask
            mask_indices = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
            
            if self.bar_level_masking and bar_positions is not None:
                # Bar-level masking
                for b in range(batch_size):
                    bar_pos = bar_positions[b]
                    num_bars = len(bar_pos[bar_pos >= 0])  # Count valid bars
                    
                    if num_bars == 0:
                        continue
                    
                    # Select bars to mask (15%)
                    num_mask = max(1, int(num_bars * self.mask_prob))
                    mask_bar_indices = torch.randperm(num_bars, device=device)[:num_mask]
                    
                    # Mask all tokens in selected bars
                    for bar_idx in mask_bar_indices:
                        if bar_idx >= len(bar_pos) - 1:
                            continue
                        start_pos = bar_pos[bar_idx]
                        end_pos = bar_pos[bar_idx + 1] if bar_idx + 1 < num_bars else seq_length
                        if start_pos >= 0 and end_pos <= seq_length:
                            mask_indices[b, start_pos:end_pos] = True
            else:
                # Token-level masking (fallback)
                mask_indices = torch.rand(batch_size, seq_length, device=device) < self.mask_prob
            
            # Apply masking: 80% MASK, 10% random, 10% unchanged
            masked_tokens = tokens.clone()
            mask_token_id = self.special_token_ids['mask']
            
            for b in range(batch_size):
                masked_positions = mask_indices[b].nonzero(as_tuple=True)[0]
                
                for pos in masked_positions:
                    rand = torch.rand(1, device=device).item()
                    
                    if rand < 0.8:
                        # 80%: Replace with [MASK]
                        masked_tokens[b, pos] = mask_token_id

                    elif rand < 0.9:
                        # 10%: Replace with random token (include special tokens)
                        random_token = torch.randint(
                            0, self.token_embedding.num_embeddings,  # ← FIXED!
                            (1,), device=device
                        ).item()
                        masked_tokens[b, pos] = random_token
                    # else: 10% unchanged
            
            return masked_tokens, mask_indices
    
    def forward(
        self,
        midi_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        midi_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            midi_tokens: MIDI token IDs (batch, seq)
            text_tokens: Text token IDs (batch, text_len) - Phase 2 only
            text_attention_mask: Text mask (batch, text_len) - Phase 2 only
            midi_attention_mask: MIDI mask (batch, seq)
        
        Returns:
            Hidden states (batch, seq, d_model) for multi-head output
        """
        batch_size, seq_length = midi_tokens.shape
        device = midi_tokens.device
        
        # MIDI embeddings
        token_embeds = self.token_embedding(midi_tokens)
        positions = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        midi_embeds = token_embeds + pos_embeds
        midi_embeds = self.layer_norm(midi_embeds)
        
        # Text conditioning (Phase 2 only)
        if self.use_text_conditioning and text_tokens is not None:
            # Encode text
            text_embeds = self.text_encoder(text_tokens, text_attention_mask)
            
            # Concatenate [text | midi]
            combined_embeds = torch.cat([text_embeds, midi_embeds], dim=1)
            
            # Combined mask
            if midi_attention_mask is not None:
                if text_attention_mask is not None:
                    combined_mask = torch.cat([text_attention_mask, midi_attention_mask], dim=1)
                else:
                    text_mask = torch.ones(batch_size, text_embeds.shape[1], device=device)
                    combined_mask = torch.cat([text_mask, midi_attention_mask], dim=1)
                combined_mask = ~combined_mask.bool()
            else:
                combined_mask = None
        else:
            # Phase 1: MIDI only
            combined_embeds = midi_embeds
            combined_mask = ~midi_attention_mask.bool() if midi_attention_mask is not None else None
        
        # Transformer encoding
        encoded = self.transformer_encoder(
            combined_embeds,
            src_key_padding_mask=combined_mask,
        )
        
        # Extract MIDI portion if text was used
        if self.use_text_conditioning and text_tokens is not None:
            text_len = text_embeds.shape[1]
            midi_encoded = encoded[:, text_len:, :]
        else:
            midi_encoded = encoded
        
        return midi_encoded
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step with masking and weighted loss.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
        
        Returns:
            Total loss
        """
        midi_tokens = batch["midi_tokens"]
        targets = batch["targets"]  # Dict of targets per token type
        midi_attention_mask = batch.get("midi_attention_mask", None)
        bar_positions = batch.get("bar_positions", None)
        mask_labels = batch.get("mask_labels", None)  # From dataset masking

        # Phase 2: optional text
        text_tokens = batch.get("text_tokens", None) if self.phase == 2 else None
        text_attention_mask = batch.get("text_attention_mask", None) if self.phase == 2 else None

        # midi_tokens already masked by dataset — no second masking applied here

        # Forward pass
        hidden_states = self(
            midi_tokens,
            text_tokens,
            text_attention_mask,
            midi_attention_mask,
        )

        # Get class weights
        class_weights_dict = self.get_class_weights_dict()

        # Compute multi-head loss (with class weights)
        losses = self.multi_head_output.compute_loss(
            hidden_states,
            targets,
            self.lambda_weights,
            mask=mask_labels,
            class_weights=class_weights_dict,
        )
        
        # Log losses
        self.log("train_loss", losses['total'], prog_bar=True, sync_dist=True)
        for token_type, loss_value in losses.items():
            if token_type != 'total':
                self.log(f"train/{token_type}", loss_value, sync_dist=True)
        
        return losses['total']
    
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
        
        Returns:
            Validation loss
        """
        midi_tokens = batch["midi_tokens"]
        targets = batch["targets"]
        midi_attention_mask = batch.get("midi_attention_mask", None)
        bar_positions = batch.get("bar_positions", None)
        mask_labels = batch.get("mask_labels", None)  # From dataset masking

        # Phase 2: optional text
        text_tokens = batch.get("text_tokens", None) if self.phase == 2 else None
        text_attention_mask = batch.get("text_attention_mask", None) if self.phase == 2 else None

        # midi_tokens already masked by dataset — no second masking applied here

        # Forward pass
        hidden_states = self(
            midi_tokens,
            text_tokens,
            text_attention_mask,
            midi_attention_mask,
        )

        # Get class weights
        class_weights_dict = self.get_class_weights_dict()

        # Compute loss (with class weights)
        losses = self.multi_head_output.compute_loss(
            hidden_states,
            targets,
            self.lambda_weights,
            mask=mask_labels,
            class_weights=class_weights_dict,
        )
        
        # Log
        self.log("val_loss", losses['total'], prog_bar=True, sync_dist=True)
        for token_type, loss_value in losses.items():
            if token_type != 'total':
                self.log(f"val/{token_type}", loss_value, sync_dist=True)
        
        return losses['total']

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Use explicitly-set total steps; fall back to Lightning estimate only if not set.
        # estimated_stepping_batches returns inf with DataLoaderWrapper, breaking cosine decay.
        total_steps = (self.num_training_steps if self.num_training_steps > 0
                       else self.trainer.estimated_stepping_batches)

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = min(1.0, float(current_step - self.warmup_steps) / float(max(1, total_steps - self.warmup_steps)))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.141592653589793 * progress))))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# ============================================================================
# Testing
# ============================================================================

def test_phase1():
    """Test Phase 1 (MIDI-only)."""
    print("\n" + "="*70)
    print("🧪 TESTING PHASE 1 (MIDI-only)")
    print("="*70 + "\n")
    
    # Config
    batch_size = 2
    seq_length = 128
    
    # Create model (Phase 1)
    model = MaskedTransformerREMI(
        phase=1,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        max_seq_length=seq_length,
    )
    
    # Test data - use correct vocab size
    total_vocab = sum(model.vocab_sizes.values()) + 4
    midi_tokens = torch.randint(0, total_vocab, (batch_size, seq_length))
    
    print(f"📊 Test config:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Vocab size: {total_vocab}\n")
    
    # Forward (no text!)
    hidden_states = model(midi_tokens)
    
    print(f"✅ Phase 1 forward pass successful")
    print(f"   Hidden states shape: {hidden_states.shape}\n")
    
    # Get predictions
    logits = model.multi_head_output(hidden_states)
    print(f"✅ Multi-head output successful")
    print(f"   Logits keys: {list(logits.keys())}\n")


def test_phase2():
    """Test Phase 2 (text-conditioned)."""
    print("\n" + "="*70)
    print("🧪 TESTING PHASE 2 (Text-conditioned)")
    print("="*70 + "\n")
    
    # Config
    batch_size = 2
    seq_length = 128
    text_length = 32
    
    # Create model (Phase 2)
    model = MaskedTransformerREMI(
        phase=2,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        max_seq_length=seq_length,
        text_encoder_config={
            'model_name': 'google/flan-t5-base',
            'freeze': True,
        },
    )
    
    # Test data
    total_vocab = sum(model.vocab_sizes.values()) + 4
    midi_tokens = torch.randint(0, total_vocab, (batch_size, seq_length))
    text_tokens = torch.randint(0, 32000, (batch_size, text_length))
    
    print(f"📊 Test config:")
    print(f"   Batch size: {batch_size}")
    print(f"   MIDI sequence length: {seq_length}")
    print(f"   Text sequence length: {text_length}")
    print(f"   MIDI vocab size: {total_vocab}\n")
    
    # Forward (with text!)
    hidden_states = model(midi_tokens, text_tokens)
    
    print(f"✅ Phase 2 forward pass successful")
    print(f"   Hidden states shape: {hidden_states.shape}\n")


if __name__ == "__main__":
    print("\n🎵 Testing MaskedTransformerREMI\n")
    
    # Test Phase 1 (priority!)
    test_phase1()
    
    # Test Phase 2 (future)
    # Uncomment when needed:
    # test_phase2()
    
    print("="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70 + "\n")