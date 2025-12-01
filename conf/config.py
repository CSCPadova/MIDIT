"""Configuration dataclasses for MIDIT project."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Tokenizer:
    """Tokenizer configuration for REMI+ tokenization."""
    type: str
    pitch_range: List[int]
    beat_res: Dict[int, int]
    use_programs: bool
    one_token_stream_for_programs: bool
    use_chords: bool
    use_rests: bool
    use_tempos: bool
    use_time_signatures: bool
    vocab_size: int


@dataclass
class Chunking:
    """Chunking strategy configuration (flexible preprocessing)."""
    # Strategy
    strategy: str  # 'bar_based' or 'token_based'
    
    # Bar-based parameters
    bars_per_chunk: int
    overlap_bars: int
    
    # Token-based parameters
    tokens_per_chunk: int
    overlap_tokens: int
    
    # Filtering
    max_tokens: int
    min_tokens_per_bar: int
    filter_empty_bars: bool
    
    # Paths
    raw_data_dir: str
    preprocessed_dir: str


@dataclass
class Masking:
    """Masking configuration for MLM training."""
    mask_prob: float
    bar_level: bool
    mask_strategy: str


@dataclass
class LambdaWeights:
    """Per-token-type loss weights."""
    pitch: float
    velocity: float
    duration: float
    program: float
    position: float
    rest: float
    bar: float
    timesig: float
    tempo: float


@dataclass
class VocabSizes:
    """Vocabulary sizes for each token type."""
    pitch: int
    velocity: int
    duration: int
    program: int
    position: int
    bar: int
    tempo: int
    timesig: int
    rest: int


@dataclass
class SpecialTokens:
    """Special token IDs."""
    pad: int
    bos: int
    eos: int
    mask: int


@dataclass
class Model:
    """Model architecture configuration."""
    type: str
    d_model: int
    nhead: int
    num_encoder_layers: int
    dim_feedforward: int
    dropout: float
    max_seq_length: int
    vocab_sizes: VocabSizes
    special_tokens: SpecialTokens


@dataclass
class Train:
    """Training configuration."""
    phase: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_steps: int
    gradient_clip_val: float
    precision: str
    accumulate_grad_batches: int
    save_top_k: int
    monitor: str
    log_every_n_steps: int
    val_check_interval: float


@dataclass
class Paths:
    """Data and output paths."""
    lakh_dir: str
    preprocessed_phase1: str
    midicaps_dir: str
    preprocessed_phase2: str
    checkpoints_dir: str
    logs_dir: str


@dataclass
class Config:
    """Main configuration."""
    tokenizer: Tokenizer
    chunking: Chunking
    masking: Masking
    lambda_weights: LambdaWeights
    model: Model
    train: Train
    paths: Paths