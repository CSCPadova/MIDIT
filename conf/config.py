from dataclasses import dataclass
from typing import Optional


@dataclass
class Data:
    data_dir: str
    logs_dir: str
    num_workers: int
    max_midi_length: int
    max_text_length: int
    midi_vocab_size: int
    text_vocab_size: int


@dataclass
class Model:
    d_model: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    max_seq_length: int


@dataclass
class TextEncoder:
    model_name: str
    freeze: bool


@dataclass
class Diffusion:
    timesteps: int
    beta_schedule: str


@dataclass
class Train:
    accelerator: str
    batch_size: int
    epochs: int
    fast_run: bool


@dataclass
class Optim:
    lr: float


@dataclass
class Logging:
    use_wandb: bool
    use_tensorboard: bool
    wandb_project: Optional[str]
    wandb_entity: Optional[str]


@dataclass
class Config:
    data: Data
    model: Model
    text_encoder: TextEncoder
    diffusion: Diffusion
    train: Train
    optim: Optim
    logging: Logging
