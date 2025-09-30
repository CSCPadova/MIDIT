from dataclasses import dataclass


@dataclass
class Data:
    data_dir: str
    logs_dir: str
    num_workers: int
    latent_dim: int


@dataclass
class Train:
    accelerator: str
    batch_size: int
    epochs: int


@dataclass
class Optim:
    lr: float


@dataclass
class Config:
    data: Data
    train: Train
    optim: Optim
