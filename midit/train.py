import os
import hydra
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from midit.model import VAE
from midit.dataloader import MNISTDataModule
from conf.config import Config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: Config) -> int:

    # makes torch happy, remove this line if you know your GPU
    torch.set_float32_matmul_precision("high")

    logger = TensorBoardLogger(cfg.data.logs_dir, name="midit")
    logger.log_hyperparams(
        {
            "latent_dim": cfg.data.latent_dim,
            "batch_size": cfg.train.batch_size,
            "lr": cfg.optim.lr,
            "epochs": cfg.train.epochs,
            "accelerator": cfg.train.accelerator,
        }
    )

    # model
    autoencoder = VAE(cfg.data.latent_dim, cfg.optim.lr)

    data_module = MNISTDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # train model
    trainer = L.Trainer(
        logger=logger,
        accelerator=cfg.train.accelerator,
        default_root_dir=cfg.data.logs_dir,
        min_epochs=1,
        max_epochs=cfg.train.epochs,
        callbacks=[EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(autoencoder, data_module)
    trainer.validate(autoencoder, data_module)

    return os.EX_OK


if __name__ == "__main__":
    train()
