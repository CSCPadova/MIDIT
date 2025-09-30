import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import torchvision.utils as vutils


class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.l1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.l1 = nn.Linear(latent_dim, 64)
        self.l2 = nn.Linear(64, 28 * 28)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        return x


class VAE(L.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        lr: float = 1e-3,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
    ):
        super().__init__()
        self.lr = lr
        self.encoder = Encoder(latent_dim) if encoder is None else encoder
        self.decoder = Decoder(latent_dim) if decoder is None else decoder
        self.latent_dim = latent_dim
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        r"""Applica la reparametrizzazione:

        .. math::
            z = \mu + \sigma * \epsilon
        """
        std = torch.exp(0.5 * logvar)  # Calcola la deviazione standard
        eps = torch.randn_like(std)  # Sample casuale N(0,1)
        return mu + std * eps  # Sample z dalla distribuzione

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # training_step defines the train loop.
        x, x_hat, mu, logvar = self._common_step(batch, batch_idx)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + kl_loss

        self.log_dict(
            {
                "train_loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
            }
        )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # this is the validation loop
        x, x_hat, mu, logvar = self._common_step(batch, batch_idx)
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        val_loss = reconstruction_loss + kl_loss
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # this is the test loop
        x, x_hat, mu, logvar = self._common_step(batch, batch_idx)

        reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        test_loss = reconstruction_loss + kl_loss

        self.log("test_loss", test_loss)
        return test_loss

    def _common_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, _ = batch
        x = x.view(x.size(0), -1)

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)

        return x, x_hat, mu, logvar

    def on_train_epoch_end(self):
        """Salva immagini ogni 10 epoche."""
        if self.current_epoch % 10 == 0:  # Ogni 10 epoche
            self.log_generated_images("Generated Images (Every 10 Epochs)")

    def on_train_end(self):
        """Salva immagini alla fine del training."""
        self.log_generated_images("Generated Images (Final)")

    def log_generated_images(self, tag: str):
        """Genera e logga immagini in TensorBoard."""
        z = torch.randn(
            16, self.latent_dim, device=self.device
        )  # 16 campioni dal latent space
        generated_images = self.decoder(z).view(
            -1, 1, 28, 28
        )  # Reshape per immagini MNIST
        grid = vutils.make_grid(generated_images, normalize=True, nrow=4)
        self.logger.experiment.add_image(tag, grid, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
