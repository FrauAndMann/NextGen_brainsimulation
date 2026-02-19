import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class WorldModel(nn.Module):
    """
    Predicts external world based on observations.
    Architecture: VAE + Temporal Transformer
    """

    def __init__(self,
                 observation_dim: int = 512,
                 latent_dim: int = 256,
                 sequence_length: int = 32):
        super().__init__()

        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        # Encoder: observation -> latent distribution (mean + logvar)
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, latent_dim * 2)
        )

        # Temporal model: past latents -> future latent
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Decoder: latent -> predicted observation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, observation_dim)
        )

    def encode(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to latent distribution"""
        h = self.encoder(observation)
        mean, logvar = torch.chunk(h, 2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation"""
        return self.decoder(latent)

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode -> reparameterize -> decode"""
        mean, logvar = self.encode(observation)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar
