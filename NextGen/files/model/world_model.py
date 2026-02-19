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

    def predict_next(self,
                     past_observations: torch.Tensor,
                     actions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next observation based on history

        Args:
            past_observations: [batch, seq_len, obs_dim]
            actions: [batch, seq_len, action_dim] (optional)

        Returns:
            predicted_next_obs: [batch, obs_dim]
            uncertainty: [batch, 1]
        """
        batch_size, seq_len, _ = past_observations.shape

        # Encode all past observations
        latents = []
        logvars = []
        for t in range(seq_len):
            mean, logvar = self.encode(past_observations[:, t])
            z = self.reparameterize(mean, logvar)
            latents.append(z)
            logvars.append(logvar)

        latents = torch.stack(latents, dim=1)  # [batch, seq_len, latent_dim]

        # Temporal prediction
        context = self.temporal_model(latents)  # [batch, seq_len, latent_dim]

        # Use last timestep to predict next
        next_latent = context[:, -1, :]  # [batch, latent_dim]

        # Decode to observation space
        predicted_obs = self.decode(next_latent)

        # Estimate uncertainty from latent variance
        last_logvar = logvars[-1]
        uncertainty = torch.exp(last_logvar).mean(dim=-1, keepdim=True)

        return predicted_obs, uncertainty

    def compute_loss(self,
                     observations: torch.Tensor,
                     actions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Training loss: reconstruction + KL divergence
        """
        batch_size, seq_len, _ = observations.shape

        total_recon_loss = torch.tensor(0.0, device=observations.device)
        total_kl_loss = torch.tensor(0.0, device=observations.device)

        for t in range(1, seq_len):
            past_obs = observations[:, :t]
            target_obs = observations[:, t]

            pred_obs, _ = self.predict_next(past_obs,
                                            actions[:, :t] if actions is not None else None)

            # Reconstruction loss
            recon_loss = F.mse_loss(pred_obs, target_obs)
            total_recon_loss = total_recon_loss + recon_loss

            # KL divergence (for VAE)
            mean, logvar = self.encode(target_obs)
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            total_kl_loss = total_kl_loss + kl_loss

        # Average over sequence
        total_recon_loss = total_recon_loss / (seq_len - 1)
        total_kl_loss = total_kl_loss / (seq_len - 1)

        # Total loss with KL weight
        loss = total_recon_loss + 0.001 * total_kl_loss

        return loss, {
            'reconstruction_loss': total_recon_loss.item(),
            'kl_loss': total_kl_loss.item()
        }
