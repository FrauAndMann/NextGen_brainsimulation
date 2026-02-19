import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SelfModel(nn.Module):
    """
    Predicts own internal state.
    This is critical: the system models ITSELF, not just the world.
    """

    def __init__(self,
                 world_latent_dim: int = 256,
                 self_state_dim: int = 128,
                 hidden_dim: int = 512):
        super().__init__()

        self.self_state_dim = self_state_dim
        self.world_latent_dim = world_latent_dim
        self.hidden_dim = hidden_dim

        # Components of self-state:
        # - Neurochemistry (dopamine, oxytocin, serotonin, etc.)
        # - Energy level
        # - Emotional valence
        # - Attention focus
        self.neurochemistry_dim = 32
        self.energy_dim = 8
        self.emotion_dim = 16
        self.attention_dim = 72

        assert (self.neurochemistry_dim + self.energy_dim +
                self.emotion_dim + self.attention_dim == self_state_dim), \
            f"Self state dimensions must sum to {self_state_dim}"

        # Self-state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(self_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Combined world + self predictor
        self.self_predictor = nn.Sequential(
            nn.Linear(hidden_dim + world_latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self_state_dim)
        )

        # Self-observation: system observes its own predictions
        self.self_observer = nn.Sequential(
            nn.Linear(self_state_dim * 2, hidden_dim),  # current + predicted
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.Sigmoid()
        )

    def forward(self,
                current_self_state: torch.Tensor,
                world_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next self state

        Args:
            current_self_state: [batch, self_state_dim]
            world_latent: [batch, world_latent_dim]

        Returns:
            predicted_self_state: [batch, self_state_dim]
            self_prediction_confidence: [batch, 64]
        """
        # Encode current self
        self_encoded = self.state_encoder(current_self_state)

        # Combine world and self information
        combined = torch.cat([self_encoded, world_latent], dim=-1)

        # Predict next self state
        next_self_state = self.self_predictor(combined)

        # Observe how well we predicted ourselves
        self_observation = torch.cat([current_self_state, next_self_state], dim=-1)
        confidence = self.self_observer(self_observation)

        return next_self_state, confidence

    def decompose_state(self, self_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose state into components for interpretation
        """
        neurochemistry = self_state[:, :self.neurochemistry_dim]
        energy = self_state[:, self.neurochemistry_dim:self.neurochemistry_dim + self.energy_dim]
        emotion = self_state[:, self.neurochemistry_dim + self.energy_dim:
                              self.neurochemistry_dim + self.energy_dim + self.emotion_dim]
        attention = self_state[:, -self.attention_dim:]

        return {
            'neurochemistry': neurochemistry,
            'energy': energy,
            'emotion': emotion,
            'attention': attention
        }

    def compute_self_prediction_error(self,
                                       predicted_self: torch.Tensor,
                                       actual_self: torch.Tensor) -> torch.Tensor:
        """
        Self prediction error = basis for updating self-model
        """
        error = F.mse_loss(predicted_self, actual_self, reduction='none')

        # Weighted error by components
        components = self.decompose_state(error)

        weighted_error = (
            1.0 * components['neurochemistry'].mean() +
            0.5 * components['energy'].mean() +
            1.5 * components['emotion'].mean() +
            1.0 * components['attention'].mean()
        )

        return weighted_error
