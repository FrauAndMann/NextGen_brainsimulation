import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class AgencyModel(nn.Module):
    """
    Agency model: distinguishes "I did this" from "this happened by itself"
    Key to self-awareness: understanding causality of own actions.
    """

    def __init__(self,
                 action_dim: int = 64,
                 world_latent_dim: int = 256,
                 self_state_dim: int = 128,
                 hidden_dim: int = 512):
        super().__init__()

        self.action_dim = action_dim
        self.world_latent_dim = world_latent_dim
        self.self_state_dim = self_state_dim
        self.hidden_dim = hidden_dim

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )

        # Forward model: action + state -> predicted world change
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim // 2 + world_latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, world_latent_dim)
        )

        # Inverse model: world change -> predicted action
        self.inverse_model = nn.Sequential(
            nn.Linear(world_latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Self-effect model: action -> predicted change in self
        self.self_effect_model = nn.Sequential(
            nn.Linear(hidden_dim // 2 + self_state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self_state_dim)
        )

        # Agency detector: prediction error -> agency signal
        self.agency_detector = nn.Sequential(
            nn.Linear(world_latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self,
                action: torch.Tensor,
                world_state_before: torch.Tensor,
                world_state_after: torch.Tensor,
                self_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute agency signal

        Args:
            action: [batch, action_dim] - what I did
            world_state_before: [batch, world_latent_dim] - world before
            world_state_after: [batch, world_latent_dim] - world after
            self_state: [batch, self_state_dim] - my state

        Returns:
            agency_signal: [batch, 1] - "how much was this me"
            predicted_world_change: [batch, world_latent_dim]
            predicted_self_change: [batch, self_state_dim]
        """
        # Encode action
        action_encoded = self.action_encoder(action)

        # Forward model: predict world change from my action
        predicted_world_change = self.forward_model(
            torch.cat([action_encoded, world_state_before], dim=-1)
        )

        # Inverse model: what action explains the world change?
        inferred_action = self.inverse_model(
            torch.cat([world_state_before, world_state_after], dim=-1)
        )

        # Self-effect: how did my action change me?
        predicted_self_change = self.self_effect_model(
            torch.cat([action_encoded, self_state], dim=-1)
        )

        # Compute agency signal
        # High agency = my predictions matched reality
        actual_world_change = world_state_after - world_state_before
        prediction_error = torch.abs(predicted_world_change - actual_world_change)

        # Agency signal (lower error = higher agency)
        agency_input = torch.cat([prediction_error, action], dim=-1)
        agency_signal = self.agency_detector(agency_input)

        # Additional check: does action match inferred action?
        action_consistency = F.cosine_similarity(action, inferred_action, dim=-1, eps=1e-8)
        action_consistency = (action_consistency + 1) / 2  # [0, 1]

        # Final agency = prediction accuracy * action consistency
        final_agency = agency_signal.squeeze(-1) * action_consistency

        return final_agency, predicted_world_change, predicted_self_change

    def compute_loss(self,
                      action: torch.Tensor,
                      world_before: torch.Tensor,
                      world_after: torch.Tensor,
                      self_state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Training loss for agency model
        """
        agency, pred_world, pred_self = self.forward(
            action, world_before, world_after, self_state
        )

        # Forward model loss
        actual_world_change = world_after - world_before
        forward_loss = F.mse_loss(pred_world, actual_world_change)

        # Inverse model loss
        inferred_action = self.inverse_model(
            torch.cat([world_before, world_after], dim=-1)
        )
        inverse_loss = F.mse_loss(inferred_action, action)

        # Total loss
        loss = forward_loss + inverse_loss

        return loss, {
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'mean_agency': agency.mean().item()
        }
