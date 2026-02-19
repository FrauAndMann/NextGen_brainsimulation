import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BehaviorGenerator(nn.Module):
    """
    Behavior generation based on conscious content.
    After integration, system decides what to do.
    """

    def __init__(self,
                 conscious_dim: int = 512,
                 action_dim: int = 64,
                 hidden_dim: int = 512):
        super().__init__()

        self.action_dim = action_dim
        self.conscious_dim = conscious_dim
        self.hidden_dim = hidden_dim

        # Policy network: conscious state -> action
        self.policy = nn.Sequential(
            nn.Linear(conscious_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Value network: estimate value of current state
        self.value = nn.Sequential(
            nn.Linear(conscious_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # Action sampler (for stochastic policies)
        self.action_logstd = nn.Parameter(torch.zeros(action_dim))

    def forward(self,
                conscious_content: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate action from conscious content

        Args:
            conscious_content: [batch, conscious_dim]
            deterministic: if True, return mean action

        Returns:
            action: [batch, action_dim]
            action_logprob: [batch, 1]
            value: [batch, 1]
        """
        # Compute action mean
        action_mean = self.policy(conscious_content)

        # Compute value
        value = self.value(conscious_content)

        if deterministic:
            return action_mean, torch.zeros_like(value), value

        # Sample action
        action_std = torch.exp(self.action_logstd)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        action = action_dist.sample()

        # Compute log probability
        action_logprob = action_dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, action_logprob, value
