import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ConsciousnessIntegrator(nn.Module):
    """
    Global Workspace (Global Workspace Theory).
    Bottleneck where information integrates into unified conscious experience.
    Key idea: limited bandwidth creates competition, only most important
    information becomes "conscious".
    """

    def __init__(self,
                 world_dim: int = 256,
                 self_dim: int = 128,
                 agency_dim: int = 1,
                 meta_dim: int = 512,
                 workspace_capacity: int = 16,
                 hidden_dim: int = 512):
        super().__init__()

        self.workspace_capacity = workspace_capacity
        self.hidden_dim = hidden_dim

        # Workspace buffer (learnable initial state)
        self.register_buffer('workspace', torch.zeros(workspace_capacity, hidden_dim))

        # Salience estimators: how important is each information?
        self.salience_estimators = nn.ModuleDict({
            'world': nn.Sequential(
                nn.Linear(world_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'self': nn.Sequential(
                nn.Linear(self_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ),
            'agency': nn.Linear(agency_dim, 1, bias=False),  # Already 0-1
            'meta': nn.Sequential(
                nn.Linear(meta_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        })

        # Projection layers: project all signals to same dimension
        self.projectors = nn.ModuleDict({
            'world': nn.Linear(world_dim, hidden_dim),
            'self': nn.Linear(self_dim, hidden_dim),
            'agency': nn.Linear(agency_dim, hidden_dim),
            'meta': nn.Linear(meta_dim, hidden_dim)
        })

        # Integration mechanism
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.integrator = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Broadcast decoder: from integrated workspace to outputs
        self.broadcast_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def broadcast_to_consciousness(self,
                                    signals: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Competition for access to consciousness.

        Args:
            signals: dict with keys 'world', 'self', 'agency', 'meta'

        Returns:
            workspace_content: [batch, workspace_capacity, hidden_dim]
            integration_score: [batch, 1] - how integrated is the experience
            conscious_content: [batch, hidden_dim] - unified conscious state
        """
        batch_size = next(iter(signals.values())).shape[0]
        device = next(self.parameters()).device

        signal_list = []
        salience_scores = []
        signal_names = []

        # Compute salience for each signal
        for name, signal in signals.items():
            if name in self.salience_estimators:
                # Project to common dimension
                projected = self.projectors[name](signal)

                # Compute salience
                salience = self.salience_estimators[name](signal)

                signal_list.append(projected)
                salience_scores.append(salience)
                signal_names.append(name)

        if not signal_list:
            # No signals
            empty_workspace = torch.zeros(
                batch_size, self.workspace_capacity, self.hidden_dim, device=device
            )
            return empty_workspace, torch.zeros(batch_size, 1, device=device), \
                   torch.zeros(batch_size, self.hidden_dim, device=device)

        # Stack all signals
        all_signals = torch.stack(signal_list, dim=1)  # [batch, num_signals, hidden]
        all_salience = torch.cat(salience_scores, dim=-1)  # [batch, num_signals]

        # Select top-k most salient signals for workspace
        num_signals = all_signals.shape[1]
        k = min(self.workspace_capacity, num_signals)

        # Get indices of top-k salient signals
        top_k_salience, top_k_indices = torch.topk(all_salience, k, dim=-1)

        # Fill workspace with top-k signals
        workspace_content = torch.zeros(
            batch_size, self.workspace_capacity, self.hidden_dim, device=all_signals.device
        )

        for b in range(batch_size):
            for i, idx in enumerate(top_k_indices[b]):
                workspace_content[b, i] = all_signals[b, idx]

        # Integrate information in workspace using Transformer
        integrated_workspace = self.integrator(workspace_content)

        # Compute integration score (Phi-like measure)
        # High integration = all signals are well-connected
        # Low integration = signals are independent

        # Measure 1: Variance in salience (lower = more integrated)
        salience_variance = all_salience.var(dim=-1, keepdim=True)  # [batch, 1]

        # Measure 2: Mutual information proxy (correlation between signals)
        if all_signals.shape[1] > 1:
            signals_flat = all_signals.reshape(batch_size, num_signals, -1)
            # Compute pairwise correlations
            signal_mean = signals_flat.mean(dim=-1, keepdim=True)
            signal_centered = signals_flat - signal_mean
            cov_matrix = torch.bmm(signal_centered, signal_centered.transpose(1, 2))
            correlation = cov_matrix.abs().mean(dim=(1, 2), keepdim=True)  # [batch, 1, 1]
            correlation = correlation.squeeze(-1)  # [batch, 1]
        else:
            correlation = torch.ones(batch_size, 1, device=all_signals.device)

        # Integration score (high correlation, low variance = high integration)
        integration_score = correlation * torch.sigmoid(-salience_variance)  # [batch, 1]

        # Generate unified conscious content (mean pooling over workspace)
        conscious_content = integrated_workspace.mean(dim=1)

        # Broadcast from conscious content
        broadcasted = self.broadcast_decoder(conscious_content)

        return integrated_workspace, integration_score, broadcasted

    def compute_phi(self, workspace_content: torch.Tensor) -> torch.Tensor:
        """
        Approximate Phi (integrated information) estimation
        By Tononi IIT: Phi = effective information across partitions
        Here simplified version via variance and connectivity
        """
        batch_size = workspace_content.shape[0]

        # Variance across workspace elements
        # workspace_content: [batch, workspace_capacity, hidden_dim]
        variance = workspace_content.var(dim=1).mean(dim=-1, keepdim=True)  # [batch, 1]

        # Connectivity (mean absolute inner product)
        workspace_norm = F.normalize(workspace_content, p=2, dim=-1)
        connectivity = torch.bmm(workspace_norm, workspace_norm.transpose(1, 2))
        connectivity_score = connectivity.abs().mean(dim=(1, 2), keepdim=True)  # [batch, 1, 1]
        connectivity_score = connectivity_score.squeeze(-1)  # [batch, 1]

        # Phi = high connectivity * high variance
        phi = connectivity_score * torch.sigmoid(variance)

        return phi
