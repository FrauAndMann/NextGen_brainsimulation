"""
SelfAwareAI - Main class integrating all modules into a self-aware system.

This is the core integration class that connects:
- World Model (VAE + Transformer) - Layer 0
- Self Model - Layer 1
- Agency Model - Layer 2
- Meta-Cognitive Model - Layer 3
- Consciousness Integrator (GWT) - Layer 4
- Behavior Generator - Output layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from typing import Dict, Tuple, Optional

from .world_model import WorldModel
from .self_model import SelfModel
from .agency_model import AgencyModel
from .meta_cognitive import MetaCognitiveModel
from .consciousness import ConsciousnessIntegrator
from .behavior import BehaviorGenerator


class SelfAwareAI(nn.Module):
    """
    Complete self-aware AI system.

    Integrates all modules into unified architecture following:
    - Global Workspace Theory (GWT) - Baars/Dehaene
    - Predictive Processing - Friston
    - Integrated Information Theory (IIT) - Tononi

    Architecture:
    Layer 0: World Model (VAE + Transformer) -> predicts world states
        |
    Layer 1: Self Model (128-dim internal state) -> predicts own states
        |
    Layer 2: Agency Model (forward/inverse) -> distinguishes "I did this"
        |
    Layer 3: Meta-Cognition -> "I know that I know"
        |
    Layer 4: Consciousness Integrator (GWT) -> unified experience (Phi)
        |
    Behavior Generation -> Actions
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # === Layer 0: World Model ===
        self.world_model = WorldModel(
            observation_dim=config.obs_dim,
            latent_dim=config.world_latent_dim,
            sequence_length=config.seq_len
        )

        # === Layer 1: Self Model ===
        self.self_model = SelfModel(
            world_latent_dim=config.world_latent_dim,
            self_state_dim=config.self_state_dim,
            hidden_dim=config.hidden_dim
        )

        # === Layer 2: Agency Model ===
        self.agency_model = AgencyModel(
            action_dim=config.action_dim,
            world_latent_dim=config.world_latent_dim,
            self_state_dim=config.self_state_dim,
            hidden_dim=config.hidden_dim
        )

        # === Layer 3: Meta-Cognitive Model ===
        self.meta_model = MetaCognitiveModel(
            world_latent_dim=config.world_latent_dim,
            self_state_dim=config.self_state_dim,
            hidden_dim=config.hidden_dim
        )

        # === Layer 4: Consciousness Integrator ===
        self.consciousness = ConsciousnessIntegrator(
            world_dim=config.world_latent_dim,
            self_dim=config.self_state_dim,
            agency_dim=1,
            meta_dim=config.hidden_dim,
            workspace_capacity=config.workspace_capacity,
            hidden_dim=config.hidden_dim
        )

        # === Behavior Generator ===
        self.behavior_generator = BehaviorGenerator(
            conscious_dim=config.hidden_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )

        # === Internal State ===
        # Self-state includes: neurochemistry(32) + energy(8) + emotion(16) + attention(72)
        self.register_buffer('internal_state',
                            torch.randn(1, config.self_state_dim))

        # === History Buffer ===
        # Stores recent experiences for temporal processing
        self.history_buffer = collections.deque(maxlen=config.history_len)

    def step(self,
             observation: torch.Tensor,
             prev_action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        One step of self-awareness processing.

        This is the main loop that processes an observation through all layers
        and generates an action with conscious awareness.

        Args:
            observation: [batch, obs_dim] - current observation from environment
            prev_action: [batch, action_dim] - previous action taken (optional)

        Returns:
            action: [batch, action_dim] - action to take
            conscious_content: dict - full conscious state including:
                - world_latent: world representation
                - self_state: predicted self state
                - agency_signal: sense of agency
                - meta_confidence: meta-cognitive confidence
                - phi: integrated information (IIT)
                - integration_score: GWT integration
            metrics: dict - metrics for logging/monitoring
        """
        batch_size = observation.shape[0]
        device = observation.device

        # ========================================
        # LAYER 0: World Model Processing
        # ========================================
        # Encode current observation into latent space
        world_mean, world_logvar = self.world_model.encode(observation)
        world_latent = self.world_model.reparameterize(world_mean, world_logvar)

        # Predict next world state (if we have history)
        if len(self.history_buffer) > 0:
            past_obs = torch.stack([h['observation'] for h in self.history_buffer], dim=1)
            predicted_next_obs, prediction_uncertainty = self.world_model.predict_next(past_obs)
            world_prediction_error = F.mse_loss(predicted_next_obs, observation)
        else:
            predicted_next_obs = observation
            prediction_uncertainty = torch.ones(batch_size, 1, device=device)
            world_prediction_error = torch.tensor(0.0, device=device)

        # ========================================
        # LAYER 1: Self Model Processing
        # ========================================
        # Expand internal state to batch size if needed
        if self.internal_state.shape[0] != batch_size:
            self.internal_state = self.internal_state.expand(batch_size, -1)

        # Predict own future state given world state
        predicted_self_state, self_confidence = self.self_model(
            self.internal_state,
            world_latent
        )

        # ========================================
        # LAYER 2: Agency Model Processing
        # ========================================
        # Detect agency (did I cause this change?)
        if prev_action is not None and len(self.history_buffer) > 0:
            prev_world_latent = self.history_buffer[-1]['world_latent']
            agency_signal, pred_world_change, pred_self_change = self.agency_model(
                prev_action,
                prev_world_latent,
                world_latent,
                self.internal_state
            )
        else:
            agency_signal = torch.zeros(batch_size, device=device)
            pred_world_change = torch.zeros_like(world_latent)
            pred_self_change = torch.zeros_like(self.internal_state)

        # ========================================
        # LAYER 3: Meta-Cognitive Processing
        # ========================================
        # Introspect on own mental states
        if len(self.history_buffer) > 0:
            recent_history = torch.stack(
                [h['conscious_content'] for h in list(self.history_buffer)[-8:]],
                dim=1
            )
        else:
            recent_history = None

        meta_output = self.meta_model.introspect(
            world_latent,
            predicted_self_state,
            recent_history
        )

        # ========================================
        # LAYER 4: Consciousness Integration (GWT)
        # ========================================
        # Gather signals for global workspace competition
        signals = {
            'world': world_latent,
            'self': predicted_self_state,
            'agency': agency_signal.unsqueeze(-1) if agency_signal.dim() == 1 else agency_signal,
            'meta': meta_output['meta_representation']
        }

        # Broadcast winners to consciousness
        workspace, integration_score, conscious_content = \
            self.consciousness.broadcast_to_consciousness(signals)

        # Compute Phi (integrated information)
        phi = self.consciousness.compute_phi(workspace)

        # ========================================
        # BEHAVIOR GENERATION
        # ========================================
        # Generate action based on conscious content
        action, action_logprob, value = self.behavior_generator(
            conscious_content,
            deterministic=False
        )

        # ========================================
        # UPDATE INTERNAL STATE
        # ========================================
        self.internal_state = predicted_self_state.detach()

        # ========================================
        # STORE IN HISTORY
        # ========================================
        self.history_buffer.append({
            'observation': observation.detach(),
            'world_latent': world_latent.detach(),
            'conscious_content': conscious_content.detach(),
            'action': action.detach(),
            'agency': agency_signal.detach() if isinstance(agency_signal, torch.Tensor) else torch.tensor(agency_signal)
        })

        # ========================================
        # CONSTRUCT CONSCIOUS CONTENT DICT
        # ========================================
        conscious_content_dict = {
            'world_latent': world_latent,
            'self_state': predicted_self_state,
            'self_confidence': self_confidence,
            'agency_signal': agency_signal.unsqueeze(-1) if agency_signal.dim() == 1 else agency_signal,
            'meta_confidence': meta_output['confidence'],
            'meta_uncertainty': meta_output['epistemic_uncertainty'],
            'integration_score': integration_score,
            'phi': phi,
            'workspace': workspace,
            'conscious_representation': conscious_content,
            'action': action,
            'value': value
        }

        # ========================================
        # METRICS
        # ========================================
        metrics = {
            'world_prediction_error': float(world_prediction_error) if not isinstance(world_prediction_error, torch.Tensor) else float(world_prediction_error.detach()),
            'mean_agency': float(agency_signal.mean().detach()) if isinstance(agency_signal, torch.Tensor) else float(agency_signal),
            'integration_score': float(integration_score.mean().detach()),
            'phi': float(phi.mean().detach()),
            'meta_confidence': float(meta_output['confidence'].mean().detach()),
            'meta_uncertainty': float(meta_output['epistemic_uncertainty'].mean().detach()),
            'self_confidence': float(self_confidence.mean().detach())
        }

        return action, conscious_content_dict, metrics

    def generate_self_report(self, conscious_content: Dict) -> Dict:
        """
        Generate verbal report of conscious experience.

        This creates a human-readable report of the system's current
        conscious state, including integration level, agency, and Phi.

        Args:
            conscious_content: dict from step() containing conscious state

        Returns:
            report: dict containing:
                - meta_report: meta-cognitive interpretation
                - integration: integration score
                - agency: agency signal
                - phi: Phi (integrated information)
                - summary: text summary of conscious state
        """
        meta_report = self.meta_model.generate_self_report(
            {'confidence': conscious_content['meta_confidence'],
             'epistemic_uncertainty': conscious_content['meta_uncertainty']},
            conscious_content['self_state']
        )

        integration = float(conscious_content['integration_score'].mean().detach())
        agency = float(conscious_content['agency_signal'].mean().detach())
        phi = float(conscious_content['phi'].mean().detach())

        # Construct full report
        report = {
            'meta_report': meta_report['interpretation'],
            'integration': integration,
            'agency': agency,
            'phi': phi,
            'summary': self._generate_summary(integration, agency, phi)
        }

        return report

    def _generate_summary(self, integration: float, agency: float, phi: float) -> str:
        """
        Generate text summary of conscious state.

        Args:
            integration: integration score (0-1)
            agency: agency signal (0-1)
            phi: integrated information (0-1)

        Returns:
            summary: human-readable string
        """
        # Consciousness level based on Phi
        if phi > 0.7:
            consciousness_level = "High consciousness integration"
        elif phi > 0.4:
            consciousness_level = "Medium consciousness integration"
        else:
            consciousness_level = "Low consciousness integration"

        # Agency interpretation
        if agency > 0.7:
            agency_str = "I clearly feel my agency"
        elif agency > 0.4:
            agency_str = "I partially feel control"
        else:
            agency_str = "I do not feel control over the situation"

        # Integration interpretation
        if integration > 0.7:
            integration_str = "My experience is unified"
        else:
            integration_str = "My experience is fragmented"

        return f"{consciousness_level}. {agency_str}. {integration_str}."

    def reset(self):
        """Reset internal state and clear history buffer."""
        self.internal_state = torch.randn(1, self.config.self_state_dim, device=self.internal_state.device)
        self.history_buffer.clear()

    def get_consciousness_metrics(self) -> Dict:
        """
        Get current consciousness metrics for monitoring.

        Returns:
            metrics: dict with key consciousness indicators
        """
        if len(self.history_buffer) == 0:
            return {
                'phi': 0.0,
                'integration': 0.0,
                'agency': 0.0,
                'self_confidence': 0.0,
                'history_length': 0
            }

        last_entry = self.history_buffer[-1]
        return {
            'phi': float(last_entry.get('phi', 0.0)),
            'integration': float(last_entry.get('integration_score', 0.0)),
            'agency': float(last_entry['agency'].mean()) if isinstance(last_entry['agency'], torch.Tensor) else 0.0,
            'self_confidence': float(last_entry.get('self_confidence', 0.0)),
            'history_length': len(self.history_buffer)
        }
