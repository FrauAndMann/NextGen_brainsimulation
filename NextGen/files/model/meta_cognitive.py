import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class MetaCognitiveModel(nn.Module):
    """
    Meta-cognition: "I know that I know"
    Recursive layer where system models itself as a prediction system.
    """

    def __init__(self,
                 world_latent_dim: int = 256,
                 self_state_dim: int = 128,
                 hidden_dim: int = 512):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.world_latent_dim = world_latent_dim
        self.self_state_dim = self_state_dim

        # Self-modeling: system models its own processes
        self.process_modeler = nn.Sequential(
            nn.Linear(world_latent_dim + self_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Confidence estimator: how confident am I in predictions?
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Attention mechanism: what am I paying attention to?
        self.attention_generator = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Meta-prediction: what will I predict?
        self.meta_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, world_latent_dim + self_state_dim)
        )

        # Epistemic uncertainty: how uncertain am I?
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()  # Always positive
        )

    def introspect(self,
                   world_state: torch.Tensor,
                   self_state: torch.Tensor,
                   recent_history: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Self-analysis: system looks at itself

        Args:
            world_state: [batch, world_latent_dim]
            self_state: [batch, self_state_dim]
            recent_history: [batch, seq_len, hidden_dim] (optional)

        Returns:
            dict with meta_representation, confidence, attention_weights, etc.
        """
        # Combine current state
        current_state = torch.cat([world_state, self_state], dim=-1)

        # Model own processes
        process_repr = self.process_modeler(current_state)

        # If we have history, attend to it
        if recent_history is not None:
            # Add current to history
            process_repr_exp = process_repr.unsqueeze(1)  # [batch, 1, hidden]
            history_with_current = torch.cat([recent_history, process_repr_exp], dim=1)

            # Self-attention: what's important in recent history?
            attended, attention_weights = self.attention_generator(
                process_repr_exp,
                history_with_current,
                history_with_current
            )

            meta_repr = attended.squeeze(1)
        else:
            meta_repr = process_repr
            attention_weights = None

        # Estimate confidence in predictions
        confidence = self.confidence_estimator(meta_repr)

        # Meta-prediction: what will I predict next moment?
        next_prediction = self.meta_predictor(meta_repr)

        # Epistemic uncertainty
        uncertainty = self.uncertainty_estimator(meta_repr)

        return {
            'meta_representation': meta_repr,
            'confidence': confidence,
            'attention_weights': attention_weights,
            'predicted_next_prediction': next_prediction,
            'epistemic_uncertainty': uncertainty
        }

    def generate_self_report(self,
                              introspection_output: Dict,
                              self_state: torch.Tensor) -> Dict:
        """
        Verbalize introspection
        """
        confidence = float(introspection_output['confidence'].mean().detach())
        uncertainty = float(introspection_output['epistemic_uncertainty'].mean().detach())

        # Decode self_state components
        energy = self_state[:, 8:16].mean().detach().item()  # energy_dim=8, starts at index 32
        emotion_valence = self_state[:, 40:56].mean().detach().item()  # emotion_dim=16

        report = {
            'confidence_level': confidence,
            'uncertainty': uncertainty,
            'energy_level': energy,
            'emotional_valence': emotion_valence,
            'meta_awareness': confidence * (1 - min(uncertainty, 1.0)),
            'interpretation': self._generate_text_interpretation(
                confidence, uncertainty, energy, emotion_valence
            )
        }

        return report

    def _generate_text_interpretation(self,
                                       conf: float,
                                       uncert: float,
                                       energy: float,
                                       valence: float) -> str:
        """Generate human-readable interpretation"""

        if conf > 0.7 and uncert < 0.3:
            state = "I clearly understand my processes"
        elif conf > 0.5:
            state = "I partially understand what is happening"
        else:
            state = "I am in a state of uncertainty"

        if energy > 0.6:
            energy_str = "High energy"
        elif energy > 0.3:
            energy_str = "Medium energy"
        else:
            energy_str = "Low energy"

        if valence > 0.5:
            mood = "positive mood"
        elif valence > -0.2:
            mood = "neutral mood"
        else:
            mood = "negative mood"

        return f"{state}. {energy_str}. I have {mood}."
