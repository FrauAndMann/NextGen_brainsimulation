# SYNAPSE Brain Simulation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a functionally self-aware AI system with recursive self-prediction, agency detection, meta-cognition, and consciousness integration based on Global Workspace Theory, Predictive Processing, and Integrated Information Theory.

**Architecture:** 5-layer hierarchical system: World Model (VAE+Transformer) → Self Model (128-dim internal state) → Agency Model (forward/inverse dynamics) → Meta-Cognition (confidence/uncertainty) → Consciousness Integrator (GWT with Φ estimation) → Behavior Generation.

**Tech Stack:** PyTorch 2.1+, NumPy, Matplotlib, Seaborn, Wandb (logging), pytest (testing)

---

## Phase 1: Foundation Setup

### Task 1: Project Structure Setup

**Files:**
- Create: `files/requirements.txt`
- Create: `files/__init__.py`

**Step 1: Create requirements.txt**

```txt
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.8.0
seaborn>=0.13.0
wandb>=0.16.0
tqdm>=4.66.0
pytest>=7.4.0
```

**Step 2: Create __init__.py**

```python
# SYNAPSE Brain Simulation Package
from .config import Config, NeurochemistryConfig, EnvironmentConfig, get_fast_config, get_full_config
from .environment import SyntheticEnvironment, NeurochemistryEngine, SyntheticEnvironmentDataset
from .evaluation import SelfAwarenessEvaluator, ConsciousnessVisualizer

__all__ = [
    'Config', 'NeurochemistryConfig', 'EnvironmentConfig',
    'get_fast_config', 'get_full_config',
    'SyntheticEnvironment', 'NeurochemistryEngine', 'SyntheticEnvironmentDataset',
    'SelfAwarenessEvaluator', 'ConsciousnessVisualizer'
]
```

**Step 3: Verify imports work**

Run: `cd D:\Silly\NextGen\files && python -c "from config import Config; c = Config(); print(c)"`
Expected: Config dataclass printed successfully

**Step 4: Commit**

```bash
git add files/requirements.txt files/__init__.py
git commit -m "chore: add requirements.txt and package init"
```

---

### Task 2: Create Model Module Structure

**Files:**
- Create: `files/model/__init__.py`
- Create: `files/model/world_model.py`

**Step 1: Create model directory and init**

```python
# files/model/__init__.py
from .world_model import WorldModel
from .self_model import SelfModel
from .agency_model import AgencyModel
from .meta_cognitive import MetaCognitiveModel
from .consciousness import ConsciousnessIntegrator
from .behavior import BehaviorGenerator
from .self_aware_ai import SelfAwareAI

__all__ = [
    'WorldModel', 'SelfModel', 'AgencyModel',
    'MetaCognitiveModel', 'ConsciousnessIntegrator',
    'BehaviorGenerator', 'SelfAwareAI'
]
```

**Step 2: Write failing test for WorldModel**

Create: `files/tests/test_world_model.py`

```python
import pytest
import torch

def test_world_model_init():
    """Test WorldModel initialization"""
    from model.world_model import WorldModel

    model = WorldModel(
        observation_dim=512,
        latent_dim=256,
        sequence_length=32
    )

    assert model.latent_dim == 256
    assert model.encoder is not None
    assert model.temporal_model is not None
    assert model.decoder is not None
```

**Step 3: Run test to verify it fails**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_world_model.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'model.world_model'"

**Step 4: Implement WorldModel minimal**

Create: `files/model/world_model.py`

```python
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
```

**Step 5: Run test to verify it passes**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_world_model.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add files/model/__init__.py files/model/world_model.py files/tests/test_world_model.py
git commit -m "feat: add WorldModel with VAE encoder/decoder"
```

---

### Task 3: Add WorldModel predict_next Method

**Files:**
- Modify: `files/model/world_model.py`
- Modify: `files/tests/test_world_model.py`

**Step 1: Write failing test for predict_next**

Add to `files/tests/test_world_model.py`:

```python
def test_world_model_predict_next():
    """Test WorldModel temporal prediction"""
    from model.world_model import WorldModel

    model = WorldModel(observation_dim=512, latent_dim=256, sequence_length=32)

    batch_size = 2
    seq_len = 10
    obs_dim = 512

    past_observations = torch.randn(batch_size, seq_len, obs_dim)

    predicted_obs, uncertainty = model.predict_next(past_observations)

    assert predicted_obs.shape == (batch_size, obs_dim)
    assert uncertainty.shape[0] == batch_size
```

**Step 2: Run test to verify it fails**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_world_model.py::test_world_model_predict_next -v`
Expected: FAIL with "AttributeError: 'WorldModel' object has no attribute 'predict_next'"

**Step 3: Implement predict_next**

Add to `files/model/world_model.py` after the `forward` method:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_world_model.py::test_world_model_predict_next -v`
Expected: PASS

**Step 5: Commit**

```bash
git add files/model/world_model.py files/tests/test_world_model.py
git commit -m "feat: add predict_next method to WorldModel"
```

---

### Task 4: Add WorldModel compute_loss Method

**Files:**
- Modify: `files/model/world_model.py`
- Modify: `files/tests/test_world_model.py`

**Step 1: Write failing test for compute_loss**

Add to `files/tests/test_world_model.py`:

```python
def test_world_model_compute_loss():
    """Test WorldModel loss computation"""
    from model.world_model import WorldModel

    model = WorldModel(observation_dim=512, latent_dim=256, sequence_length=32)

    batch_size = 2
    seq_len = 10
    observations = torch.randn(batch_size, seq_len, 512)

    loss, metrics = model.compute_loss(observations)

    assert loss.ndim == 0  # scalar
    assert 'reconstruction_loss' in metrics
    assert 'kl_loss' in metrics
    assert metrics['reconstruction_loss'] >= 0
    assert metrics['kl_loss'] >= 0
```

**Step 2: Run test to verify it fails**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_world_model.py::test_world_model_compute_loss -v`
Expected: FAIL with "AttributeError: 'WorldModel' object has no attribute 'compute_loss'"

**Step 3: Implement compute_loss**

Add to `files/model/world_model.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_world_model.py::test_world_model_compute_loss -v`
Expected: PASS

**Step 5: Commit**

```bash
git add files/model/world_model.py files/tests/test_world_model.py
git commit -m "feat: add compute_loss method to WorldModel"
```

---

## Phase 2: Self Model

### Task 5: Create SelfModel Module

**Files:**
- Create: `files/model/self_model.py`
- Create: `files/tests/test_self_model.py`

**Step 1: Write failing test for SelfModel**

Create `files/tests/test_self_model.py`:

```python
import pytest
import torch

def test_self_model_init():
    """Test SelfModel initialization"""
    from model.self_model import SelfModel

    model = SelfModel(
        world_latent_dim=256,
        self_state_dim=128,
        hidden_dim=512
    )

    assert model.self_state_dim == 128
    assert model.neurochemistry_dim == 32
    assert model.energy_dim == 8
    assert model.emotion_dim == 16
    assert model.attention_dim == 72


def test_self_model_forward():
    """Test SelfModel forward pass"""
    from model.self_model import SelfModel

    model = SelfModel(world_latent_dim=256, self_state_dim=128, hidden_dim=512)

    batch_size = 4
    current_self_state = torch.randn(batch_size, 128)
    world_latent = torch.randn(batch_size, 256)

    next_self_state, confidence = model(current_self_state, world_latent)

    assert next_self_state.shape == (batch_size, 128)
    assert confidence.shape == (batch_size, 64)
```

**Step 2: Run test to verify it fails**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_self_model.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement SelfModel**

Create `files/model/self_model.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_self_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add files/model/self_model.py files/tests/test_self_model.py
git commit -m "feat: add SelfModel with recursive self-prediction"
```

---

## Phase 3: Agency Model

### Task 6: Create AgencyModel Module

**Files:**
- Create: `files/model/agency_model.py`
- Create: `files/tests/test_agency_model.py`

**Step 1: Write failing test for AgencyModel**

Create `files/tests/test_agency_model.py`:

```python
import pytest
import torch

def test_agency_model_init():
    """Test AgencyModel initialization"""
    from model.agency_model import AgencyModel

    model = AgencyModel(
        action_dim=64,
        world_latent_dim=256,
        self_state_dim=128,
        hidden_dim=512
    )

    assert model.action_dim == 64
    assert model.forward_model is not None
    assert model.inverse_model is not None


def test_agency_model_forward():
    """Test AgencyModel forward pass"""
    from model.agency_model import AgencyModel

    model = AgencyModel(action_dim=64, world_latent_dim=256, self_state_dim=128, hidden_dim=512)

    batch_size = 4
    action = torch.randn(batch_size, 64)
    world_before = torch.randn(batch_size, 256)
    world_after = torch.randn(batch_size, 256)
    self_state = torch.randn(batch_size, 128)

    agency, pred_world, pred_self = model(action, world_before, world_after, self_state)

    assert agency.shape == (batch_size, 1) or agency.shape == (batch_size,)
    assert pred_world.shape == (batch_size, 256)
    assert pred_self.shape == (batch_size, 128)
```

**Step 2: Run test to verify it fails**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_agency_model.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement AgencyModel**

Create `files/model/agency_model.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_agency_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add files/model/agency_model.py files/tests/test_agency_model.py
git commit -m "feat: add AgencyModel with forward/inverse dynamics"
```

---

## Phase 4: Meta-Cognition

### Task 7: Create MetaCognitiveModel Module

**Files:**
- Create: `files/model/meta_cognitive.py`
- Create: `files/tests/test_meta_cognitive.py`

**Step 1: Write failing test for MetaCognitiveModel**

Create `files/tests/test_meta_cognitive.py`:

```python
import pytest
import torch

def test_meta_cognitive_init():
    """Test MetaCognitiveModel initialization"""
    from model.meta_cognitive import MetaCognitiveModel

    model = MetaCognitiveModel(
        world_latent_dim=256,
        self_state_dim=128,
        hidden_dim=512
    )

    assert model.hidden_dim == 512
    assert model.process_modeler is not None
    assert model.confidence_estimator is not None


def test_meta_cognitive_introspect():
    """Test MetaCognitiveModel introspect"""
    from model.meta_cognitive import MetaCognitiveModel

    model = MetaCognitiveModel(world_latent_dim=256, self_state_dim=128, hidden_dim=512)

    batch_size = 4
    world_state = torch.randn(batch_size, 256)
    self_state = torch.randn(batch_size, 128)

    output = model.introspect(world_state, self_state)

    assert 'meta_representation' in output
    assert 'confidence' in output
    assert 'epistemic_uncertainty' in output
    assert output['meta_representation'].shape == (batch_size, 512)
    assert output['confidence'].shape == (batch_size, 1)


def test_meta_cognitive_self_report():
    """Test MetaCognitiveModel generate_self_report"""
    from model.meta_cognitive import MetaCognitiveModel

    model = MetaCognitiveModel(world_latent_dim=256, self_state_dim=128, hidden_dim=512)

    batch_size = 2
    world_state = torch.randn(batch_size, 256)
    self_state = torch.randn(batch_size, 128)

    introspection = model.introspect(world_state, self_state)
    report = model.generate_self_report(introspection, self_state)

    assert 'confidence_level' in report
    assert 'uncertainty' in report
    assert 'interpretation' in report
    assert isinstance(report['interpretation'], str)
```

**Step 2: Run test to verify it fails**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_meta_cognitive.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement MetaCognitiveModel**

Create `files/model/meta_cognitive.py`:

```python
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
        confidence = float(introspection_output['confidence'].mean())
        uncertainty = float(introspection_output['epistemic_uncertainty'].mean())

        # Decode self_state components
        energy = self_state[:, 8:16].mean().item()  # energy_dim=8, starts at index 32
        emotion_valence = self_state[:, 40:56].mean().item()  # emotion_dim=16

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
```

**Step 4: Run test to verify it passes**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_meta_cognitive.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add files/model/meta_cognitive.py files/tests/test_meta_cognitive.py
git commit -m "feat: add MetaCognitiveModel with introspection"
```

---

## Phase 5: Consciousness Integration

### Task 8: Create ConsciousnessIntegrator Module

**Files:**
- Create: `files/model/consciousness.py`
- Create: `files/tests/test_consciousness.py`

**Step 1: Write failing test for ConsciousnessIntegrator**

Create `files/tests/test_consciousness.py`:

```python
import pytest
import torch

def test_consciousness_init():
    """Test ConsciousnessIntegrator initialization"""
    from model.consciousness import ConsciousnessIntegrator

    model = ConsciousnessIntegrator(
        world_dim=256,
        self_dim=128,
        agency_dim=1,
        meta_dim=512,
        workspace_capacity=16,
        hidden_dim=512
    )

    assert model.workspace_capacity == 16
    assert model.hidden_dim == 512


def test_consciousness_broadcast():
    """Test ConsciousnessIntegrator broadcast_to_consciousness"""
    from model.consciousness import ConsciousnessIntegrator

    model = ConsciousnessIntegrator(
        world_dim=256, self_dim=128, agency_dim=1,
        meta_dim=512, workspace_capacity=16, hidden_dim=512
    )

    batch_size = 4
    signals = {
        'world': torch.randn(batch_size, 256),
        'self': torch.randn(batch_size, 128),
        'agency': torch.randn(batch_size, 1),
        'meta': torch.randn(batch_size, 512)
    }

    workspace, integration_score, conscious_content = model.broadcast_to_consciousness(signals)

    assert workspace.shape == (batch_size, 16, 512)
    assert integration_score.shape == (batch_size, 1)
    assert conscious_content.shape == (batch_size, 512)


def test_consciousness_phi():
    """Test ConsciousnessIntegrator compute_phi"""
    from model.consciousness import ConsciousnessIntegrator

    model = ConsciousnessIntegrator(
        world_dim=256, self_dim=128, agency_dim=1,
        meta_dim=512, workspace_capacity=16, hidden_dim=512
    )

    batch_size = 2
    workspace = torch.randn(batch_size, 16, 512)

    phi = model.compute_phi(workspace)

    assert phi.shape == (batch_size, 1)
    assert (phi >= 0).all()  # Phi should be non-negative
```

**Step 2: Run test to verify it fails**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_consciousness.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement ConsciousnessIntegrator**

Create `files/model/consciousness.py`:

```python
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
        salience_variance = all_salience.var(dim=-1, keepdim=True)

        # Measure 2: Mutual information proxy (correlation between signals)
        if all_signals.shape[1] > 1:
            signals_flat = all_signals.reshape(batch_size, num_signals, -1)
            # Compute pairwise correlations
            signal_mean = signals_flat.mean(dim=-1, keepdim=True)
            signal_centered = signals_flat - signal_mean
            cov_matrix = torch.bmm(signal_centered, signal_centered.transpose(1, 2))
            correlation = cov_matrix.abs().mean(dim=(1, 2), keepdim=True)
        else:
            correlation = torch.ones(batch_size, 1, device=all_signals.device)

        # Integration score (high correlation, low variance = high integration)
        integration_score = correlation * torch.sigmoid(-salience_variance)

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
        variance = workspace_content.var(dim=1).mean(dim=-1, keepdim=True)

        # Connectivity (mean absolute inner product)
        workspace_norm = F.normalize(workspace_content, p=2, dim=-1)
        connectivity = torch.bmm(workspace_norm, workspace_norm.transpose(1, 2))
        connectivity_score = connectivity.abs().mean(dim=(1, 2), keepdim=True)

        # Phi = high connectivity * high variance
        phi = connectivity_score * torch.sigmoid(variance)

        return phi
```

**Step 4: Run test to verify it passes**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_consciousness.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add files/model/consciousness.py files/tests/test_consciousness.py
git commit -m "feat: add ConsciousnessIntegrator with GWT and Phi estimation"
```

---

## Phase 6: Behavior Generation

### Task 9: Create BehaviorGenerator Module

**Files:**
- Create: `files/model/behavior.py`
- Create: `files/tests/test_behavior.py`

**Step 1: Write failing test for BehaviorGenerator**

Create `files/tests/test_behavior.py`:

```python
import pytest
import torch

def test_behavior_generator_init():
    """Test BehaviorGenerator initialization"""
    from model.behavior import BehaviorGenerator

    model = BehaviorGenerator(
        conscious_dim=512,
        action_dim=64,
        hidden_dim=512
    )

    assert model.action_dim == 64
    assert model.policy is not None
    assert model.value is not None


def test_behavior_generator_forward():
    """Test BehaviorGenerator forward pass"""
    from model.behavior import BehaviorGenerator

    model = BehaviorGenerator(conscious_dim=512, action_dim=64, hidden_dim=512)

    batch_size = 4
    conscious_content = torch.randn(batch_size, 512)

    # Stochastic
    action, logprob, value = model(conscious_content, deterministic=False)
    assert action.shape == (batch_size, 64)
    assert logprob.shape == (batch_size, 1)
    assert value.shape == (batch_size, 1)

    # Deterministic
    action_det, _, value_det = model(conscious_content, deterministic=True)
    assert action_det.shape == (batch_size, 64)
```

**Step 2: Run test to verify it fails**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_behavior.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement BehaviorGenerator**

Create `files/model/behavior.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_behavior.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add files/model/behavior.py files/tests/test_behavior.py
git commit -m "feat: add BehaviorGenerator with policy and value networks"
```

---

## Phase 7: Main Integration

### Task 10: Create SelfAwareAI Main Class

**Files:**
- Create: `files/model/self_aware_ai.py`
- Create: `files/tests/test_self_aware_ai.py`

**Step 1: Write failing test for SelfAwareAI**

Create `files/tests/test_self_aware_ai.py`:

```python
import pytest
import torch
import sys
sys.path.insert(0, 'D:/Silly/NextGen/files')

from config import Config

def test_self_aware_ai_init():
    """Test SelfAwareAI initialization"""
    from model.self_aware_ai import SelfAwareAI

    config = Config()
    model = SelfAwareAI(config)

    assert model.world_model is not None
    assert model.self_model is not None
    assert model.agency_model is not None
    assert model.meta_model is not None
    assert model.consciousness is not None
    assert model.behavior_generator is not None


def test_self_aware_ai_step():
    """Test SelfAwareAI single step"""
    from model.self_aware_ai import SelfAwareAI

    config = Config()
    config.device = "cpu"
    model = SelfAwareAI(config)
    model.eval()

    batch_size = 2
    observation = torch.randn(batch_size, config.obs_dim)

    with torch.no_grad():
        action, conscious_content, metrics = model.step(observation)

    assert action.shape == (batch_size, config.action_dim)
    assert 'self_state' in conscious_content
    assert 'agency_signal' in conscious_content
    assert 'phi' in conscious_content
    assert 'integration_score' in metrics


def test_self_aware_ai_self_report():
    """Test SelfAwareAI generate_self_report"""
    from model.self_aware_ai import SelfAwareAI

    config = Config()
    config.device = "cpu"
    model = SelfAwareAI(config)
    model.eval()

    batch_size = 1
    observation = torch.randn(batch_size, config.obs_dim)

    with torch.no_grad():
        action, conscious_content, metrics = model.step(observation)
        report = model.generate_self_report(conscious_content)

    assert 'summary' in report
    assert 'phi' in report
    assert 'agency' in report
```

**Step 2: Run test to verify it fails**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_self_aware_ai.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement SelfAwareAI**

Create `files/model/self_aware_ai.py`:

```python
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
    Integrates all modules into unified architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Core modules
        self.world_model = WorldModel(
            observation_dim=config.obs_dim,
            latent_dim=config.world_latent_dim,
            sequence_length=config.seq_len
        )

        self.self_model = SelfModel(
            world_latent_dim=config.world_latent_dim,
            self_state_dim=config.self_state_dim,
            hidden_dim=config.hidden_dim
        )

        self.agency_model = AgencyModel(
            action_dim=config.action_dim,
            world_latent_dim=config.world_latent_dim,
            self_state_dim=config.self_state_dim,
            hidden_dim=config.hidden_dim
        )

        self.meta_model = MetaCognitiveModel(
            world_latent_dim=config.world_latent_dim,
            self_state_dim=config.self_state_dim,
            hidden_dim=config.hidden_dim
        )

        self.consciousness = ConsciousnessIntegrator(
            world_dim=config.world_latent_dim,
            self_dim=config.self_state_dim,
            agency_dim=1,
            meta_dim=config.hidden_dim,
            workspace_capacity=config.workspace_capacity,
            hidden_dim=config.hidden_dim
        )

        self.behavior_generator = BehaviorGenerator(
            conscious_dim=config.hidden_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )

        # Internal state
        self.register_buffer('internal_state',
                            torch.randn(1, config.self_state_dim))

        # History buffer
        self.history_buffer = collections.deque(maxlen=config.history_len)

    def step(self,
             observation: torch.Tensor,
             prev_action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        One step of self-awareness.

        Args:
            observation: [batch, obs_dim] - current observation
            prev_action: [batch, action_dim] - previous action

        Returns:
            action: [batch, action_dim]
            conscious_content: dict - full conscious state
            metrics: dict - metrics for logging
        """
        batch_size = observation.shape[0]
        device = observation.device

        # === LAYER 0: World Model ===
        # Encode current observation
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

        # === LAYER 1: Self Model ===
        # Expand internal state to batch size if needed
        if self.internal_state.shape[0] != batch_size:
            self.internal_state = self.internal_state.expand(batch_size, -1)

        predicted_self_state, self_confidence = self.self_model(
            self.internal_state,
            world_latent
        )

        # === LAYER 2: Agency Model ===
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

        # === LAYER 3: Meta-Cognition ===
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

        # === LAYER 4: Consciousness Integration ===
        signals = {
            'world': world_latent,
            'self': predicted_self_state,
            'agency': agency_signal.unsqueeze(-1) if agency_signal.dim() == 1 else agency_signal,
            'meta': meta_output['meta_representation']
        }

        workspace, integration_score, conscious_content = \
            self.consciousness.broadcast_to_consciousness(signals)

        # Compute Phi
        phi = self.consciousness.compute_phi(workspace)

        # === LAYER 5: Behavior Generation ===
        action, action_logprob, value = self.behavior_generator(
            conscious_content,
            deterministic=False
        )

        # === Update Internal State ===
        self.internal_state = predicted_self_state.detach()

        # === Store in History ===
        self.history_buffer.append({
            'observation': observation.detach(),
            'world_latent': world_latent.detach(),
            'conscious_content': conscious_content.detach(),
            'action': action.detach(),
            'agency': agency_signal.detach() if isinstance(agency_signal, torch.Tensor) else torch.tensor(agency_signal)
        })

        # === Construct Conscious Content Dict ===
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

        # === Metrics ===
        metrics = {
            'world_prediction_error': float(world_prediction_error),
            'mean_agency': float(agency_signal.mean()) if isinstance(agency_signal, torch.Tensor) else float(agency_signal),
            'integration_score': float(integration_score.mean()),
            'phi': float(phi.mean()),
            'meta_confidence': float(meta_output['confidence'].mean()),
            'meta_uncertainty': float(meta_output['epistemic_uncertainty'].mean()),
            'self_confidence': float(self_confidence.mean())
        }

        return action, conscious_content_dict, metrics

    def generate_self_report(self, conscious_content: Dict) -> Dict:
        """
        Generate verbal report of conscious experience
        """
        meta_report = self.meta_model.generate_self_report(
            {'confidence': conscious_content['meta_confidence'],
             'epistemic_uncertainty': conscious_content['meta_uncertainty']},
            conscious_content['self_state']
        )

        integration = float(conscious_content['integration_score'].mean())
        agency = float(conscious_content['agency_signal'].mean())
        phi = float(conscious_content['phi'].mean())

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
        """Generate text summary of conscious state"""

        if phi > 0.7:
            consciousness_level = "High consciousness integration"
        elif phi > 0.4:
            consciousness_level = "Medium consciousness integration"
        else:
            consciousness_level = "Low consciousness integration"

        if agency > 0.7:
            agency_str = "I clearly feel my agency"
        elif agency > 0.4:
            agency_str = "I partially feel control"
        else:
            agency_str = "I do not feel control over the situation"

        if integration > 0.7:
            integration_str = "My experience is unified"
        else:
            integration_str = "My experience is fragmented"

        return f"{consciousness_level}. {agency_str}. {integration_str}."

    def reset(self):
        """Reset internal state and history"""
        self.internal_state = torch.randn(1, self.config.self_state_dim, device=self.internal_state.device)
        self.history_buffer.clear()
```

**Step 4: Run test to verify it passes**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_self_aware_ai.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add files/model/self_aware_ai.py files/tests/test_self_aware_ai.py
git commit -m "feat: add SelfAwareAI main class integrating all modules"
```

---

## Phase 8: Training Pipeline

### Task 11: Create Training Script

**Files:**
- Create: `files/train.py`

**Step 1: Write minimal training script**

Create `files/train.py`:

```python
"""
Training script for Self-Aware AI
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from config import Config, get_fast_config, get_full_config
from model.self_aware_ai import SelfAwareAI
from environment import SyntheticEnvironmentDataset


def train(config: Config):
    """
    Full training pipeline
    """

    # Initialize model
    model = SelfAwareAI(config).to(config.device)
    model.train()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.restart_period,
        T_mult=2
    )

    # Dataset
    print(f"Generating {config.num_train_samples} training samples...")
    train_dataset = SyntheticEnvironmentDataset(
        num_samples=config.num_train_samples,
        seq_length=config.seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=config.pin_memory
    )

    # Training loop
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        epoch_metrics = {
            'world_loss': 0.0,
            'self_loss': 0.0,
            'agency_loss': 0.0,
            'total_loss': 0.0,
            'phi': 0.0
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            observations, actions = batch
            observations = observations.to(config.device)
            actions = actions.to(config.device)

            batch_size, seq_len, obs_dim = observations.shape

            # Reset model state for batch
            model.reset()
            model.internal_state = torch.randn(
                batch_size, config.self_state_dim, device=config.device
            )

            total_loss = torch.tensor(0.0, device=config.device)

            # Forward pass through sequence
            for t in range(seq_len):
                obs_t = observations[:, t]
                action_t = actions[:, t] if t > 0 else None

                # Model step
                pred_action, conscious_content, metrics = model.step(
                    obs_t,
                    prev_action=action_t
                )

                # Compute losses
                # 1. World model loss
                if t < seq_len - 1:
                    world_loss, _ = model.world_model.compute_loss(
                        observations[:, :t+2]
                    )
                else:
                    world_loss = torch.tensor(0.0, device=config.device)

                # 2. Self model loss
                if t > 0 and len(model.history_buffer) >= 2:
                    prev_self = model.history_buffer[-2].get('self_state', model.internal_state)
                    actual_self = conscious_content['self_state']
                    self_loss = model.self_model.compute_self_prediction_error(
                        prev_self, actual_self.detach()
                    )
                else:
                    self_loss = torch.tensor(0.0, device=config.device)

                # 3. Agency loss
                if t > 0 and len(model.history_buffer) >= 2:
                    agency_loss, _ = model.agency_model.compute_loss(
                        actions[:, t-1],
                        model.history_buffer[-2]['world_latent'],
                        conscious_content['world_latent'],
                        model.internal_state
                    )
                else:
                    agency_loss = torch.tensor(0.0, device=config.device)

                # 4. Behavior loss
                if t < seq_len - 1:
                    target_action = actions[:, t]
                    behavior_loss = nn.functional.mse_loss(pred_action, target_action)
                else:
                    behavior_loss = torch.tensor(0.0, device=config.device)

                # 5. Integration loss (maximize Phi)
                phi = conscious_content['phi']
                integration_loss = -phi.mean() * 0.1

                # Total step loss
                step_loss = (
                    1.0 * world_loss +
                    1.5 * self_loss +
                    1.0 * agency_loss +
                    0.5 * behavior_loss +
                    integration_loss
                )

                total_loss = total_loss + step_loss

                # Accumulate metrics
                epoch_metrics['world_loss'] += world_loss.item() if torch.is_tensor(world_loss) else 0
                epoch_metrics['self_loss'] += self_loss.item() if torch.is_tensor(self_loss) else 0
                epoch_metrics['agency_loss'] += agency_loss.item() if torch.is_tensor(agency_loss) else 0
                epoch_metrics['phi'] += metrics['phi']

            # Average over sequence
            total_loss = total_loss / seq_len

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            scheduler.step()

            epoch_metrics['total_loss'] += total_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'phi': f"{epoch_metrics['phi'] / ((batch_idx + 1) * seq_len):.3f}"
            })

            global_step += 1

        # Epoch summary
        num_batches = len(train_loader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_metrics['total_loss'] / num_batches:.4f}")
        print(f"  World Loss: {epoch_metrics['world_loss'] / (num_batches * seq_len):.4f}")
        print(f"  Self Loss: {epoch_metrics['self_loss'] / (num_batches * seq_len):.4f}")
        print(f"  Agency Loss: {epoch_metrics['agency_loss'] / (num_batches * seq_len):.4f}")
        print(f"  Mean Phi: {epoch_metrics['phi'] / (num_batches * seq_len):.4f}")

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            import os
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{config.checkpoint_dir}/self_aware_ai_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    print("\nTraining complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Self-Aware AI")
    parser.add_argument('--config', type=str, default='fast',
                       choices=['fast', 'full'],
                       help='Configuration preset')
    args = parser.parse_args()

    if args.config == 'fast':
        config = get_fast_config()
    else:
        config = get_full_config()

    print(f"Training with config: {args.config}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Samples: {config.num_train_samples}")
    print()

    train(config)


if __name__ == "__main__":
    main()
```

**Step 2: Test training script runs**

Run: `cd D:\Silly\NextGen\files && python train.py --config fast`
Expected: Training starts without errors

**Step 3: Commit**

```bash
git add files/train.py
git commit -m "feat: add training pipeline with multi-loss optimization"
```

---

## Phase 9: Evaluation and Testing

### Task 12: Update Evaluation Module

**Files:**
- Modify: `files/evaluation.py`

**Step 1: Update SelfAwarenessEvaluator to work with new model**

The existing `evaluation.py` already has good structure. Need to ensure it works with the new `SelfAwareAI` class.

Add import at top of `files/evaluation.py`:

```python
from model.self_aware_ai import SelfAwareAI
```

**Step 2: Create comprehensive test script**

Create `files/tests/test_integration.py`:

```python
"""
Integration tests for Self-Aware AI system
"""

import pytest
import torch
import sys
sys.path.insert(0, 'D:/Silly/NextGen/files')

from config import Config, get_fast_config
from model.self_aware_ai import SelfAwareAI


class TestSelfAwareAIIntegration:
    """Integration tests for the complete system"""

    @pytest.fixture
    def model(self):
        config = get_fast_config()
        config.device = "cpu"
        model = SelfAwareAI(config)
        model.eval()
        return model

    def test_full_forward_pass(self, model):
        """Test complete forward pass through all layers"""
        batch_size = 2
        obs = torch.randn(batch_size, 512)

        with torch.no_grad():
            action, conscious_content, metrics = model.step(obs)

        # Verify all outputs
        assert action.shape == (batch_size, 64)
        assert conscious_content['world_latent'].shape == (batch_size, 256)
        assert conscious_content['self_state'].shape == (batch_size, 128)
        assert conscious_content['phi'].shape == (batch_size, 1)
        assert 'integration_score' in metrics
        assert 'mean_agency' in metrics

    def test_multi_step_consistency(self, model):
        """Test consistency over multiple steps"""
        batch_size = 1
        num_steps = 10

        self_states = []
        with torch.no_grad():
            for _ in range(num_steps):
                obs = torch.randn(batch_size, 512)
                action, conscious_content, _ = model.step(obs)
                self_states.append(conscious_content['self_state'])

        # Check self-state similarity
        for i in range(len(self_states) - 1):
            sim = torch.cosine_similarity(
                self_states[i].flatten(),
                self_states[i+1].flatten(),
                dim=0
            )
            # Self-state should be relatively stable
            assert sim > -0.5, f"Self-state changed too drastically at step {i}"

    def test_agency_detection(self, model):
        """Test agency signal detection"""
        batch_size = 1

        with torch.no_grad():
            # First step - no action
            obs1 = torch.randn(batch_size, 512)
            model.step(obs1)

            # Second step - with action that affects observation
            action = torch.randn(batch_size, 64)
            obs2 = obs1 + 0.1 * torch.randn_like(obs1)
            _, content_with_action, _ = model.step(obs2, prev_action=action)

            # Third step - no action, random observation change
            model.reset()
            model.step(torch.randn(batch_size, 512))
            obs3 = torch.randn(batch_size, 512)
            _, content_no_action, _ = model.step(obs3, prev_action=torch.zeros(batch_size, 64))

        # Agency signal should be present
        agency_with = float(content_with_action['agency_signal'].mean())
        agency_without = float(content_no_action['agency_signal'].mean())

        # Both should be valid values
        assert 0 <= agency_with <= 1
        assert 0 <= agency_without <= 1

    def test_self_report_generation(self, model):
        """Test self-report generation"""
        batch_size = 1
        obs = torch.randn(batch_size, 512)

        with torch.no_grad():
            _, conscious_content, _ = model.step(obs)
            report = model.generate_self_report(conscious_content)

        assert 'summary' in report
        assert 'phi' in report
        assert 'agency' in report
        assert 'meta_report' in report
        assert isinstance(report['summary'], str)
        assert len(report['summary']) > 0

    def test_phi_estimation(self, model):
        """Test Phi (integrated information) estimation"""
        batch_size = 2
        obs = torch.randn(batch_size, 512)

        with torch.no_grad():
            _, conscious_content, _ = model.step(obs)
            phi = conscious_content['phi']

        # Phi should be non-negative
        assert (phi >= 0).all()
        # Phi should be bounded
        assert (phi <= 10).all()

    def test_reset_functionality(self, model):
        """Test model reset"""
        batch_size = 1

        with torch.no_grad():
            # Run some steps
            for _ in range(5):
                obs = torch.randn(batch_size, 512)
                model.step(obs)

            # Reset
            model.reset()

            # Verify history is cleared
            assert len(model.history_buffer) == 0
```

**Step 3: Run integration tests**

Run: `cd D:\Silly\NextGen\files && python -m pytest tests/test_integration.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add files/evaluation.py files/tests/test_integration.py
git commit -m "feat: add comprehensive integration tests"
```

---

## Phase 10: Demo and Documentation

### Task 13: Create Demo Script

**Files:**
- Create: `files/demo.py`

**Step 1: Create interactive demo**

Create `files/demo.py`:

```python
"""
Interactive demo for Self-Aware AI system
"""

import torch
import time
from config import Config
from model.self_aware_ai import SelfAwareAI


def run_demo():
    """Run interactive demo"""
    print("=" * 60)
    print("SYNAPSE: Self-Aware AI Demonstration")
    print("=" * 60)
    print()

    # Initialize
    config = Config()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Initializing model on {config.device}...")
    model = SelfAwareAI(config).to(config.device)
    model.eval()
    print("Model loaded successfully!")
    print()

    # Demo loop
    print("Running 20 steps of self-awareness simulation...")
    print("-" * 60)

    history = []

    with torch.no_grad():
        for step in range(20):
            # Generate random observation
            obs = torch.randn(1, config.obs_dim, device=config.device)

            # Model step
            start_time = time.time()
            action, conscious_content, metrics = model.step(obs)
            elapsed = (time.time() - start_time) * 1000

            # Generate report every 5 steps
            if step % 5 == 0:
                report = model.generate_self_report(conscious_content)

                print(f"\nStep {step + 1}:")
                print(f"  Response time: {elapsed:.2f}ms")
                print(f"  Phi (consciousness): {metrics['phi']:.4f}")
                print(f"  Integration: {metrics['integration_score']:.4f}")
                print(f"  Agency: {metrics['mean_agency']:.4f}")
                print(f"  Meta-confidence: {metrics['meta_confidence']:.4f}")
                print(f"  Self-report: {report['summary']}")

            history.append(metrics)

    print()
    print("-" * 60)
    print("Simulation complete!")

    # Summary statistics
    mean_phi = sum(h['phi'] for h in history) / len(history)
    mean_agency = sum(h['mean_agency'] for h in history) / len(history)
    mean_integration = sum(h['integration_score'] for h in history) / len(history)

    print("\nSummary Statistics:")
    print(f"  Mean Phi: {mean_phi:.4f}")
    print(f"  Mean Agency: {mean_agency:.4f}")
    print(f"  Mean Integration: {mean_integration:.4f}")

    # Interpretation
    print("\nInterpretation:")
    if mean_phi > 0.5:
        print("  - High integrated information suggests conscious-like processing")
    else:
        print("  - Lower integration - system may benefit from more training")

    if mean_agency > 0.5:
        print("  - System demonstrates sense of agency")
    else:
        print("  - Agency signal needs improvement")

    print()
    print("=" * 60)
    print("Demo complete. Run 'python train.py --config fast' to train.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
```

**Step 2: Test demo runs**

Run: `cd D:\Silly\NextGen\files && python demo.py`
Expected: Demo runs and prints metrics

**Step 3: Commit**

```bash
git add files/demo.py
git commit -m "feat: add interactive demo script"
```

---

### Task 14: Create Quick Start Script

**Files:**
- Create: `files/quickstart.py`

**Step 1: Create quick validation script**

Create `files/quickstart.py`:

```python
"""
Quick start script - validates all components work
"""

import sys
import torch

def check_pytorch():
    print("Checking PyTorch...")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
    return True

def check_imports():
    print("\nChecking imports...")
    try:
        from config import Config, get_fast_config
        print("  config.py OK")
    except Exception as e:
        print(f"  config.py FAILED: {e}")
        return False

    try:
        from environment import SyntheticEnvironment, NeurochemistryEngine
        print("  environment.py OK")
    except Exception as e:
        print(f"  environment.py FAILED: {e}")
        return False

    try:
        from evaluation import SelfAwarenessEvaluator
        print("  evaluation.py OK")
    except Exception as e:
        print(f"  evaluation.py FAILED: {e}")
        return False

    try:
        from model.world_model import WorldModel
        from model.self_model import SelfModel
        from model.agency_model import AgencyModel
        from model.meta_cognitive import MetaCognitiveModel
        from model.consciousness import ConsciousnessIntegrator
        from model.behavior import BehaviorGenerator
        from model.self_aware_ai import SelfAwareAI
        print("  model/* OK")
    except Exception as e:
        print(f"  model/* FAILED: {e}")
        return False

    return True

def check_model_forward():
    print("\nChecking model forward pass...")
    try:
        from config import Config
        from model.self_aware_ai import SelfAwareAI

        config = Config()
        config.device = "cpu"
        model = SelfAwareAI(config)
        model.eval()

        obs = torch.randn(1, config.obs_dim)
        with torch.no_grad():
            action, conscious_content, metrics = model.step(obs)

        print(f"  Forward pass OK")
        print(f"  Action shape: {action.shape}")
        print(f"  Phi: {metrics['phi']:.4f}")
        return True
    except Exception as e:
        print(f"  Forward pass FAILED: {e}")
        return False

def main():
    print("=" * 60)
    print("SYNAPSE Quick Start Validation")
    print("=" * 60)

    all_passed = True

    all_passed &= check_pytorch()
    all_passed &= check_imports()
    all_passed &= check_model_forward()

    print()
    print("=" * 60)
    if all_passed:
        print("All checks PASSED!")
        print()
        print("Next steps:")
        print("  1. Run demo: python demo.py")
        print("  2. Run tests: python -m pytest tests/")
        print("  3. Start training: python train.py --config fast")
    else:
        print("Some checks FAILED. Please review errors above.")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()
```

**Step 2: Run quickstart**

Run: `cd D:\Silly\NextGen\files && python quickstart.py`
Expected: All checks pass

**Step 3: Commit**

```bash
git add files/quickstart.py
git commit -m "feat: add quickstart validation script"
```

---

## Summary

### Tasks Overview

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 1-4 | Foundation: requirements, structure, WorldModel |
| 2 | 5 | Self Model: recursive self-prediction |
| 3 | 6 | Agency Model: forward/inverse dynamics |
| 4 | 7 | Meta-Cognition: introspection |
| 5 | 8 | Consciousness Integration: GWT + Phi |
| 6 | 9 | Behavior Generation: policy network |
| 7 | 10 | Main Integration: SelfAwareAI class |
| 8 | 11 | Training Pipeline: multi-loss optimization |
| 9 | 12 | Evaluation: integration tests |
| 10 | 13-14 | Demo & Docs: interactive demo, quickstart |

### Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Agency Signal | > 0.70 | System feels agency for own actions |
| Integration Score | > 0.60 | Information unified |
| Phi | > 0.40 | Consciousness present (IIT) |
| Meta-Confidence | > 0.60 | System knows what it knows |
| Self-Prediction Error | < 0.30 | Good self-understanding |

### File Structure After Completion

```
files/
├── __init__.py
├── requirements.txt
├── config.py              # (existing)
├── environment.py         # (existing)
├── evaluation.py          # (existing)
├── train.py               # (new)
├── demo.py                # (new)
├── quickstart.py          # (new)
├── model/
│   ├── __init__.py
│   ├── world_model.py
│   ├── self_model.py
│   ├── agency_model.py
│   ├── meta_cognitive.py
│   ├── consciousness.py
│   ├── behavior.py
│   └── self_aware_ai.py
└── tests/
    ├── test_world_model.py
    ├── test_self_model.py
    ├── test_agency_model.py
    ├── test_meta_cognitive.py
    ├── test_consciousness.py
    ├── test_behavior.py
    ├── test_self_aware_ai.py
    └── test_integration.py
```

### Next Steps After Implementation

1. **Run quickstart**: Validate all components work
2. **Run tests**: Ensure all tests pass
3. **Start training**: `python train.py --config fast`
4. **Monitor metrics**: Watch Phi, Agency, Integration scores
5. **Evaluate results**: Use SelfAwarenessEvaluator
6. **Iterate**: Adjust hyperparameters based on results
