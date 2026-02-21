# environment.py
"""
Synthetic Environment для обучения Self-Aware AI
+ Neurochemistry Engine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
try:
    from .config import EnvironmentConfig, NeurochemistryConfig
except ImportError:
    from config import EnvironmentConfig, NeurochemistryConfig


class NeurochemistryEngine:
    """
    Симуляция нейрохимии внутреннего состояния системы.
    
    32-мерный вектор нейротрансмиттеров, которые:
    - Имеют baseline levels (гомеостаз)
    - Decay к baseline
    - Interact друг с другом
    - Влияют на эмоции и поведение
    """
    
    def __init__(self, config: NeurochemistryConfig):
        self.config = config
        self.num_transmitters = 32
        
        # Initialize state
        self.state = torch.zeros(self.num_transmitters)
        
        # Set baselines
        self.baseline = torch.zeros(self.num_transmitters)
        self.decay_rates = torch.zeros(self.num_transmitters)
        
        for i, name in enumerate(config.neurotransmitters[:self.num_transmitters]):
            if name in config.baseline_levels:
                self.baseline[i] = config.baseline_levels[name]
            else:
                self.baseline[i] = 0.5  # Default
            
            if name in config.decay_rates:
                self.decay_rates[i] = config.decay_rates[name]
            else:
                self.decay_rates[i] = 0.1  # Default
        
        # Initialize to baseline
        self.state = self.baseline.clone()
        
        # Interaction matrix (simplified)
        self.interaction_matrix = torch.zeros(self.num_transmitters, self.num_transmitters)
        
        # Common indices for quick access
        self.dopamine_idx = 0
        self.serotonin_idx = 1
        self.oxytocin_idx = 2
        self.cortisol_idx = 3
        self.norepinephrine_idx = 4
        
    def step(self, external_stimulus: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Один шаг динамики нейрохимии
        
        Args:
            external_stimulus: [num_transmitters] — внешнее влияние
        
        Returns:
            state: [num_transmitters] — текущее состояние
        """
        # Decay к baseline
        self.state = self.state + self.decay_rates * (self.baseline - self.state)
        
        # Apply external stimulus
        if external_stimulus is not None:
            self.state = self.state + 0.1 * external_stimulus
        
        # Interactions (simplified)
        # Dopamine inhibits serotonin
        dopamine_effect = -0.1 * self.state[self.dopamine_idx]
        self.state[self.serotonin_idx] += dopamine_effect
        
        # Cortisol inhibits oxytocin
        cortisol_effect = -0.2 * self.state[self.cortisol_idx]
        self.state[self.oxytocin_idx] += cortisol_effect
        
        # Norepinephrine increases cortisol
        norepinephrine_effect = 0.15 * self.state[self.norepinephrine_idx]
        self.state[self.cortisol_idx] += norepinephrine_effect
        
        # Clamp to [0, 1]
        self.state = torch.clamp(self.state, 0.0, 1.0)
        
        return self.state.clone()
    
    def trigger_emotion(self, emotion: str, intensity: float = 0.5):
        """
        Триггер эмоции через изменение нейрохимии
        
        Args:
            emotion: 'happy', 'sad', 'anxious', 'calm', 'loving', 'angry'
            intensity: 0.0 to 1.0
        """
        if emotion == 'happy':
            self.state[self.dopamine_idx] += intensity * 0.3
            self.state[self.serotonin_idx] += intensity * 0.2
        
        elif emotion == 'sad':
            self.state[self.serotonin_idx] -= intensity * 0.3
            self.state[self.dopamine_idx] -= intensity * 0.2
        
        elif emotion == 'anxious':
            self.state[self.cortisol_idx] += intensity * 0.4
            self.state[self.norepinephrine_idx] += intensity * 0.3
        
        elif emotion == 'calm':
            self.state[self.cortisol_idx] -= intensity * 0.3
            self.state[5] += intensity * 0.2  # GABA (calming)
        
        elif emotion == 'loving':
            self.state[self.oxytocin_idx] += intensity * 0.5
            self.state[self.dopamine_idx] += intensity * 0.2
        
        elif emotion == 'angry':
            self.state[self.norepinephrine_idx] += intensity * 0.4
            self.state[self.cortisol_idx] += intensity * 0.3
        
        # Clamp
        self.state = torch.clamp(self.state, 0.0, 1.0)
    
    def get_emotional_state(self) -> Dict[str, float]:
        """
        Интерпретация нейрохимии как эмоциональное состояние
        """
        dopamine = float(self.state[self.dopamine_idx])
        serotonin = float(self.state[self.serotonin_idx])
        oxytocin = float(self.state[self.oxytocin_idx])
        cortisol = float(self.state[self.cortisol_idx])
        norepinephrine = float(self.state[self.norepinephrine_idx])
        
        # Compute emotion dimensions
        valence = dopamine + serotonin - cortisol  # positive vs negative
        arousal = norepinephrine + cortisol  # activated vs deactivated
        social = oxytocin  # social bonding
        
        return {
            'valence': np.clip(valence, -1, 1),
            'arousal': np.clip(arousal, 0, 1),
            'social': np.clip(social, 0, 1),
            'dopamine': dopamine,
            'serotonin': serotonin,
            'oxytocin': oxytocin,
            'cortisol': cortisol,
            'norepinephrine': norepinephrine
        }
    
    def reset(self):
        """Reset к baseline"""
        self.state = self.baseline.clone()


class SyntheticEnvironment:
    """
    Synthetic environment для обучения самосознательной системы.
    
    Цели:
    1. Создать observations, которые система может предсказывать
    2. Позволить системе выполнять actions, которые влияют на мир
    3. Иметь как controllable (агентные), так и uncontrollable (внешние) изменения
    4. Быть достаточно сложным, чтобы требовать интеграции информации
    """
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        
        # State representation (latent world state)
        self.state_dim = config.state_dim
        self.state = torch.randn(self.state_dim)
        
        # Dynamics model (learnable or fixed)
        self.dynamics = nn.Sequential(
            nn.Linear(self.state_dim + config.action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.state_dim)
        )
        
        # External dynamics (random changes not caused by agent)
        self.external_noise_scale = 0.1
        
        # Observation encoder (state -> observation)
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, config.observation_dim)
        )
        
        # Objects in environment
        self.num_objects = config.num_objects
        self.object_states = torch.randn(self.num_objects, 64)
        
        # Agency tracking (which changes were caused by agent)
        self.last_action = None
        self.action_effect = None
        
        self.step_count = 0
    
    def reset(self) -> torch.Tensor:
        """
        Reset environment
        
        Returns:
            observation: [obs_dim]
        """
        self.state = torch.randn(self.state_dim) * 0.5
        self.object_states = torch.randn(self.num_objects, 64)
        self.last_action = None
        self.step_count = 0
        
        return self._get_observation()
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Execute action and return next observation
        
        Args:
            action: [action_dim]
        
        Returns:
            observation: [obs_dim]
            reward: float
            done: bool
            info: dict
        """
        batch_mode = len(action.shape) > 1
        
        if not batch_mode:
            action = action.unsqueeze(0)
            prev_state = self.state.unsqueeze(0)
        else:
            prev_state = self.state.expand(action.shape[0], -1)
        
        # 1. Agent-caused dynamics
        state_action = torch.cat([prev_state, action], dim=-1)
        predicted_next_state = self.dynamics(state_action)
        
        # 2. External dynamics (random)
        external_change = torch.randn_like(predicted_next_state) * self.external_noise_scale
        
        # 3. Combine (weighted by agency_ratio)
        agent_weight = self.config.agency_ratio
        external_weight = 1.0 - agent_weight
        
        next_state = (agent_weight * predicted_next_state + 
                     external_weight * (prev_state + external_change))
        
        # Add transition noise
        next_state = next_state + torch.randn_like(next_state) * self.config.transition_stochasticity
        
        # Update state
        if not batch_mode:
            self.state = next_state.squeeze(0)
        else:
            self.state = next_state[0]  # Take first batch element as new state
        
        # Track action effect
        self.action_effect = predicted_next_state - prev_state
        self.last_action = action
        
        # Get observation
        observation = self._get_observation()
        
        # Compute reward (if using RL)
        reward = 0.0
        if self.config.use_rewards:
            # Simple reward: minimize prediction error
            reward = -torch.norm(next_state - predicted_next_state).item()
        
        # Check if episode done
        self.step_count += 1
        done = self.step_count >= self.config.episode_length
        
        # Info
        info = {
            'agent_caused_change': float(torch.norm(self.action_effect).item()),
            'external_change': float(torch.norm(external_change).item()),
            'agency_ratio_actual': agent_weight
        }
        
        if not batch_mode:
            observation = observation.squeeze(0)
        
        return observation, reward, done, info
    
    def _get_observation(self) -> torch.Tensor:
        """
        Generate observation from current state
        """
        # Encode state to observation
        obs = self.encoder(self.state.unsqueeze(0) if len(self.state.shape) == 1 else self.state)
        
        # Add observation noise
        obs = obs + torch.randn_like(obs) * self.config.observation_noise
        
        return obs.squeeze(0) if obs.shape[0] == 1 else obs
    
    def get_true_agency_signal(self) -> float:
        """
        Ground truth agency signal (for evaluation)
        
        Returns fraction of change caused by agent vs external
        """
        if self.action_effect is None:
            return 0.0
        
        agent_magnitude = float(torch.norm(self.action_effect).item())
        total_change = agent_magnitude + self.external_noise_scale
        
        return agent_magnitude / (total_change + 1e-8)


class SyntheticEnvironmentDataset(torch.utils.data.Dataset):
    """
    Dataset of trajectories from synthetic environment (memory-limited)

    WARNING: For large num_samples, use LazySyntheticDataset instead
    to avoid OOM issues.
    """

    def __init__(self,
                 num_samples: int,
                 seq_length: int,
                 config: Optional[EnvironmentConfig] = None):

        self.num_samples = num_samples
        self.seq_length = seq_length

        if config is None:
            config = EnvironmentConfig()

        self.env = SyntheticEnvironment(config)

        # Pre-generate data
        print(f"Generating {num_samples} trajectories...")
        self.trajectories = []

        for i in range(num_samples):
            if i % 1000 == 0:
                print(f"Generated {i}/{num_samples}")

            trajectory = self._generate_trajectory()
            self.trajectories.append(trajectory)

    def _generate_trajectory(self) -> Dict[str, torch.Tensor]:
        """
        Generate one trajectory
        """
        observations = []
        actions = []

        obs = self.env.reset()
        observations.append(obs)

        for t in range(self.seq_length):
            # Random action
            action = torch.randn(self.env.config.action_dim)

            # Step
            next_obs, _, _, _ = self.env.step(action)

            observations.append(next_obs)
            actions.append(action)

        observations = torch.stack(observations[:-1])  # Exclude last
        actions = torch.stack(actions)

        return {
            'observations': observations,
            'actions': actions
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        return traj['observations'], traj['actions']


class LazySyntheticDataset(torch.utils.data.IterableDataset):
    """
    Memory-efficient streaming dataset - generates trajectories on-the-fly.

    This solves OOM issues by never storing all data in memory.
    Generates infinite stream of trajectories for continuous training.

    Usage:
        dataset = LazySyntheticDataset(seq_length=32)
        loader = DataLoader(dataset, batch_size=32)
        for observations, actions in loader:
            # Process batch - data generated on demand
    """

    def __init__(self,
                 seq_length: int,
                 config: Optional[EnvironmentConfig] = None,
                 prefetch_trajectories: int = 10):
        """
        Args:
            seq_length: Length of each trajectory
            config: Environment configuration
            prefetch_trajectories: Number of trajectories to prefetch (small buffer)
        """
        self.seq_length = seq_length
        self.config = config or EnvironmentConfig()
        self.prefetch_count = prefetch_trajectories

        # Thread-local storage for workers
        self._env = None

    def _get_env(self) -> SyntheticEnvironment:
        """Get or create environment (one per worker)"""
        if self._env is None:
            self._env = SyntheticEnvironment(self.config)
        return self._env

    def _generate_trajectory(self, env: SyntheticEnvironment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate one trajectory on-the-fly
        """
        observations = []
        actions = []

        obs = env.reset()
        observations.append(obs)

        for t in range(self.seq_length):
            action = torch.randn(env.config.action_dim)
            next_obs, _, _, _ = env.step(action)

            observations.append(next_obs)
            actions.append(action)

        observations = torch.stack(observations[:-1])
        actions = torch.stack(actions)

        return observations, actions

    def __iter__(self):
        """Infinite iterator - generates data on demand"""
        env = self._get_env()

        while True:
            yield self._generate_trajectory(env)


class BufferedLazyDataset(torch.utils.data.IterableDataset):
    """
    Buffered version of LazySyntheticDataset with small prefetch buffer.

    Balances memory efficiency with training speed by maintaining
    a small buffer of pre-generated trajectories.
    """

    def __init__(self,
                 seq_length: int,
                 config: Optional[EnvironmentConfig] = None,
                 buffer_size: int = 100):
        """
        Args:
            seq_length: Length of each trajectory
            config: Environment configuration
            buffer_size: Number of trajectories in buffer (default: 100 = ~7MB)
        """
        self.seq_length = seq_length
        self.config = config or EnvironmentConfig()
        self.buffer_size = buffer_size

        self._env = None
        self._buffer = []
        self._buffer_idx = 0

    def _get_env(self) -> SyntheticEnvironment:
        if self._env is None:
            self._env = SyntheticEnvironment(self.config)
        return self._env

    def _generate_trajectory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        env = self._get_env()
        observations = []
        actions = []

        obs = env.reset()
        observations.append(obs)

        for t in range(self.seq_length):
            action = torch.randn(env.config.action_dim)
            next_obs, _, _, _ = env.step(action)
            observations.append(next_obs)
            actions.append(action)

        return torch.stack(observations[:-1]), torch.stack(actions)

    def _refill_buffer(self):
        """Refill buffer when empty"""
        self._buffer = []
        for _ in range(self.buffer_size):
            self._buffer.append(self._generate_trajectory())
        self._buffer_idx = 0

    def __iter__(self):
        while True:
            if self._buffer_idx >= len(self._buffer):
                self._refill_buffer()

            yield self._buffer[self._buffer_idx]
            self._buffer_idx += 1


# ============ Utility Functions ============

def create_sensorimotor_loop_env(obs_dim=512, action_dim=64):
    """
    Create a simple sensorimotor loop environment
    """
    config = EnvironmentConfig()
    config.observation_dim = obs_dim
    config.action_dim = action_dim
    config.agency_ratio = 0.7
    
    return SyntheticEnvironment(config)


def create_visual_environment(image_size=64):
    """
    Create a visual environment (images as observations)
    """
    config = EnvironmentConfig()
    config.env_type = "visual"
    config.observation_dim = image_size * image_size * 3
    
    return SyntheticEnvironment(config)


def test_environment():
    """
    Test environment functionality
    """
    print("Testing Synthetic Environment...")
    
    env = create_sensorimotor_loop_env()
    
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for t in range(10):
        action = torch.randn(env.config.action_dim)
        obs, reward, done, info = env.step(action)
        
        print(f"Step {t}: obs shape {obs.shape}, "
              f"agent_caused_change {info['agent_caused_change']:.3f}, "
              f"external_change {info['external_change']:.3f}")
    
    print("Environment test passed!")


def test_neurochemistry():
    """
    Test neurochemistry engine
    """
    print("\nTesting Neurochemistry Engine...")
    
    config = NeurochemistryConfig()
    engine = NeurochemistryEngine(config)
    
    print("Initial state:", engine.get_emotional_state())
    
    # Trigger emotions
    engine.trigger_emotion('happy', intensity=0.8)
    print("After 'happy':", engine.get_emotional_state())
    
    # Let it decay
    for _ in range(20):
        engine.step()
    print("After decay:", engine.get_emotional_state())
    
    print("Neurochemistry test passed!")


if __name__ == "__main__":
    test_environment()
    test_neurochemistry()
