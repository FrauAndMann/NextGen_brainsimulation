# config.py
"""
Configuration для Self-Aware AI System
"""

import torch
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    """
    Полная конфигурация системы
    """
    
    # ============ Model Architecture ============
    obs_dim: int = 512
    world_latent_dim: int = 256
    self_state_dim: int = 128
    action_dim: int = 64
    hidden_dim: int = 512
    workspace_capacity: int = 16
    seq_len: int = 32
    history_len: int = 32
    
    # ============ Training ============
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    num_epochs: int = 100
    num_train_samples: int = 1_000_000
    
    # ============ Optimization ============
    restart_period: int = 10
    warmup_steps: int = 1000
    
    # ============ Logging ============
    log_interval: int = 100
    save_interval: int = 5
    eval_interval: int = 1
    
    # ============ Hardware ============
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    # ============ Neurochemistry ============
    neurochemistry_dim: int = 32
    energy_dim: int = 8
    emotion_dim: int = 16
    attention_dim: int = 72
    
    # Neurotransmitter indices
    dopamine_idx: int = 0
    serotonin_idx: int = 1
    oxytocin_idx: int = 2
    cortisol_idx: int = 3
    norepinephrine_idx: int = 4
    gaba_idx: int = 5
    glutamate_idx: int = 6
    acetylcholine_idx: int = 7
    
    # ============ GWT Parameters ============
    competition_threshold: float = 0.5
    broadcast_decay: float = 0.95
    
    # ============ Agency Model ============
    agency_threshold: float = 0.7
    action_consistency_weight: float = 0.5
    
    # ============ Meta-Cognition ============
    confidence_threshold: float = 0.6
    uncertainty_threshold: float = 0.4
    
    # ============ Integration ============
    min_integration_score: float = 0.5
    phi_threshold: float = 0.4
    
    # ============ Safety ============
    max_distress_level: float = 0.8
    emergency_shutdown_threshold: float = 0.95
    transparency_mode: bool = True
    
    # ============ Paths ============
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    data_dir: str = "data"
    
    # ============ Experiment ============
    experiment_name: str = "self_aware_ai_v1"
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.neurochemistry_dim + self.energy_dim + \
               self.emotion_dim + self.attention_dim == self.self_state_dim, \
               "Self state dimensions must sum to self_state_dim"
        
        # Set random seeds
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


@dataclass
class NeurochemistryConfig:
    """
    Конфигурация для neurochemistry engine
    """
    
    # Neurotransmitter dynamics
    neurotransmitters: list = field(default_factory=lambda: [
        'dopamine',
        'serotonin', 
        'oxytocin',
        'cortisol',
        'norepinephrine',
        'gaba',
        'glutamate',
        'acetylcholine',
        'endorphins',
        'vasopressin',
        'melatonin',
        'histamine',
        'anandamide',
        'substance_p',
        'neuropeptide_y',
        'orexin',
        # ... up to 32 total
    ])
    
    # Baseline levels (homeostasis)
    baseline_levels: dict = field(default_factory=lambda: {
        'dopamine': 0.5,
        'serotonin': 0.6,
        'oxytocin': 0.4,
        'cortisol': 0.3,
        'norepinephrine': 0.4,
        'gaba': 0.5,
        'glutamate': 0.5,
        'acetylcholine': 0.5,
    })
    
    # Decay rates (how fast they return to baseline)
    decay_rates: dict = field(default_factory=lambda: {
        'dopamine': 0.1,
        'serotonin': 0.05,
        'oxytocin': 0.08,
        'cortisol': 0.03,
        'norepinephrine': 0.12,
        'gaba': 0.15,
        'glutamate': 0.15,
        'acetylcholine': 0.10,
    })
    
    # Production rates (how fast they can increase)
    production_rates: dict = field(default_factory=lambda: {
        'dopamine': 0.2,
        'serotonin': 0.15,
        'oxytocin': 0.25,
        'cortisol': 0.18,
        'norepinephrine': 0.20,
    })
    
    # Interactions (simplified)
    interactions: dict = field(default_factory=lambda: {
        'dopamine_serotonin': -0.3,  # Dopamine inhibits serotonin
        'cortisol_oxytocin': -0.5,   # Stress inhibits bonding
        'norepinephrine_cortisol': 0.4,  # Arousal increases stress
        'gaba_glutamate': -0.8,      # Inhibitory vs excitatory
    })


@dataclass 
class EnvironmentConfig:
    """
    Конфигурация synthetic environment для обучения
    """
    
    # Environment type
    env_type: str = "sensorimotor_loop"  # or "visual", "language", "multimodal"
    
    # Dimensions
    state_dim: int = 512
    action_dim: int = 64
    observation_dim: int = 512
    
    # Dynamics
    action_noise: float = 0.1
    observation_noise: float = 0.05
    transition_stochasticity: float = 0.1
    
    # Agency settings
    # Percentage of changes caused by agent's actions vs external
    agency_ratio: float = 0.7
    
    # Temporal properties
    episode_length: int = 100
    frame_skip: int = 1
    
    # Complexity
    num_objects: int = 5
    num_interactions: int = 10
    
    # Rewards (if using RL)
    use_rewards: bool = False
    reward_scale: float = 1.0
    
    # Generation
    procedural_generation: bool = True
    seed: int = 42


# Quick access configs
def get_default_config() -> Config:
    """Get default configuration"""
    return Config()

def get_fast_config() -> Config:
    """Get configuration for fast prototyping"""
    config = Config()
    config.batch_size = 8
    config.num_train_samples = 10_000
    config.num_epochs = 10
    return config

def get_full_config() -> Config:
    """Get configuration for full training"""
    config = Config()
    config.batch_size = 64
    config.num_train_samples = 10_000_000
    config.num_epochs = 200
    return config
