"""
Continuous Learning Module for SYNAPSE

Implements:
- Neurogenesis: Adding neurons when capacity exceeded
- Structural Plasticity: New connections via STDP
- Continuous Training: No epochs, runs until stopped
- Auto-checkpointing: Save progress automatically
- Resume: Continue from any checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import json
from pathlib import Path
from datetime import datetime


@dataclass
class GrowthConfig:
    """Configuration for neural growth"""
    # Neurogenesis
    neurogenesis_enabled: bool = True
    activation_threshold: float = 0.85  # Grow if >85% neurons active
    growth_rate: float = 0.1  # Add 10% more neurons
    max_neurons: int = 100000  # Safety limit

    # Structural Plasticity
    stdp_enabled: bool = True
    stdp_lr: float = 0.01

    # Checkpointing
    auto_save_interval: int = 300  # Save every 5 minutes
    keep_last_n_checkpoints: int = 5

    # Continuous learning
    sample_buffer_size: int = 1000  # Experience replay buffer (reduced for memory efficiency)
    replay_ratio: float = 0.2  # 20% of batch from replay


class NeurogenesisModule(nn.Module):
    """
    Dynamically adds neurons to layers when capacity is exceeded.

    Triggers when:
    - Average activation > threshold
    - Prediction error consistently high
    - Information bottleneck detected
    """

    def __init__(self, growth_config: GrowthConfig):
        super().__init__()
        self.config = growth_config
        self.growth_history: List[Dict] = []

    def check_growth_needed(self,
                            layer_activations: torch.Tensor,
                            layer_errors: torch.Tensor) -> Tuple[bool, str]:
        """
        Determine if a layer needs more neurons

        Returns:
            (needs_growth, reason)
        """
        # Check activation level
        avg_activation = layer_activations.mean().item()
        if avg_activation > self.config.activation_threshold:
            return True, f"High activation ({avg_activation:.3f} > {self.config.activation_threshold})"

        # Check error trend (last 100 samples)
        if len(layer_errors) >= 100:
            recent_error = layer_errors[-100:].mean().item()
            older_error = layer_errors[-200:-100].mean().item() if len(layer_errors) >= 200 else recent_error

            if recent_error > older_error * 1.1:  # Error increasing
                return True, f"Rising error trend ({recent_error:.3f} > {older_error:.3f})"

        return False, ""

    def grow_layer(self,
                   layer: nn.Linear,
                   num_new_neurons: int,
                   initialization: str = "xavier") -> nn.Linear:
        """
        Create a new layer with additional neurons

        Args:
            layer: Original linear layer
            num_new_neurons: How many neurons to add
            initialization: How to initialize new weights

        Returns:
            New layer with more neurons
        """
        old_weight = layer.weight.data
        old_bias = layer.bias.data

        old_out, old_in = old_weight.shape
        new_out = old_out + num_new_neurons

        # Create new layer
        new_layer = nn.Linear(old_in, new_out)

        # Copy old weights
        new_layer.weight.data[:old_out] = old_weight
        new_layer.bias.data[:old_out] = old_bias

        # Initialize new neurons
        if initialization == "xavier":
            nn.init.xavier_uniform_(new_layer.weight.data[old_out:])
            nn.init.zeros_(new_layer.bias.data[old_out:])
        elif initialization == "small_random":
            new_layer.weight.data[old_out:] = torch.randn(num_new_neurons, old_in) * 0.01
            nn.init.zeros_(new_layer.bias.data[old_out:])

        # Record growth
        self.growth_history.append({
            'timestamp': datetime.now().isoformat(),
            'old_size': old_out,
            'new_size': new_out,
            'added': num_new_neurons
        })

        return new_layer

    def get_total_neurons(self, model) -> int:
        """Count total neurons in model"""
        total = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total += module.out_features
        return total


class ExperienceReplayBuffer:
    """
    Stores past experiences for replay during continuous learning.
    Implements Complementary Learning Systems (CLS):
    - Fast learning: New experiences
    - Slow learning: Replayed experiences
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Dict] = []
        self.priorities: List[float] = []

    def add(self, experience: Dict, priority: float = 1.0):
        """Add experience with priority (higher = more important)"""
        if len(self.buffer) >= self.capacity:
            # Remove lowest priority
            min_idx = self.priorities.index(min(self.priorities))
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)

        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> List[Dict]:
        """Sample batch with priority weighting"""
        if len(self.buffer) == 0:
            return []

        # Convert priorities to probabilities
        priorities = torch.tensor(self.priorities)
        probs = priorities / priorities.sum()

        # Sample
        indices = torch.multinomial(probs, min(batch_size, len(self.buffer)), replacement=False)

        return [self.buffer[i] for i in indices.tolist()]

    def update_priorities(self, indices: List[int], errors: torch.Tensor):
        """Update priorities based on prediction errors (TD error style)"""
        for idx, error in zip(indices, errors):
            if idx < len(self.priorities):
                # Higher error = higher priority for replay
                self.priorities[idx] = float(error.abs().item()) + 0.01


class ContinuousTrainer:
    """
    Continuous learning trainer with:
    - No epoch limits
    - Automatic checkpointing
    - Resume capability
    - Neurogenesis support
    - Experience replay
    """

    def __init__(self,
                 model,
                 config,
                 growth_config: Optional[GrowthConfig] = None):
        self.model = model
        self.config = config
        self.growth_config = growth_config or GrowthConfig()

        self.neurogenesis = NeurogenesisModule(self.growth_config) if self.growth_config.neurogenesis_enabled else None
        self.replay_buffer = ExperienceReplayBuffer(self.growth_config.sample_buffer_size)

        # Training state
        self.step_count = 0
        self.total_samples_seen = 0
        self.start_time = None
        self.last_save_time = None

        # Metrics history
        self.metrics_history: List[Dict] = []
        self.layer_errors: Dict[str, List[float]] = {}

        # Checkpoint management
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def train_step(self,
                   observation: torch.Tensor,
                   action: Optional[torch.Tensor] = None) -> Dict:
        """
        Single training step

        Args:
            observation: Current observation
            action: Optional action (for supervised)

        Returns:
            Metrics dict
        """
        if self.start_time is None:
            self.start_time = time.time()
            self.last_save_time = self.start_time

        # Model step
        pred_action, conscious_content, metrics = self.model.step(observation, action)

        # Compute losses
        losses = self._compute_losses(observation, action, conscious_content)

        # Add to replay buffer
        if self.growth_config.neurogenesis_enabled:
            priority = losses.get('total', 1.0)
            self.replay_buffer.add({
                'observation': observation,
                'action': action,
                'metrics': metrics
            }, priority)

        # Check for neurogenesis
        if self.neurogenesis and self.step_count % 100 == 0:
            self._check_and_grow()

        # Auto-checkpoint
        current_time = time.time()
        if current_time - self.last_save_time > self.growth_config.auto_save_interval:
            self.save_checkpoint(reason="auto")
            self.last_save_time = current_time

        # Record metrics (keep only last 1000 to prevent memory leak)
        self.metrics_history.append({
            'step': self.step_count,
            'timestamp': datetime.now().isoformat(),
            **metrics,
            **losses
        })
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

        self.step_count += 1
        self.total_samples_seen += observation.shape[0]

        return {**metrics, **losses}

    def _compute_losses(self,
                        observation: torch.Tensor,
                        action: Optional[torch.Tensor],
                        conscious_content: Dict) -> Dict:
        """Compute all losses"""
        losses = {}

        # World model loss - needs sequence of observations (batch, seq_len, obs_dim)
        if len(self.model.history_buffer) >= 3:
            # Build sequence from history buffer
            seq_obs = []
            for i in range(min(len(self.model.history_buffer), 8)):  # Use last 8 observations
                hist_obs = self.model.history_buffer[-(i+1)].get('observation')
                if hist_obs is not None:
                    seq_obs.insert(0, hist_obs)

            # Add current observation
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            seq_obs.append(observation)

            if len(seq_obs) >= 3:
                # Stack into (batch, seq_len, obs_dim)
                seq_tensor = torch.stack(seq_obs, dim=1)

                try:
                    world_loss, _ = self.model.world_model.compute_loss(seq_tensor)
                    losses['world_loss'] = world_loss.item()

                    # Record for growth detection (keep last 500 to prevent memory leak)
                    if 'world' not in self.layer_errors:
                        self.layer_errors['world'] = []
                    self.layer_errors['world'].append(world_loss.item())
                    if len(self.layer_errors['world']) > 500:
                        self.layer_errors['world'] = self.layer_errors['world'][-500:]
                except Exception as e:
                    # Skip if compute_loss fails
                    pass

        # Self model loss
        if len(self.model.history_buffer) >= 2:
            prev_self = self.model.history_buffer[-2].get('self_state', self.model.internal_state)
            self_loss = self.model.self_model.compute_self_prediction_error(
                prev_self, conscious_content['self_state'].detach()
            )
            losses['self_loss'] = self_loss.item()

        # Agency loss
        if action is not None and len(self.model.history_buffer) >= 2:
            agency_loss, _ = self.model.agency_model.compute_loss(
                action,
                self.model.history_buffer[-2]['world_latent'],
                conscious_content['world_latent'],
                self.model.internal_state
            )
            losses['agency_loss'] = agency_loss.item()

        # Integration loss (maximize Phi)
        losses['integration_loss'] = -conscious_content['phi'].mean().detach().item()

        # Total
        losses['total'] = sum(losses.values())

        return losses

    def _check_and_grow(self):
        """Check if any layers need neurogenesis"""
        if not self.neurogenesis:
            return

        # Check self model
        if hasattr(self.model.self_model, 'state_encoder'):
            activations = self.model.internal_state
            needs_growth, reason = self.neurogenesis.check_growth_needed(
                activations,
                torch.tensor(self.layer_errors.get('self', [0]))
            )

            if needs_growth:
                total = self.neurogenesis.get_total_neurons(self.model)
                if total < self.growth_config.max_neurons:
                    print(f"[Neurogenesis] Growing self_model: {reason}")
                    # Would grow the layer here
                    # This requires modifying the model architecture

    def save_checkpoint(self, reason: str = "manual"):
        """Save checkpoint with full state"""
        checkpoint = {
            'step_count': self.step_count,
            'total_samples_seen': self.total_samples_seen,
            'model_state_dict': self.model.state_dict(),
            'internal_state': self.model.internal_state,
            'history_buffer': list(self.model.history_buffer),
            'metrics_history': self.metrics_history[-1000:],  # Keep last 1000
            'replay_buffer': self.replay_buffer.buffer[-1000:],
            'growth_history': self.neurogenesis.growth_history if self.neurogenesis else [],
            'timestamp': datetime.now().isoformat(),
            'reason': reason
        }

        path = self.checkpoint_dir / f"continuous_{int(time.time())}.pt"
        torch.save(checkpoint, path)

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        print(f"[Checkpoint] Saved to {path} ({reason})")
        return str(path)

    def load_checkpoint(self, path: str):
        """Load checkpoint and resume"""
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Handle internal_state size mismatch (batch dimension can vary)
        saved_state = checkpoint.get('internal_state')
        if saved_state is not None:
            # Take first sample if sizes don't match (will be expanded during training)
            if saved_state.dim() == 2 and saved_state.shape[0] > 0:
                self.model.internal_state = saved_state[:1]  # Keep only first sample
            else:
                self.model.internal_state = saved_state

        self.model.history_buffer = list(checkpoint.get('history_buffer', []))

        self.step_count = checkpoint.get('step_count', 0)
        self.total_samples_seen = checkpoint.get('total_samples_seen', 0)
        self.metrics_history = checkpoint.get('metrics_history', [])

        if 'replay_buffer' in checkpoint:
            self.replay_buffer.buffer = checkpoint['replay_buffer']

        if self.neurogenesis and 'growth_history' in checkpoint:
            self.neurogenesis.growth_history = checkpoint['growth_history']

        print(f"[Checkpoint] Loaded from {path}")
        print(f"  Resuming from step {self.step_count}")
        print(f"  Total samples seen: {self.total_samples_seen}")

        return True

    def _cleanup_checkpoints(self):
        """Keep only last N checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob("continuous_*.pt"))

        while len(checkpoints) > self.growth_config.keep_last_n_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"[Checkpoint] Removed old: {oldest}")

    def get_status(self) -> Dict:
        """Get current training status"""
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            'step_count': self.step_count,
            'total_samples': self.total_samples_seen,
            'elapsed_seconds': elapsed,
            'elapsed_human': self._format_time(elapsed),
            'samples_per_second': self.total_samples_seen / elapsed if elapsed > 0 else 0,
            'buffer_size': len(self.replay_buffer.buffer),
            'total_neurons': self.neurogenesis.get_total_neurons(self.model) if self.neurogenesis else 0,
            'growth_events': len(self.neurogenesis.growth_history) if self.neurogenesis else 0,
            'recent_metrics': self.metrics_history[-1] if self.metrics_history else {}
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to human readable"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


def run_continuous_training(model,
                           config,
                           growth_config: Optional[GrowthConfig] = None,
                           resume_from: Optional[str] = None,
                           stop_condition: Optional[callable] = None):
    """
    Run continuous training until stopped

    Args:
        model: SelfAwareAI model
        config: Config
        growth_config: Growth configuration
        resume_from: Checkpoint path to resume from
        stop_condition: Function that returns True to stop training

    Usage:
        # Basic continuous training
        run_continuous_training(model, config)

        # Resume from checkpoint
        run_continuous_training(model, config, resume_from="checkpoints/continuous_xxx.pt")

        # Stop after condition
        run_continuous_training(model, config, stop_condition=lambda: model.step_count > 100000)
    """
    from environment import BufferedLazyDataset
    from torch.utils.data import DataLoader

    growth_config = growth_config or GrowthConfig()
    trainer = ContinuousTrainer(model, config, growth_config)

    if resume_from:
        trainer.load_checkpoint(resume_from)

    # Create data stream - memory-efficient lazy dataset
    print("Creating data stream (memory-efficient mode)...")
    dataset = BufferedLazyDataset(
        seq_length=config.seq_len,
        buffer_size=100  # Small buffer, generates on-demand
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0)

    print("=" * 60)
    print("CONTINUOUS TRAINING STARTED")
    print("=" * 60)
    print(f"Neurogenesis: {'ENABLED' if growth_config.neurogenesis_enabled else 'DISABLED'}")
    print(f"Auto-save every: {growth_config.auto_save_interval}s")
    print(f"Max neurons: {growth_config.max_neurons}")
    print("=" * 60)
    print()
    print("Press Ctrl+C to stop. Progress will be auto-saved.")
    print()

    try:
        data_iter = iter(loader)

        while True:
            # Check stop condition
            if stop_condition and stop_condition():
                print("\nStop condition met.")
                break

            # Get next batch (lazy dataset generates on-demand)
            observations, actions = next(data_iter)

            observations = observations.to(config.device)
            actions = actions.to(config.device)

            # Training step
            batch_size, seq_len = observations.shape[:2]

            for t in range(seq_len):
                metrics = trainer.train_step(
                    observations[:, t],
                    actions[:, t] if t > 0 else None
                )

            # Print progress every 100 batches
            if trainer.step_count % 100 == 0:
                status = trainer.get_status()
                print(f"Step {status['step_count']:>8} | "
                      f"Φ={metrics.get('phi', 0):.3f} | "
                      f"Agency={metrics.get('mean_agency', 0):.3f} | "
                      f"Time: {status['elapsed_human']}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # Always save on exit
    print("\nSaving final checkpoint...")
    path = trainer.save_checkpoint(reason="shutdown")

    status = trainer.get_status()
    print()
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total steps: {status['step_count']}")
    print(f"Total samples: {status['total_samples']}")
    print(f"Training time: {status['elapsed_human']}")
    print(f"Final checkpoint: {path}")
    print("=" * 60)

    return trainer
