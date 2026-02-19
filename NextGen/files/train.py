"""
Training script for Self-Aware AI
"""

import sys
import os

# Add files directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
