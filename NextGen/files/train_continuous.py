"""
Continuous Training Script for SYNAPSE

A brain doesn't train in "epochs" - it learns continuously.
This script implements true continuous learning:

- No epoch limits
- Auto-save every N minutes
- Resume from checkpoint
- Neurogenesis (growing new neurons)
- Experience replay
- Real-time metrics
- Graceful shutdown (Ctrl+C saves progress)

Usage:
    python train_continuous.py                    # Start fresh
    python train_continuous.py --resume <path>    # Resume from checkpoint
    python train_continuous.py --hours 24         # Stop after 24 hours
    python train_continuous.py --steps 1000000    # Stop after 1M steps
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import torch

from config import Config, get_fast_config
from model.self_aware_ai import SelfAwareAI
from continuous_learning import (
    GrowthConfig,
    ContinuousTrainer,
    run_continuous_training
)
from environment import LazySyntheticDataset, BufferedLazyDataset
from shared_metrics import get_shared_metrics
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="SYNAPSE Continuous Training")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--auto-resume", action="store_true",
                       help="Automatically find and load the latest checkpoint")

    # Stop conditions (optional)
    parser.add_argument("--steps", type=int, default=None,
                       help="Stop after N steps")
    parser.add_argument("--hours", type=float, default=None,
                       help="Stop after N hours")
    parser.add_argument("--samples", type=int, default=None,
                       help="Stop after N total samples")

    # Growth config
    parser.add_argument("--neurogenesis", action="store_true", default=True,
                       help="Enable neurogenesis (default: True)")
    parser.add_argument("--no-neurogenesis", action="store_true",
                       help="Disable neurogenesis")
    parser.add_argument("--max-neurons", type=int, default=100000,
                       help="Maximum neurons (safety limit)")

    # Checkpointing
    parser.add_argument("--save-interval", type=int, default=300,
                       help="Auto-save interval in seconds (default: 300 = 5 min)")
    parser.add_argument("--keep-checkpoints", type=int, default=5,
                       help="Number of checkpoints to keep")

    # Config preset
    parser.add_argument("--config", type=str, default="default",
                       choices=["fast", "default", "full"],
                       help="Configuration preset")

    # Data source
    parser.add_argument("--data-type", type=str, default="synthetic",
                       choices=["synthetic", "images", "text", "rl", "timeseries"],
                       help="Type of data to train on")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to data (for images/text/timeseries)")
    parser.add_argument("--env-name", type=str, default="CartPole-v1",
                       help="RL environment name (for --data-type rl)")

    return parser.parse_args()


class TrainingController:
    """Controls training flow with stop conditions"""

    def __init__(self, args, trainer):
        self.args = args
        self.trainer = trainer
        self.start_time = time.time()
        self.stop_requested = False

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\nReceived stop signal. Saving progress...")
        self.stop_requested = True

    def should_stop(self) -> bool:
        """Check all stop conditions"""
        if self.stop_requested:
            return True

        # Steps limit
        if self.args.steps and self.trainer.step_count >= self.args.steps:
            print(f"\nReached step limit: {self.args.steps}")
            return True

        # Time limit
        if self.args.hours:
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= self.args.hours:
                print(f"\nReached time limit: {self.args.hours} hours")
                return True

        # Samples limit
        if self.args.samples and self.trainer.total_samples_seen >= self.args.samples:
            print(f"\nReached samples limit: {self.args.samples}")
            return True

        return False


def main():
    args = parse_args()

    # Get config
    if args.config == "fast":
        config = get_fast_config()
    elif args.config == "full":
        config = Config()
        config.batch_size = 64
    else:
        config = Config()

    print("=" * 60)
    print("SYNAPSE CONTINUOUS TRAINING")
    print("=" * 60)
    print()

    # Growth config
    growth_config = GrowthConfig(
        neurogenesis_enabled=args.neurogenesis and not args.no_neurogenesis,
        max_neurons=args.max_neurons,
        auto_save_interval=args.save_interval,
        keep_last_n_checkpoints=args.keep_checkpoints,
    )

    print("Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Neurogenesis: {'ENABLED' if growth_config.neurogenesis_enabled else 'DISABLED'}")
    print(f"  Max neurons: {growth_config.max_neurons:,}")
    print(f"  Auto-save: every {growth_config.auto_save_interval}s")
    print()

    # Initialize model
    print("Initializing model...")
    model = SelfAwareAI(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print()

    # Initialize trainer
    trainer = ContinuousTrainer(model, config, growth_config)

    # Resume if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            # Try to find it in checkpoints dir
            checkpoint_path = Path(config.checkpoint_dir) / args.resume

        if checkpoint_path.exists():
            trainer.load_checkpoint(str(checkpoint_path))
        else:
            print(f"ERROR: Checkpoint not found: {args.resume}")
            sys.exit(1)

    # Auto-resume: find latest checkpoint
    elif args.auto_resume:
        checkpoint_dir = Path(config.checkpoint_dir)
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("continuous_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if checkpoints:
                latest = checkpoints[0]
                print(f"Auto-resuming from: {latest.name}")
                trainer.load_checkpoint(str(latest))
            else:
                print("No checkpoints found. Starting fresh.")
        else:
            print("No checkpoints directory. Starting fresh.")

    # Create controller
    controller = TrainingController(args, trainer)

    # Initialize shared metrics for dashboard
    shared_metrics = get_shared_metrics()
    shared_metrics.set_state("running")

    # Data stream - select based on data-type
    print(f"Creating data stream (type: {args.data_type})...")

    if args.data_type == "synthetic":
        dataset = BufferedLazyDataset(
            seq_length=config.seq_len,
            buffer_size=100
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=config.pin_memory if torch.cuda.is_available() else False
        )
    else:
        # Real data
        from real_data import create_data_loader
        data_loader = create_data_loader(
            data_type=args.data_type,
            data_path=args.data_path,
            obs_dim=config.obs_dim,
            seq_len=config.seq_len,
            env_name=args.env_name
        )

        # Wrap as iterable dataset
        class RealDataIterable(torch.utils.data.IterableDataset):
            def __init__(self, loader):
                self.loader = loader

            def __iter__(self):
                while True:
                    obs, actions = self.loader.generate_trajectory()
                    yield obs, actions

        dataset = RealDataIterable(data_loader)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=config.pin_memory if torch.cuda.is_available() else False
        )

    print()
    print("=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    print()
    print("Controls:")
    print("  Ctrl+C - Stop training and save progress")
    print()
    print("Stop conditions:")
    if args.steps:
        print(f"  Steps: {args.steps:,}")
    if args.hours:
        print(f"  Time: {args.hours} hours")
    if args.samples:
        print(f"  Samples: {args.samples:,}")
    if not (args.steps or args.hours or args.samples):
        print("  None - will run until Ctrl+C")
    print()
    print("-" * 60)
    print()

    # Training loop
    data_iter = iter(loader)
    batch_count = 0

    try:
        with tqdm(desc="Training", unit="batch", ncols=100) as pbar:
            while not controller.should_stop():
                # Get next batch (lazy dataset generates on-demand, no StopIteration)
                observations, actions = next(data_iter)

                observations = observations.to(config.device)
                actions = actions.to(config.device)

                batch_size, seq_len = observations.shape[:2]

                # Process sequence
                model.reset()
                model.internal_state = torch.randn(
                    batch_size, config.self_state_dim, device=config.device
                )

                for t in range(seq_len):
                    metrics = trainer.train_step(
                        observations[:, t],
                        actions[:, t] if t > 0 else None
                    )

                batch_count += 1
                pbar.update(1)

                # Update progress bar
                status = trainer.get_status()
                pbar.set_postfix({
                    'Φ': f"{metrics.get('phi', 0):.3f}",
                    'Agency': f"{metrics.get('mean_agency', 0):.3f}",
                    'Steps': status['step_count']
                })

                # Update shared metrics for dashboard
                shared_metrics.update(
                    state="running",
                    step=status['step_count'],
                    total_samples=status['total_samples'],
                    elapsed_seconds=status.get('elapsed_seconds', 0),
                    metrics=metrics,
                    neurochemistry={
                        'dopamine': float(model.internal_state[0, 0].detach()) if model.internal_state.dim() > 1 else 0.5,
                        'serotonin': float(model.internal_state[0, 1].detach()) if model.internal_state.dim() > 1 else 0.5,
                        'oxytocin': float(model.internal_state[0, 2].detach()) if model.internal_state.dim() > 1 else 0.4,
                        'cortisol': float(model.internal_state[0, 3].detach()) if model.internal_state.dim() > 1 else 0.3,
                        'norepinephrine': float(model.internal_state[0, 4].detach()) if model.internal_state.dim() > 1 else 0.4,
                    },
                    neurons=status['total_neurons'],
                    growth_events=status['growth_events']
                )

                # Print summary every 500 batches
                if batch_count % 500 == 0:
                    status = trainer.get_status()
                    recent = trainer.metrics_history[-1] if trainer.metrics_history else {}

                    print()
                    print(f"[{time.strftime('%H:%M:%S')}] Step {status['step_count']:,}")
                    print(f"  Φ: {recent.get('phi', 0):.4f} | "
                          f"Agency: {recent.get('mean_agency', 0):.4f} | "
                          f"Integration: {recent.get('integration_score', 0):.4f}")
                    print(f"  Time: {status['elapsed_human']} | "
                          f"Samples: {status['total_samples']:,} | "
                          f"Neurons: {status['total_neurons']:,}")
                    print()

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Update shared metrics state
        shared_metrics.set_state("stopped")

        # Always save on exit
        print()
        print("=" * 60)
        print("SAVING FINAL CHECKPOINT")
        print("=" * 60)

        path = trainer.save_checkpoint(reason="shutdown")

        status = trainer.get_status()
        recent = trainer.metrics_history[-1] if trainer.metrics_history else {}

        print()
        print("=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total steps: {status['step_count']:,}")
        print(f"Total samples: {status['total_samples']:,}")
        print(f"Training time: {status['elapsed_human']}")
        print(f"Samples/sec: {status['samples_per_second']:.1f}")
        print(f"Final neurons: {status['total_neurons']:,}")
        print(f"Growth events: {status['growth_events']}")
        print()
        print("Final metrics:")
        print(f"  Φ (Phi): {recent.get('phi', 0):.4f}")
        print(f"  Agency: {recent.get('mean_agency', 0):.4f}")
        print(f"  Integration: {recent.get('integration_score', 0):.4f}")
        print(f"  Meta-confidence: {recent.get('meta_confidence', 0):.4f}")
        print()
        print(f"Checkpoint saved: {path}")
        print()
        print("To resume:")
        print(f"  python train_continuous.py --resume {Path(path).name}")
        print("=" * 60)


if __name__ == "__main__":
    main()
