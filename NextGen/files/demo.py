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
