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
