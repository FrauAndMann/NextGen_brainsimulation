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
