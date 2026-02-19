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
