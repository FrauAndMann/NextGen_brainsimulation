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
