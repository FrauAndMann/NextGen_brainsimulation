import pytest
import torch

def test_consciousness_init():
    """Test ConsciousnessIntegrator initialization"""
    from model.consciousness import ConsciousnessIntegrator

    model = ConsciousnessIntegrator(
        world_dim=256,
        self_dim=128,
        agency_dim=1,
        meta_dim=512,
        workspace_capacity=16,
        hidden_dim=512
    )

    assert model.workspace_capacity == 16
    assert model.hidden_dim == 512


def test_consciousness_broadcast():
    """Test ConsciousnessIntegrator broadcast_to_consciousness"""
    from model.consciousness import ConsciousnessIntegrator

    model = ConsciousnessIntegrator(
        world_dim=256, self_dim=128, agency_dim=1,
        meta_dim=512, workspace_capacity=16, hidden_dim=512
    )

    batch_size = 4
    signals = {
        'world': torch.randn(batch_size, 256),
        'self': torch.randn(batch_size, 128),
        'agency': torch.randn(batch_size, 1),
        'meta': torch.randn(batch_size, 512)
    }

    workspace, integration_score, conscious_content = model.broadcast_to_consciousness(signals)

    assert workspace.shape == (batch_size, 16, 512)
    assert integration_score.shape == (batch_size, 1)
    assert conscious_content.shape == (batch_size, 512)


def test_consciousness_phi():
    """Test ConsciousnessIntegrator compute_phi"""
    from model.consciousness import ConsciousnessIntegrator

    model = ConsciousnessIntegrator(
        world_dim=256, self_dim=128, agency_dim=1,
        meta_dim=512, workspace_capacity=16, hidden_dim=512
    )

    batch_size = 2
    workspace = torch.randn(batch_size, 16, 512)

    phi = model.compute_phi(workspace)

    assert phi.shape == (batch_size, 1)
    assert (phi >= 0).all()  # Phi should be non-negative
