import pytest
import torch

def test_world_model_init():
    """Test WorldModel initialization"""
    from model.world_model import WorldModel

    model = WorldModel(
        observation_dim=512,
        latent_dim=256,
        sequence_length=32
    )

    assert model.latent_dim == 256
    assert model.encoder is not None
    assert model.temporal_model is not None
    assert model.decoder is not None
