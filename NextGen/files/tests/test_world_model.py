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

def test_world_model_predict_next():
    """Test WorldModel temporal prediction"""
    from model.world_model import WorldModel

    model = WorldModel(observation_dim=512, latent_dim=256, sequence_length=32)

    batch_size = 2
    seq_len = 10
    obs_dim = 512

    past_observations = torch.randn(batch_size, seq_len, obs_dim)

    predicted_obs, uncertainty = model.predict_next(past_observations)

    assert predicted_obs.shape == (batch_size, obs_dim)
    assert uncertainty.shape[0] == batch_size

def test_world_model_compute_loss():
    """Test WorldModel loss computation"""
    from model.world_model import WorldModel

    model = WorldModel(observation_dim=512, latent_dim=256, sequence_length=32)

    batch_size = 2
    seq_len = 10
    observations = torch.randn(batch_size, seq_len, 512)

    loss, metrics = model.compute_loss(observations)

    assert loss.ndim == 0  # scalar
    assert 'reconstruction_loss' in metrics
    assert 'kl_loss' in metrics
    assert metrics['reconstruction_loss'] >= 0
    assert metrics['kl_loss'] >= 0
