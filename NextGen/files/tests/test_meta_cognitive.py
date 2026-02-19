import pytest
import torch

def test_meta_cognitive_init():
    """Test MetaCognitiveModel initialization"""
    from model.meta_cognitive import MetaCognitiveModel

    model = MetaCognitiveModel(
        world_latent_dim=256,
        self_state_dim=128,
        hidden_dim=512
    )

    assert model.hidden_dim == 512
    assert model.process_modeler is not None
    assert model.confidence_estimator is not None


def test_meta_cognitive_introspect():
    """Test MetaCognitiveModel introspect"""
    from model.meta_cognitive import MetaCognitiveModel

    model = MetaCognitiveModel(world_latent_dim=256, self_state_dim=128, hidden_dim=512)

    batch_size = 4
    world_state = torch.randn(batch_size, 256)
    self_state = torch.randn(batch_size, 128)

    output = model.introspect(world_state, self_state)

    assert 'meta_representation' in output
    assert 'confidence' in output
    assert 'epistemic_uncertainty' in output
    assert output['meta_representation'].shape == (batch_size, 512)
    assert output['confidence'].shape == (batch_size, 1)


def test_meta_cognitive_self_report():
    """Test MetaCognitiveModel generate_self_report"""
    from model.meta_cognitive import MetaCognitiveModel

    model = MetaCognitiveModel(world_latent_dim=256, self_state_dim=128, hidden_dim=512)

    batch_size = 2
    world_state = torch.randn(batch_size, 256)
    self_state = torch.randn(batch_size, 128)

    introspection = model.introspect(world_state, self_state)
    report = model.generate_self_report(introspection, self_state)

    assert 'confidence_level' in report
    assert 'uncertainty' in report
    assert 'interpretation' in report
    assert isinstance(report['interpretation'], str)
