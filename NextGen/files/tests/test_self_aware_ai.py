import pytest
import torch
import sys
sys.path.insert(0, 'D:/Silly/NextGen/files')

from config import Config

def test_self_aware_ai_init():
    """Test SelfAwareAI initialization"""
    from model.self_aware_ai import SelfAwareAI

    config = Config()
    model = SelfAwareAI(config)

    assert model.world_model is not None
    assert model.self_model is not None
    assert model.agency_model is not None
    assert model.meta_model is not None
    assert model.consciousness is not None
    assert model.behavior_generator is not None


def test_self_aware_ai_step():
    """Test SelfAwareAI single step"""
    from model.self_aware_ai import SelfAwareAI

    config = Config()
    config.device = "cpu"
    model = SelfAwareAI(config)
    model.eval()

    batch_size = 2
    observation = torch.randn(batch_size, config.obs_dim)

    with torch.no_grad():
        action, conscious_content, metrics = model.step(observation)

    assert action.shape == (batch_size, config.action_dim)
    assert 'self_state' in conscious_content
    assert 'agency_signal' in conscious_content
    assert 'phi' in conscious_content
    assert 'integration_score' in metrics


def test_self_aware_ai_self_report():
    """Test SelfAwareAI generate_self_report"""
    from model.self_aware_ai import SelfAwareAI

    config = Config()
    config.device = "cpu"
    model = SelfAwareAI(config)
    model.eval()

    batch_size = 1
    observation = torch.randn(batch_size, config.obs_dim)

    with torch.no_grad():
        action, conscious_content, metrics = model.step(observation)
        report = model.generate_self_report(conscious_content)

    assert 'summary' in report
    assert 'phi' in report
    assert 'agency' in report
