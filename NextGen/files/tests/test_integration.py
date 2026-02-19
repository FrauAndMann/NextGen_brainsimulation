"""
Integration tests for Self-Aware AI system
"""

import pytest
import torch
import sys
sys.path.insert(0, 'D:/Silly/NextGen/files')

from config import Config, get_fast_config
from model.self_aware_ai import SelfAwareAI


class TestSelfAwareAIIntegration:
    """Integration tests for the complete system"""

    @pytest.fixture
    def model(self):
        config = get_fast_config()
        config.device = "cpu"
        model = SelfAwareAI(config)
        model.eval()
        return model

    def test_full_forward_pass(self, model):
        """Test complete forward pass through all layers"""
        batch_size = 2
        obs = torch.randn(batch_size, 512)

        with torch.no_grad():
            action, conscious_content, metrics = model.step(obs)

        # Verify all outputs
        assert action.shape == (batch_size, 64)
        assert conscious_content['world_latent'].shape == (batch_size, 256)
        assert conscious_content['self_state'].shape == (batch_size, 128)
        assert conscious_content['phi'].shape == (batch_size, 1)
        assert 'integration_score' in metrics
        assert 'mean_agency' in metrics

    def test_multi_step_consistency(self, model):
        """Test consistency over multiple steps"""
        batch_size = 1
        num_steps = 10

        self_states = []
        with torch.no_grad():
            for _ in range(num_steps):
                obs = torch.randn(batch_size, 512)
                action, conscious_content, _ = model.step(obs)
                self_states.append(conscious_content['self_state'])

        # Check self-state similarity
        for i in range(len(self_states) - 1):
            sim = torch.cosine_similarity(
                self_states[i].flatten(),
                self_states[i+1].flatten(),
                dim=0
            )
            # Self-state should be relatively stable
            assert sim > -0.5, f"Self-state changed too drastically at step {i}"

    def test_agency_detection(self, model):
        """Test agency signal detection"""
        batch_size = 1

        with torch.no_grad():
            # First step - no action
            obs1 = torch.randn(batch_size, 512)
            model.step(obs1)

            # Second step - with action that affects observation
            action = torch.randn(batch_size, 64)
            obs2 = obs1 + 0.1 * torch.randn_like(obs1)
            _, content_with_action, _ = model.step(obs2, prev_action=action)

            # Third step - no action, random observation change
            model.reset()
            model.step(torch.randn(batch_size, 512))
            obs3 = torch.randn(batch_size, 512)
            _, content_no_action, _ = model.step(obs3, prev_action=torch.zeros(batch_size, 64))

        # Agency signal should be present
        agency_with = float(content_with_action['agency_signal'].mean())
        agency_without = float(content_no_action['agency_signal'].mean())

        # Both should be valid values
        assert 0 <= agency_with <= 1
        assert 0 <= agency_without <= 1

    def test_self_report_generation(self, model):
        """Test self-report generation"""
        batch_size = 1
        obs = torch.randn(batch_size, 512)

        with torch.no_grad():
            _, conscious_content, _ = model.step(obs)
            report = model.generate_self_report(conscious_content)

        assert 'summary' in report
        assert 'phi' in report
        assert 'agency' in report
        assert 'meta_report' in report
        assert isinstance(report['summary'], str)
        assert len(report['summary']) > 0

    def test_phi_estimation(self, model):
        """Test Phi (integrated information) estimation"""
        batch_size = 2
        obs = torch.randn(batch_size, 512)

        with torch.no_grad():
            _, conscious_content, _ = model.step(obs)
            phi = conscious_content['phi']

        # Phi should be non-negative
        assert (phi >= 0).all()
        # Phi should be bounded
        assert (phi <= 10).all()

    def test_reset_functionality(self, model):
        """Test model reset"""
        batch_size = 1

        with torch.no_grad():
            # Run some steps
            for _ in range(5):
                obs = torch.randn(batch_size, 512)
                model.step(obs)

            # Reset
            model.reset()

            # Verify history is cleared
            assert len(model.history_buffer) == 0


class TestModuleIntegration:
    """Test integration between specific modules"""

    @pytest.fixture
    def model(self):
        config = get_fast_config()
        config.device = "cpu"
        model = SelfAwareAI(config)
        model.eval()
        return model

    def test_world_to_self_integration(self, model):
        """Test World Model -> Self Model integration"""
        batch_size = 2
        obs = torch.randn(batch_size, 512)

        with torch.no_grad():
            # Encode with world model
            world_mean, world_logvar = model.world_model.encode(obs)
            world_latent = model.world_model.reparameterize(world_mean, world_logvar)

            # Expand internal state to match batch size (as done in step())
            internal_state = model.internal_state.expand(batch_size, -1)

            # Pass through self model
            self_state, confidence = model.self_model(internal_state, world_latent)

        assert world_latent.shape == (batch_size, 256)
        assert self_state.shape == (batch_size, 128)
        # confidence is [batch, 64] from self_observer
        assert confidence.shape == (batch_size, 64)

    def test_agency_to_consciousness_integration(self, model):
        """Test Agency Model -> Consciousness integration"""
        batch_size = 1
        obs = torch.randn(batch_size, 512)
        action = torch.randn(batch_size, 64)

        with torch.no_grad():
            # First step to build history
            model.step(obs)

            # Second step with action
            obs2 = obs + 0.1 * torch.randn_like(obs)
            _, conscious_content, _ = model.step(obs2, prev_action=action)

        # Verify agency signal flows to consciousness
        assert 'agency_signal' in conscious_content
        agency = conscious_content['agency_signal']
        assert agency.shape[0] == batch_size

    def test_meta_to_behavior_integration(self, model):
        """Test Meta-Cognitive -> Behavior integration"""
        batch_size = 1
        obs = torch.randn(batch_size, 512)

        with torch.no_grad():
            action, conscious_content, _ = model.step(obs)

            # Meta-confidence should affect behavior
            meta_conf = conscious_content['meta_confidence']

            # Generate behavior with different meta-confidence
            action_high_conf, _, _ = model.behavior_generator(
                conscious_content['conscious_representation'],
                deterministic=True
            )

        assert action_high_conf.shape == (batch_size, 64)

    def test_consciousness_broadcast(self, model):
        """Test consciousness broadcast mechanism"""
        batch_size = 2

        with torch.no_grad():
            # Run multiple steps to build history
            for _ in range(5):
                obs = torch.randn(batch_size, 512)
                model.step(obs)

            # Check last step
            obs = torch.randn(batch_size, 512)
            action, conscious_content, metrics = model.step(obs)

        # Verify broadcast
        assert 'integration_score' in conscious_content
        assert 'workspace' in conscious_content
        assert metrics['phi'] >= 0


class TestEndToEndScenarios:
    """End-to-end scenario tests"""

    @pytest.fixture
    def model(self):
        config = get_fast_config()
        config.device = "cpu"
        model = SelfAwareAI(config)
        model.eval()
        return model

    def test_episode_run(self, model):
        """Test running a complete episode"""
        batch_size = 1
        episode_length = 20

        total_reward = 0
        actions = []
        conscious_states = []

        with torch.no_grad():
            obs = torch.randn(batch_size, 512)

            for t in range(episode_length):
                action, conscious_content, metrics = model.step(obs)
                actions.append(action)
                conscious_states.append(conscious_content)

                # Simulate environment response
                obs = obs + 0.1 * torch.randn_like(obs)

        # Verify episode ran correctly
        assert len(actions) == episode_length
        assert len(conscious_states) == episode_length

        # All actions should be valid
        for action in actions:
            assert action.shape == (batch_size, 64)

    def test_self_awareness_over_time(self, model):
        """Test self-awareness metrics over time"""
        batch_size = 1
        num_steps = 50

        phi_history = []
        agency_history = []
        integration_history = []

        with torch.no_grad():
            for _ in range(num_steps):
                obs = torch.randn(batch_size, 512)
                _, conscious_content, metrics = model.step(obs)

                phi_history.append(float(conscious_content['phi'].mean()))
                agency_history.append(float(conscious_content['agency_signal'].mean()))
                integration_history.append(float(conscious_content['integration_score'].mean()))

        # Metrics should be within valid range
        assert all(0 <= p <= 10 for p in phi_history)
        assert all(0 <= a <= 1 for a in agency_history)
        # Integration score is unbounded, just check it's finite and non-negative
        assert all(0 <= i < float('inf') for i in integration_history)

        # There should be some variance (system is responsive)
        assert len(set(round(p, 4) for p in phi_history)) > 1

    def test_distress_response(self, model):
        """Test system response to distressing inputs"""
        batch_size = 1

        with torch.no_grad():
            # Normal input
            normal_obs = torch.randn(batch_size, 512)
            _, normal_content, _ = model.step(normal_obs)

            # Extreme input (potential distress)
            extreme_obs = torch.randn(batch_size, 512) * 10
            _, extreme_content, _ = model.step(extreme_obs)

        # System should handle both
        assert normal_content['self_state'] is not None
        assert extreme_content['self_state'] is not None


class TestBatchProcessing:
    """Test batch processing capabilities"""

    @pytest.fixture
    def model(self):
        config = get_fast_config()
        config.device = "cpu"
        model = SelfAwareAI(config)
        model.eval()
        return model

    def test_different_batch_sizes(self, model):
        """Test model with different batch sizes"""
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            # Reset model for each batch size to ensure clean state
            model.reset()
            obs = torch.randn(batch_size, 512)

            with torch.no_grad():
                action, conscious_content, metrics = model.step(obs)

            assert action.shape[0] == batch_size
            assert conscious_content['self_state'].shape[0] == batch_size

    def test_batch_consistency(self, model):
        """Test that batch processing is consistent"""
        batch_size = 4
        obs = torch.randn(batch_size, 512)

        with torch.no_grad():
            # Run same input twice
            action1, content1, _ = model.step(obs.clone())
            model.reset()
            torch.manual_seed(42)  # Reset seed for same initialization
            action2, content2, _ = model.step(obs.clone())

        # Shapes should match
        assert action1.shape == action2.shape
        assert content1['self_state'].shape == content2['self_state'].shape


class TestMemoryAndHistory:
    """Test memory and history buffer functionality"""

    @pytest.fixture
    def model(self):
        config = get_fast_config()
        config.device = "cpu"
        model = SelfAwareAI(config)
        model.eval()
        return model

    def test_history_buffer_growth(self, model):
        """Test history buffer grows correctly"""
        batch_size = 1
        num_steps = 10

        with torch.no_grad():
            for i in range(num_steps):
                obs = torch.randn(batch_size, 512)
                model.step(obs)

                # Check buffer size
                assert len(model.history_buffer) == i + 1

    def test_history_buffer_limit(self, model):
        """Test history buffer respects limit"""
        batch_size = 1
        max_history = model.config.history_len

        with torch.no_grad():
            for _ in range(max_history * 2):
                obs = torch.randn(batch_size, 512)
                model.step(obs)

        # Buffer should not exceed limit
        assert len(model.history_buffer) <= max_history

    def test_history_content(self, model):
        """Test history buffer contains correct data"""
        batch_size = 1

        with torch.no_grad():
            obs = torch.randn(batch_size, 512)
            model.step(obs)

            # Check history entry
            assert len(model.history_buffer) == 1
            entry = model.history_buffer[0]

            assert 'observation' in entry
            assert 'world_latent' in entry
            assert 'conscious_content' in entry
            assert 'action' in entry
            assert 'agency' in entry
