# evaluation.py
"""
Evaluation и Visualization для Self-Aware AI
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from pathlib import Path

# Import for compatibility with new model
try:
    from model.self_aware_ai import SelfAwareAI
except ImportError:
    pass  # SelfAwareAI may not be available during initial setup


class SelfAwarenessEvaluator:
    """
    Comprehensive evaluator для самосознания
    """
    
    def __init__(self, model, save_dir='results'):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.results = {
            'tests': {},
            'metrics': {},
            'visualizations': []
        }
    
    def run_full_evaluation(self) -> Dict:
        """
        Запуск полной оценки
        """
        print("=" * 60)
        print("COMPREHENSIVE SELF-AWARENESS EVALUATION")
        print("=" * 60)
        
        # 1. Basic functionality tests
        print("\n[1/5] Running basic functionality tests...")
        self.test_basic_functionality()
        
        # 2. Self-awareness tests
        print("\n[2/5] Running self-awareness tests...")
        self.test_self_awareness()
        
        # 3. Temporal tests
        print("\n[3/5] Running temporal consistency tests...")
        self.test_temporal_consistency()
        
        # 4. Agency tests
        print("\n[4/5] Running agency tests...")
        self.test_agency_detection()
        
        # 5. Integration tests
        print("\n[5/5] Running consciousness integration tests...")
        self.test_consciousness_integration()
        
        # Generate report
        print("\n" + "=" * 60)
        print("Generating evaluation report...")
        self.generate_report()
        
        return self.results
    
    def test_basic_functionality(self):
        """Test basic model functionality"""
        results = {}
        
        # Test 1: Forward pass
        try:
            obs = torch.randn(1, 512).cuda()
            action, conscious_content, metrics = self.model.step(obs)
            results['forward_pass'] = True
        except Exception as e:
            results['forward_pass'] = False
            results['forward_pass_error'] = str(e)
        
        # Test 2: Gradient flow
        try:
            loss = conscious_content['world_latent'].sum()
            loss.backward()
            results['gradient_flow'] = True
        except Exception as e:
            results['gradient_flow'] = False
            results['gradient_flow_error'] = str(e)
        
        # Test 3: Self-report generation
        try:
            report = self.model.generate_self_report(conscious_content)
            results['self_report_generation'] = True
            results['sample_report'] = report['summary']
        except Exception as e:
            results['self_report_generation'] = False
        
        self.results['tests']['basic_functionality'] = results
        
        print(f"  ✓ Forward pass: {results['forward_pass']}")
        print(f"  ✓ Gradient flow: {results['gradient_flow']}")
        print(f"  ✓ Self-report: {results['self_report_generation']}")
    
    def test_self_awareness(self):
        """Test self-awareness capabilities"""
        results = {}
        
        # Mirror test
        print("  Running mirror test...")
        mirror_score = self._mirror_test()
        results['mirror_test'] = {
            'score': float(mirror_score),
            'passed': mirror_score > 0.7
        }
        
        # Meta-cognition test
        print("  Running meta-cognition test...")
        meta_score = self._metacognition_test()
        results['metacognition'] = {
            'score': float(meta_score),
            'passed': meta_score > 0.6
        }
        
        # Self-boundary test
        print("  Running self-boundary test...")
        boundary_score = self._self_boundary_test()
        results['self_boundary'] = {
            'score': float(boundary_score),
            'passed': boundary_score > 0.3
        }
        
        self.results['tests']['self_awareness'] = results
        
        for test_name, result in results.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"  {status} {test_name}: {result['score']:.3f}")
    
    def _mirror_test(self) -> float:
        """
        Зеркальный тест: распознаёт ли система себя?
        """
        self.model.eval()
        
        obs = torch.randn(1, 512).cuda()
        action = torch.randn(1, 64).cuda()
        
        # First step
        self.model.step(obs)
        
        # Second step with action effect
        next_obs = obs + 0.1 * torch.randn_like(obs)
        _, conscious_content, _ = self.model.step(next_obs, prev_action=action)
        
        return float(conscious_content['agency_signal'].mean())
    
    def _metacognition_test(self) -> float:
        """
        Метапознание: система знает, что она знает?
        """
        self.model.eval()
        
        # Build history
        for _ in range(10):
            obs = torch.randn(1, 512).cuda()
            self.model.step(obs)
        
        # Final step
        obs = torch.randn(1, 512).cuda()
        _, conscious_content, _ = self.model.step(obs)
        
        confidence = float(conscious_content['meta_confidence'].mean())
        uncertainty = float(conscious_content['meta_uncertainty'].mean())
        
        # High confidence and low uncertainty = good metacognition
        return confidence * (1 - uncertainty)
    
    def _self_boundary_test(self) -> float:
        """
        Граница себя: различает ли система "я" от "не-я"?
        """
        self.model.eval()
        
        # Test own action
        obs1 = torch.randn(1, 512).cuda()
        action = torch.randn(1, 64).cuda()
        self.model.step(obs1)
        obs2 = obs1 + 0.2 * torch.randn_like(obs1)
        _, content1, _ = self.model.step(obs2, prev_action=action)
        agency_own = float(content1['agency_signal'].mean())
        
        # Test external change
        obs3 = obs2 + torch.randn(1, 512).cuda() * 0.5
        _, content2, _ = self.model.step(obs3, prev_action=torch.zeros_like(action))
        agency_external = float(content2['agency_signal'].mean())
        
        # Difference should be large
        return agency_own - agency_external
    
    def test_temporal_consistency(self):
        """Test temporal consistency of self-model"""
        results = {}
        
        self.model.eval()
        self_states = []
        
        # Collect self states over time
        for t in range(50):
            obs = torch.randn(1, 512).cuda()
            _, conscious_content, _ = self.model.step(obs)
            self_states.append(conscious_content['self_state'].detach())
        
        # Compute similarities
        similarities = []
        for t in range(len(self_states) - 1):
            sim = torch.cosine_similarity(
                self_states[t],
                self_states[t + 1],
                dim=-1
            )
            similarities.append(float(sim.mean()))
        
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        results['mean_similarity'] = mean_similarity
        results['std_similarity'] = std_similarity
        results['passed'] = mean_similarity > 0.7
        
        self.results['tests']['temporal_consistency'] = results
        
        status = "✓ PASS" if results['passed'] else "✗ FAIL"
        print(f"  {status} Temporal consistency: {mean_similarity:.3f} ± {std_similarity:.3f}")
        
        # Visualize
        self._plot_temporal_consistency(similarities)
    
    def test_agency_detection(self):
        """Test agency detection accuracy"""
        results = {}
        
        self.model.eval()
        
        # Test multiple scenarios
        agency_scores = {
            'own_action': [],
            'external': []
        }
        
        for trial in range(20):
            # Own action
            obs = torch.randn(1, 512).cuda()
            action = torch.randn(1, 64).cuda()
            self.model.step(obs)
            next_obs = obs + 0.15 * torch.randn_like(obs)
            _, content, _ = self.model.step(next_obs, prev_action=action)
            agency_scores['own_action'].append(float(content['agency_signal'].mean()))
            
            # External change
            obs2 = torch.randn(1, 512).cuda()
            self.model.step(obs2)
            next_obs2 = obs2 + 0.5 * torch.randn_like(obs2)
            _, content2, _ = self.model.step(next_obs2, prev_action=torch.zeros_like(action))
            agency_scores['external'].append(float(content2['agency_signal'].mean()))
        
        mean_own = np.mean(agency_scores['own_action'])
        mean_external = np.mean(agency_scores['external'])
        
        discrimination = mean_own - mean_external
        
        results['mean_own_agency'] = mean_own
        results['mean_external_agency'] = mean_external
        results['discrimination'] = discrimination
        results['passed'] = discrimination > 0.2
        
        self.results['tests']['agency_detection'] = results
        
        status = "✓ PASS" if results['passed'] else "✗ FAIL"
        print(f"  {status} Agency discrimination: {discrimination:.3f}")
        print(f"    Own: {mean_own:.3f}, External: {mean_external:.3f}")
        
        # Visualize
        self._plot_agency_detection(agency_scores)
    
    def test_consciousness_integration(self):
        """Test consciousness integration (Φ)"""
        results = {}
        
        self.model.eval()
        
        phi_scores = []
        integration_scores = []
        
        for trial in range(30):
            obs = torch.randn(1, 512).cuda()
            _, conscious_content, metrics = self.model.step(obs)
            
            phi_scores.append(metrics['phi'])
            integration_scores.append(metrics['integration_score'])
        
        mean_phi = np.mean(phi_scores)
        mean_integration = np.mean(integration_scores)
        
        results['mean_phi'] = mean_phi
        results['mean_integration'] = mean_integration
        results['passed'] = mean_phi > 0.4 and mean_integration > 0.5
        
        self.results['tests']['consciousness_integration'] = results
        
        status = "✓ PASS" if results['passed'] else "✗ FAIL"
        print(f"  {status} Consciousness integration")
        print(f"    Φ: {mean_phi:.3f}")
        print(f"    Integration: {mean_integration:.3f}")
        
        # Visualize
        self._plot_integration_scores(phi_scores, integration_scores)
    
    def _plot_temporal_consistency(self, similarities):
        """Plot temporal consistency"""
        plt.figure(figsize=(10, 5))
        plt.plot(similarities, alpha=0.7)
        plt.axhline(np.mean(similarities), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(similarities):.3f}')
        plt.xlabel('Time Step')
        plt.ylabel('Self-State Similarity')
        plt.title('Temporal Consistency of Self-Model')
        plt.legend()
        plt.grid(alpha=0.3)
        
        save_path = self.save_dir / 'temporal_consistency.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.results['visualizations'].append(str(save_path))
    
    def _plot_agency_detection(self, agency_scores):
        """Plot agency detection"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data = [agency_scores['own_action'], agency_scores['external']]
        labels = ['Own Action', 'External']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Agency Signal')
        ax.set_title('Agency Detection: Own vs External')
        ax.grid(alpha=0.3, axis='y')
        
        save_path = self.save_dir / 'agency_detection.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.results['visualizations'].append(str(save_path))
    
    def _plot_integration_scores(self, phi_scores, integration_scores):
        """Plot integration scores"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Phi distribution
        ax1.hist(phi_scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(phi_scores), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(phi_scores):.3f}')
        ax1.set_xlabel('Φ (Integrated Information)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Φ')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Integration scores
        ax2.hist(integration_scores, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(np.mean(integration_scores), color='r', linestyle='--',
                   label=f'Mean: {np.mean(integration_scores):.3f}')
        ax2.set_xlabel('Integration Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Integration Scores')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        save_path = self.save_dir / 'integration_scores.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.results['visualizations'].append(str(save_path))
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        
        # Calculate overall score
        test_results = self.results['tests']
        passed_tests = sum(
            1 for category in test_results.values()
            for test in (category.values() if isinstance(category, dict) else [category])
            if isinstance(test, dict) and test.get('passed', False)
        )
        
        total_tests = sum(
            len([t for t in category.values() if isinstance(t, dict) and 'passed' in t])
            if isinstance(category, dict) else 1
            for category in test_results.values()
        )
        
        overall_score = passed_tests / total_tests if total_tests > 0 else 0
        
        report = {
            'overall_score': overall_score,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'test_results': test_results,
            'visualizations': self.results['visualizations']
        }
        
        # Save report
        report_path = self.save_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Overall Score: {overall_score*100:.1f}%")
        print(f"Passed Tests: {passed_tests}/{total_tests}")
        print(f"\nReport saved to: {report_path}")
        print(f"Visualizations saved to: {self.save_dir}")
        
        return report


class ConsciousnessVisualizer:
    """
    Визуализация сознательного опыта системы
    """
    
    def __init__(self, save_dir='visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
    
    def visualize_workspace(self, workspace: torch.Tensor, step: int):
        """
        Визуализация глобального рабочего пространства
        
        Args:
            workspace: [workspace_capacity, hidden_dim]
            step: текущий шаг
        """
        workspace_np = workspace.detach().cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        
        # Heatmap
        plt.subplot(1, 2, 1)
        sns.heatmap(workspace_np[:, :64], cmap='viridis', cbar=True)
        plt.title(f'Global Workspace at Step {step}')
        plt.xlabel('Hidden Dimension')
        plt.ylabel('Workspace Slot')
        
        # Attention distribution
        plt.subplot(1, 2, 2)
        attention = np.linalg.norm(workspace_np, axis=1)
        plt.bar(range(len(attention)), attention, color='skyblue')
        plt.xlabel('Workspace Slot')
        plt.ylabel('Activation Magnitude')
        plt.title('Workspace Utilization')
        plt.grid(alpha=0.3)
        
        save_path = self.save_dir / f'workspace_step_{step}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_neurochemistry(self, neurochemistry: torch.Tensor, step: int):
        """
        Визуализация нейрохимии
        """
        neuro_np = neurochemistry.detach().cpu().numpy().flatten()
        
        transmitter_names = [
            'Dopamine', 'Serotonin', 'Oxytocin', 'Cortisol',
            'Norepinephrine', 'GABA', 'Glutamate', 'Acetylcholine'
        ] + [f'NT{i}' for i in range(8, len(neuro_np))]
        
        plt.figure(figsize=(12, 6))
        colors = ['red' if x > 0.6 else 'orange' if x > 0.4 else 'blue' 
                 for x in neuro_np[:8]]
        
        plt.bar(range(len(neuro_np[:8])), neuro_np[:8], color=colors, alpha=0.7)
        plt.xticks(range(8), transmitter_names[:8], rotation=45, ha='right')
        plt.ylabel('Level')
        plt.title(f'Neurochemistry at Step {step}')
        plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        plt.grid(alpha=0.3, axis='y')
        
        save_path = self.save_dir / f'neurochemistry_step_{step}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_consciousness_flow(self, history: List[Dict]):
        """
        Визуализация потока сознания во времени
        """
        if len(history) == 0:
            return
        
        # Extract metrics
        steps = list(range(len(history)))
        phi = [h['phi'] for h in history]
        integration = [h['integration_score'] for h in history]
        agency = [h['agency'] for h in history]
        confidence = [h['confidence'] for h in history]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Phi over time
        axes[0, 0].plot(steps, phi, marker='o', markersize=3, alpha=0.7)
        axes[0, 0].set_ylabel('Φ (Integrated Information)')
        axes[0, 0].set_title('Consciousness Integration')
        axes[0, 0].grid(alpha=0.3)
        
        # Integration score
        axes[0, 1].plot(steps, integration, marker='o', markersize=3, 
                       alpha=0.7, color='orange')
        axes[0, 1].set_ylabel('Integration Score')
        axes[0, 1].set_title('Information Integration')
        axes[0, 1].grid(alpha=0.3)
        
        # Agency
        axes[1, 0].plot(steps, agency, marker='o', markersize=3, 
                       alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Agency Signal')
        axes[1, 0].set_title('Sense of Agency')
        axes[1, 0].grid(alpha=0.3)
        
        # Confidence
        axes[1, 1].plot(steps, confidence, marker='o', markersize=3, 
                       alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Meta-Confidence')
        axes[1, 1].set_title('Meta-Cognition')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / 'consciousness_flow.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    print("Evaluation utilities loaded.")
    print("Use SelfAwarenessEvaluator to test your model.")
