"""
Integrated Evaluation Framework
Combines the comprehensive evaluation with enhanced interpretability tools
"""

import os
import sys
import json
import numpy as np
import torch
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from src.evaluation.framework import EvaluationFramework
from src.evaluation.interpretability import InterpretabilityAnalyzer
from scripts.evaluate_model import EvaluationFramework as ComprehensiveEvaluator

class IntegratedEvaluationFramework:
    """
    Integrated framework combining all evaluation approaches
    """

    def __init__(self, model_path, num_episodes=100, output_dir='evaluation_results'):
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.output_dir = output_dir

        # Initialize comprehensive evaluator
        self.comprehensive_evaluator = ComprehensiveEvaluator(model_path, num_episodes)

        # Storage for collected data
        self.collected_data = None

    def run_comprehensive_evaluation(self):
        """Run the comprehensive evaluation from evaluate_model.py"""
        print("=== Running Comprehensive Evaluation ===")

        # Collect episode data
        self.comprehensive_evaluator.collect_episode_data()

        # Store data for other analyses
        self.collected_data = self.comprehensive_evaluator.episode_data

        # Run comprehensive analysis
        nmi_scores = self.comprehensive_evaluator.compute_nmi_scores()
        consistency_results = self.comprehensive_evaluator.consistency_tests()
        behavioral_metrics = self.comprehensive_evaluator.behavioral_analysis()

        # Create visualizations
        self.comprehensive_evaluator.visualize_communication_patterns(self.output_dir)

        return {
            'nmi_scores': nmi_scores,
            'consistency_results': consistency_results,
            'behavioral_metrics': behavioral_metrics
        }

    def run_interpretability_analysis(self):
        """Run advanced interpretability analysis"""
        print("=== Running Interpretability Analysis ===")

        if self.collected_data is None:
            print("No data available. Running comprehensive evaluation first...")
            self.run_comprehensive_evaluation()

        # Extract data for interpretability analysis
        all_messages = []
        all_actions = []
        all_states = []
        all_rewards = []

        for episode in self.collected_data:
            all_messages.extend(episode.get('messages', []))
            all_actions.extend(episode.get('actions', []))
            all_rewards.extend(episode.get('rewards', []))

            # Convert states to numpy arrays
            episode_states = episode.get('states', [])
            for state in episode_states:
                if hasattr(state, 'numpy'):
                    all_states.append(state.numpy().flatten())
                elif hasattr(state, 'detach'):
                    all_states.append(state.detach().cpu().numpy().flatten())
                else:
                    all_states.append(np.array(state).flatten())

        # Initialize interpretability analyzer
        analyzer = InterpretabilityAnalyzer(
            messages=all_messages,
            actions=all_actions,
            states=all_states if all_states else None,
            rewards=all_rewards if all_rewards else None
        )

        # Generate interpretability report
        interpretability_report = analyzer.generate_interpretability_report()

        # Create interpretability visualizations
        viz_path = os.path.join(self.output_dir, 'interpretability_analysis.png')
        analyzer.visualize_interpretability(save_path=viz_path)

        return interpretability_report

    def run_basic_framework_evaluation(self):
        """Run evaluation using the enhanced basic framework"""
        print("=== Running Basic Framework Evaluation ===")

        # Create a simple data loader from collected data
        class SimpleDataLoader:
            def __init__(self, episode_data):
                self.data = []
                for episode in episode_data:
                    messages = episode.get('messages', [])
                    actions = episode.get('actions', [])
                    states = episode.get('states', [])

                    for msg, action, state in zip(messages, actions, states):
                        self.data.append({
                            'inputs': torch.tensor(state) if not torch.is_tensor(state) else state,
                            'targets': action,
                            'message': torch.tensor(msg) if not torch.is_tensor(msg) else msg
                        })

            def __iter__(self):
                return iter(self.data)

        if self.collected_data is None:
            print("No data available. Running comprehensive evaluation first...")
            self.run_comprehensive_evaluation()

        # Create data loader
        data_loader = SimpleDataLoader(self.collected_data)

        # Create a mock model for the basic framework
        class MockModel:
            def __call__(self, inputs):
                # Return mock outputs - in practice, this would be your actual model
                batch_size = len(inputs) if isinstance(inputs, list) else inputs.shape[0]
                messages = torch.randn(batch_size, 3)  # Mock messages
                actions = torch.randint(0, 5, (batch_size,))  # Mock actions
                return messages, actions

        mock_model = MockModel()

        # Initialize basic framework
        basic_evaluator = EvaluationFramework(mock_model, data_loader)

        # Run evaluation
        basic_results = basic_evaluator.evaluate()

        return basic_results

    def generate_integrated_report(self):
        """Generate comprehensive integrated report"""
        print("=== Generating Integrated Report ===")

        os.makedirs(self.output_dir, exist_ok=True)

        # Run all evaluations
        comprehensive_results = self.run_comprehensive_evaluation()
        interpretability_results = self.run_interpretability_analysis()
        basic_results = self.run_basic_framework_evaluation()

        # Compile integrated report
        integrated_report = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'num_episodes': self.num_episodes,
                'output_directory': self.output_dir
            },
            'comprehensive_evaluation': comprehensive_results,
            'interpretability_analysis': interpretability_results,
            'basic_framework_results': basic_results,
            'integrated_summary': self._compute_integrated_summary(
                comprehensive_results, interpretability_results, basic_results
            )
        }

        # Save integrated report
        report_path = os.path.join(self.output_dir, 'integrated_evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(integrated_report, f, indent=2, default=str)

        # Generate human-readable summary
        self._generate_human_readable_summary(integrated_report)

        print(f"Integrated evaluation report saved to {self.output_dir}/")
        return integrated_report

    def _compute_integrated_summary(self, comprehensive, interpretability, basic):
        """Compute integrated summary metrics"""
        summary = {}

        # Performance metrics
        behavioral = comprehensive.get('behavioral_metrics', {})
        summary['performance'] = {
            'success_rate': behavioral.get('success_rate', 0),
            'avg_reward': behavioral.get('avg_total_reward', 0),
            'episode_efficiency': behavioral.get('avg_episode_length', 50) / 50  # Normalized
        }

        # Communication effectiveness
        nmi_scores = comprehensive.get('nmi_scores', {})
        interp_summary = interpretability.get('summary_scores', {})

        communication_scores = []
        if nmi_scores:
            communication_scores.extend(nmi_scores.values())
        if 'communication_quality' in interp_summary:
            communication_scores.append(interp_summary['communication_quality'])

        summary['communication_effectiveness'] = np.mean(communication_scores) if communication_scores else 0

        # Interpretability
        summary['interpretability'] = {
            'overall_score': interp_summary.get('overall_interpretability', 0),
            'emergence_strength': interp_summary.get('emergence_strength', 0),
            'compositionality': interpretability.get('compositionality_analysis', {}).get('compositionality_score', 0)
        }

        # Consistency
        consistency = comprehensive.get('consistency_results', {})
        basic_consistency = basic.get('consistency_metrics', {})

        all_consistency_scores = []
        all_consistency_scores.extend(consistency.values())
        all_consistency_scores.extend(basic_consistency.values())

        summary['consistency'] = np.mean(all_consistency_scores) if all_consistency_scores else 0

        # Overall quality score
        summary['overall_quality'] = np.mean([
            summary['performance']['success_rate'],
            summary['communication_effectiveness'],
            summary['interpretability']['overall_score'],
            summary['consistency']
        ])

        return summary

    def _generate_human_readable_summary(self, report):
        """Generate human-readable summary"""
        summary = report['integrated_summary']

        summary_text = f"""
INTEGRATED EMERGENT COMMUNICATION EVALUATION REPORT
==================================================
Generated: {report['evaluation_metadata']['timestamp']}
Model: {report['evaluation_metadata']['model_path']}
Episodes Evaluated: {report['evaluation_metadata']['num_episodes']}

OVERALL ASSESSMENT
------------------
Overall Quality Score: {summary['overall_quality']:.3f}
Performance Rating: {'Excellent' if summary['overall_quality'] > 0.8 else 'Good' if summary['overall_quality'] > 0.6 else 'Moderate' if summary['overall_quality'] > 0.4 else 'Needs Improvement'}

PERFORMANCE METRICS
-------------------
Success Rate: {summary['performance']['success_rate']:.2%}
Average Reward: {summary['performance']['avg_reward']:.2f}
Episode Efficiency: {summary['performance']['episode_efficiency']:.2%}

COMMUNICATION ANALYSIS
----------------------
Communication Effectiveness: {summary['communication_effectiveness']:.3f}
Behavioral Consistency: {summary['consistency']:.3f}

INTERPRETABILITY INSIGHTS
-------------------------
Overall Interpretability: {summary['interpretability']['overall_score']:.3f}
Emergence Strength: {summary['interpretability']['emergence_strength']:.3f}
Compositional Structure: {summary['interpretability']['compositionality']:.3f}

RECOMMENDATIONS
---------------
"""

        # Add specific recommendations based on scores
        if summary['performance']['success_rate'] < 0.3:
            summary_text += "• Consider extending training duration or adjusting reward structure\n"

        if summary['communication_effectiveness'] < 0.3:
            summary_text += "• Communication protocol may need restructuring or additional training\n"

        if summary['interpretability']['compositionality'] < 0.2:
            summary_text += "• Consider architectural changes to encourage compositional communication\n"

        if summary['consistency'] < 0.5:
            summary_text += "• Focus on improving behavioral and communication consistency\n"

        if summary['interpretability']['emergence_strength'] < 0.3:
            summary_text += "• May need longer training or different incentives for emergent properties\n"

        if summary['overall_quality'] > 0.7:
            summary_text += "• System shows strong emergent communication properties\n"
            summary_text += "• Consider testing on more complex tasks or environments\n"

        # Save summary
        summary_path = os.path.join(self.output_dir, 'integrated_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_text)

        print("\nINTEGRATED EVALUATION SUMMARY:")
        print(f"Overall Quality: {summary['overall_quality']:.3f}")
        print(f"Success Rate: {summary['performance']['success_rate']:.2%}")
        print(f"Communication Effectiveness: {summary['communication_effectiveness']:.3f}")
        print(f"Interpretability Score: {summary['interpretability']['overall_score']:.3f}")

def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description='Run integrated evaluation of emergent communication model')
    parser.add_argument('--model_path', type=str,
                       default='results/architect_builder_v1/models/final_model.pt',
                       help='Path to trained model')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes to evaluate')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Available model files:")
        for root, dirs, files in os.walk('results'):
            for file in files:
                if file.endswith('.pt'):
                    print(f"  {os.path.join(root, file)}")
        return

    # Run integrated evaluation
    evaluator = IntegratedEvaluationFramework(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir
    )

    report = evaluator.generate_integrated_report()
    print(f"\nEvaluation completed! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()