#!/usr/bin/env python3
"""
Generalization Tests Framework for Emergent Communication
Zero-shot evaluation on novel blueprints and environments
"""

import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from config.config import config
from src.environments.environment import ArchitectBuilderEnv
from src.agents.agents import Architect, Builder
from src.evaluation.framework import EvaluationFramework

class GeneralizationTestFramework:
    """
    Framework for testing generalization capabilities of trained models
    """

    def __init__(self, model_path, output_dir='generalization_results'):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load trained models
        self.load_models()

        # Define test scenarios
        self.test_scenarios = self._define_test_scenarios()

        # Results storage
        self.results = {}

    def load_models(self):
        """Load trained models"""
        print(f"📦 Loading models from {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Get config from checkpoint or use default
        model_config = checkpoint.get('config', config)

        # Initialize models
        self.architect = Architect(
            grid_size=model_config.get('grid_size', 8),
            input_channels=6,
            hidden_dim=model_config.get('architect_hidden_dim', 64),
            vocab_size=model_config.get('vocab_size', 8),
            message_length=model_config.get('message_length', 3)
        ).to(self.device)

        self.builder = Builder(
            grid_size=model_config.get('grid_size', 8),
            input_channels=6,
            hidden_dim=model_config.get('builder_hidden_dim', 64),
            vocab_size=model_config.get('vocab_size', 8),
            message_length=model_config.get('message_length', 3),
            action_dim=model_config.get('action_dim', 5)
        ).to(self.device)

        # Load state dicts
        self.architect.load_state_dict(checkpoint['architect_state_dict'], strict=False)
        self.builder.load_state_dict(checkpoint['builder_state_dict'], strict=False)

        # Set to evaluation mode
        self.architect.eval()
        self.builder.eval()

        print("✅ Models loaded successfully!")

    def _define_test_scenarios(self):
        """Define different generalization test scenarios"""
        scenarios = {}

        # 1. Grid Size Generalization
        scenarios['grid_size'] = {
            'smaller_grid': {
                'grid_size': 3,
                'block_colors': ['red', 'green'],
                'max_blocks_per_color': 1,
                'description': 'Smaller 3x3 grid'
            },
            'larger_grid': {
                'grid_size': 6,
                'block_colors': ['red', 'green'],
                'max_blocks_per_color': 1,
                'description': 'Larger 6x6 grid'
            },
            'much_larger_grid': {
                'grid_size': 8,
                'block_colors': ['red', 'green'],
                'max_blocks_per_color': 1,
                'description': 'Much larger 8x8 grid'
            }
        }

        # 2. Color Generalization
        scenarios['colors'] = {
            'new_single_color': {
                'grid_size': 4,
                'block_colors': ['blue'],  # New color not seen in training
                'max_blocks_per_color': 1,
                'description': 'Novel single color (blue)'
            },
            'new_color_pair': {
                'grid_size': 4,
                'block_colors': ['blue', 'yellow'],  # New color combination
                'max_blocks_per_color': 1,
                'description': 'Novel color pair (blue, yellow)'
            },
            'more_colors': {
                'grid_size': 4,
                'block_colors': ['red', 'green', 'blue'],  # More colors than training
                'max_cks_per_color': 1,
                'description': 'More colors than training'
            },
            'many_colors': {
                'grid_size': 4,
                'block_colors': ['red', 'green', 'blue', 'yellow', 'purple'],
                'max_blocks_per_color': 1,
                'description': 'Many colors (5 total)'
            }
        }

        # 3. Complexity Generalization
        scenarios['complexity'] = {
            'more_blocks': {
                'grid_size': 4,
                'block_colors': ['red', 'green'],
                'max_blocks_per_color': 2,  # More blocks than training
                'description': 'More blocks per color'
            },
            'many_blocks': {
                'grid_size': 5,
                'block_colors': ['red', 'green', 'blue'],
                'max_blocks_per_color': 2,
                'description': 'Many blocks, multiple colors'
            },
            'dense_environment': {
                'grid_size': 4,
                'block_colors': ['red', 'green'],
                'max_blocks_per_color': 3,
                'description': 'Dense environment (many blocks)'
            }
        }

        # 4. Pattern Generalization
        scenarios['patterns'] = {
            'line_patterns': {
                'grid_size': 4,
                'block_colors': ['red', 'green'],
                'max_blocks_per_color': 1,
                'pattern_type': 'line',
                'description': 'Line pattern blueprints'
            },
            'corner_patterns': {
                'grid_size': 4,
                'block_colors': ['red', 'green'],
                'max_blocks_per_color': 1,
                'pattern_type': 'corner',
                'description': 'Corner pattern blueprints'
            },
            'symmetric_patterns': {
                'grid_size': 4,
                'block_colors': ['red', 'green'],
                'max_blocks_per_color': 1,
                'pattern_type': 'symmetric',
                'description': 'Symmetric pattern blueprints'
            }
        }

        # 5. Episode Length Generalization
        scenarios['episode_length'] = {
            'shorter_episodes': {
                'grid_size': 4,
                'block_colors': ['red', 'green'],
                'max_blocks_per_color': 1,
                'max_episode_steps': 25,  # Shorter than training
                'description': 'Shorter episodes (25 steps)'
            },
            'longer_episodes': {
                'grid_size': 4,
                'block_colors': ['red', 'green'],
                'max_blocks_per_color': 1,
                'max_episode_steps': 100,  # Longer than training
                'description': 'Longer episodes (100 steps)'
            }
        }

        return scenarios

    def generate_novel_blueprints(self, scenario_config, num_blueprints=20):
        """Generate novel blueprints for testing"""
        blueprints = []

        grid_size = scenario_config['grid_size']
        colors = scenario_config['block_colors']
        max_blocks = scenario_config['max_blocks_per_color']
        pattern_type = scenario_config.get('pattern_type', 'random')

        for _ in range(num_blueprints):
            blueprint = np.zeros((grid_size, grid_size), dtype=int)

            if pattern_type == 'line':
                # Create line patterns
                if np.random.random() < 0.5:  # Horizontal line
                    row = np.random.randint(0, grid_size)
                    start_col = np.random.randint(0, grid_size - 2)
                    end_col = min(start_col + np.random.randint(2, 4), grid_size)
                    for col in range(start_col, end_col):
                        if col < grid_size:
                            blueprint[row, col] = np.random.randint(1, len(colors) + 1)
                else:  # Vertical line
                    col = np.random.randint(0, grid_size)
                    start_row = np.random.randint(0, grid_size - 2)
                    end_row = min(start_row + np.random.randint(2, 4), grid_size)
                    for row in range(start_row, end_row):
                        if row < grid_size:
                            blueprint[row, col] = np.random.randint(1, len(colors) + 1)

            elif pattern_type == 'corner':
                # Create corner patterns
                corner = np.random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])
                size = np.random.randint(2, min(4, grid_size))

                if corner == 'top-left':
                    blueprint[:size, :size] = np.random.randint(1, len(colors) + 1, (size, size))
                elif corner == 'top-right':
                    blueprint[:size, -size:] = np.random.randint(1, len(colors) + 1, (size, size))
                elif corner == 'bottom-left':
                    blueprint[-size:, :size] = np.random.randint(1, len(colors) + 1, (size, size))
                else:  # bottom-right
                    blueprint[-size:, -size:] = np.random.randint(1, len(colors) + 1, (size, size))

            elif pattern_type == 'symmetric':
                # Create symmetric patterns
                half_size = grid_size // 2
                left_half = np.random.randint(0, len(colors) + 1, (grid_size, half_size))
                blueprint[:, :half_size] = left_half
                blueprint[:, half_size:] = np.fliplr(left_half[:, :grid_size - half_size])

            else:  # Random patterns
                # Place blocks randomly
                total_blocks = min(max_blocks * len(colors), grid_size * grid_size // 2)
                positions = np.random.choice(grid_size * grid_size, total_blocks, replace=False)

                for i, pos in enumerate(positions):
                    row, col = pos // grid_size, pos % grid_size
                    color_idx = (i % len(colors)) + 1
                    blueprint[row, col] = color_idx

            blueprints.append(blueprint)

        return blueprints

    def test_scenario(self, scenario_name, scenario_config, num_episodes=50):
        """Test a specific generalization scenario"""
        print(f"\n🧪 Testing scenario: {scenario_name}")
        print(f"📋 Description: {scenario_config['description']}")

        # Generate novel blueprints
        blueprints = self.generate_novel_blueprints(scenario_config, num_episodes)

        # Initialize environment with scenario config
        env = ArchitectBuilderEnv(
            grid_size=scenario_config['grid_size'],
            block_colors=scenario_config['block_colors'],
            max_blocks_per_color=scenario_config['max_blocks_per_color'],
            max_episode_steps=scenario_config.get('max_episode_steps', 50)
        )

        # Test results storage
        episode_results = []
        communication_data = []

        # Run episodes
        progress_bar = tqdm(range(num_episodes), desc=f"Testing {scenario_name}")

        for episode in progress_bar:
            # Set custom blueprint
            env.blueprint = blueprints[episode]

            # Reset environment
            obs = env.reset()
            done = False
            step = 0

            episode_reward = 0
            episode_messages = []
            episode_actions = []
            episode_states = []

            with torch.no_grad():
                while not done and step < scenario_config.get('max_episode_steps', 50):
                    # Get architect's observation and message
                    arch_obs = obs['architect'].unsqueeze(0).to(self.device)
                    message, _, _ = self.architect(arch_obs)

                    # Get builder's action
                    builder_obs = obs['builder'].unsqueeze(0).to(self.device)
                    carrying_state = torch.tensor([obs.get('builder_carrying', 0)]).to(self.device)
                    action_logits, _, _ = self.builder(builder_obs, message, carrying_state)
                    action = torch.argmax(action_logits, dim=-1).item()

                    # Step environment
                    obs, reward, done, info = env.step(action)

                    # Store data
                    episode_messages.append(message.cpu().numpy().flatten())
                    episode_actions.append(action)
                    episode_states.append(obs['builder'].clone().detach())
                    episode_reward += reward
                    step += 1

            # Check success
            success = env._is_blueprint_complete() or episode_reward > 500

            # Store episode results
            episode_results.append({
                'episode': episode,
                'success': success,
                'total_reward': episode_reward,
                'steps': step,
                'blueprint_complexity': np.sum(blueprints[episode] > 0)  # Number of blocks
            })

            # Store communication data
            communication_data.append({
                'messages': episode_messages,
                'actions': episode_actions,
                'states': episode_states
            })

            # Update progress
            if episode % 10 == 0:
                recent_success = np.mean([r['success'] for r in episode_results[-10:]])
                progress_bar.set_postfix({
                    'Success': f"{recent_success:.1%}",
                    'Reward': f"{episode_reward:.1f}"
                })

        # Analyze results
        success_rate = np.mean([r['success'] for r in episode_results])
        avg_reward = np.mean([r['total_reward'] for r in episode_results])
        avg_steps = np.mean([r['steps'] for r in episode_results])

        # Communication analysis
        comm_analysis = self._analyze_communication(communication_data)

        results = {
            'scenario_name': scenario_name,
            'scenario_config': scenario_config,
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'episode_results': episode_results,
            'communication_analysis': comm_analysis,
            'num_episodes': num_episodes
        }

        print(f"✅ Scenario completed!")
        print(f"📊 Success Rate: {success_rate:.1%}")
        print(f"🎯 Avg Reward: {avg_reward:.2f}")
        print(f"⏱️ Avg Steps: {avg_steps:.1f}")

        return results

    def _analyze_communication(self, communication_data):
        """Analyze communication patterns in generalization tests"""
        if not communication_data:
            return {}

        # Collect all messages and actions
        all_messages = []
        all_actions = []

        for episode_data in communication_data:
            all_messages.extend(episode_data['messages'])
            all_actions.extend(episode_data['actions'])

        if not all_messages or not all_actions:
            return {}

        # Message diversity
        messages_array = np.array(all_messages)
        unique_messages = len(set(map(tuple, messages_array.round(2))))
        message_diversity = unique_messages / len(all_messages)

        # Action diversity
        unique_actions = len(set(all_actions))
        action_diversity = unique_actions / 5  # 5 possible actions

        # Message-action correlation (simplified NMI)
        try:
            from sklearn.metrics import normalized_mutual_info_score
            from sklearn.cluster import KMeans

            # Validate data before clustering
            if messages_array.size == 0 or len(all_actions) == 0:
                message_action_nmi = 0.0
            else:
                # Check for valid data
                if not np.all(np.isfinite(messages_array)):
                    # Replace invalid values with zeros
                    messages_array = np.nan_to_num(messages_array, nan=0.0, posinf=0.0, neginf=0.0)

                # Cluster messages for NMI computation
                n_clusters = min(8, unique_messages)
                if n_clusters > 1 and len(messages_array) > n_clusters:
                    # Use more robust KMeans parameters
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10,
                                  max_iter=100, tol=1e-4, algorithm='lloyd')

                    # Suppress warnings during clustering
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        message_clusters = kmeans.fit_predict(messages_array)

                    # Compute NMI with error handling
                    message_action_nmi = normalized_mutual_info_score(message_clusters, all_actions)

                    # Ensure result is finite
                    if not np.isfinite(message_action_nmi):
                        message_action_nmi = 0.0
                else:
                    message_action_nmi = 0.0

        except Exception as e:
            print(f"Warning: Communication analysis failed: {e}")
            message_action_nmi = 0.0

        return {
            'message_diversity': message_diversity,
            'action_diversity': action_diversity,
            'message_action_nmi': message_action_nmi,
            'total_messages': len(all_messages),
            'unique_messages': unique_messages
        }

    def run_all_generalization_tests(self, episodes_per_scenario=50):
        """Run all generalization test scenarios"""
        print("🚀 Starting comprehensive generalization tests...")

        all_results = {}

        for category, scenarios in self.test_scenarios.items():
            print(f"\n{'='*60}")
            print(f"📋 Category: {category.upper()}")
            print(f"{'='*60}")

            category_results = {}

            for scenario_name, scenario_config in scenarios.items():
                try:
                    result = self.test_scenario(f"{category}_{scenario_name}", scenario_config, episodes_per_scenario)
                    category_results[scenario_name] = result
                except Exception as e:
                    print(f"❌ Failed to test {scenario_name}: {e}")
                    category_results[scenario_name] = {'error': str(e)}

            all_results[category] = category_results

        # Save results
        self.results = all_results
        results_file = self.output_dir / 'generalization_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n✅ All generalization tests completed!")
        print(f"📁 Results saved to: {self.output_dir}")

        return all_results

    def analyze_generalization_results(self, results=None):
        """Analyze and visualize generalization test results"""
        if results is None:
            results = self.results

        if not results:
            print("❌ No results to analyze. Run generalization tests first.")
            return

        print("📊 Analyzing generalization test results...")

        # Create analysis directory
        analysis_dir = self.output_dir / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Overall generalization performance
        self._create_generalization_overview(results, analysis_dir)

        # Category-specific analysis
        for category, category_results in results.items():
            self._analyze_generalization_category(category, category_results, analysis_dir)

        # Communication analysis across scenarios
        self._analyze_communication_generalization(results, analysis_dir)

        # Generate summary report
        self._generate_generalization_report(results, analysis_dir)

        print(f"📈 Analysis completed! Visualizations saved to: {analysis_dir}")

    def _create_generalization_overview(self, results, analysis_dir):
        """Create overview visualization of generalization performance"""
        # Collect data for overview
        scenario_names = []
        success_rates = []
        avg_rewards = []
        categories = []

        for category, category_results in results.items():
            for scenario_name, result in category_results.items():
                if 'error' not in result:
                    scenario_names.append(f"{category}_{scenario_name}")
                    success_rates.append(result['success_rate'] * 100)  # Convert to percentage
                    avg_rewards.append(result['avg_reward'])
                    categories.append(category)

        if not scenario_names:
            print("⚠️ No valid results for overview")
            return

        # Create overview visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Generalization Test Overview', fontsize=16, fontweight='bold')

        # Success rates by scenario
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(categories))))
        category_colors = {cat: colors[i] for i, cat in enumerate(set(categories))}
        bar_colors = [category_colors[cat] for cat in categories]

        axes[0, 0].bar(range(len(scenario_names)), success_rates, color=bar_colors, alpha=0.7)
        axes[0, 0].set_title('Success Rate by Scenario')
        axes[0, 0].set_xticks(range(len(scenario_names)))
        axes[0, 0].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)

        # Average rewards by scenario
        axes[0, 1].bar(range(len(scenario_names)), avg_rewards, color=bar_colors, alpha=0.7)
        axes[0, 1].set_title('Average Reward by Scenario')
        axes[0, 1].set_xticks(range(len(scenario_names)))
        axes[0, 1].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)

        # Success rate by category (box plot)
        category_success_data = defaultdict(list)
        for cat, success in zip(categories, success_rates):
            category_success_data[cat].append(success)

        axes[1, 0].boxplot([category_success_data[cat] for cat in category_success_data.keys()],
                          labels=list(category_success_data.keys()))
        axes[1, 0].set_title('Success Rate Distribution by Category')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # Reward vs Success Rate scatter
        axes[1, 1].scatter(success_rates, avg_rewards, c=[category_colors[cat] for cat in categories], alpha=0.7)
        axes[1, 1].set_title('Reward vs Success Rate')
        axes[1, 1].set_xlabel('Success Rate (%)')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].grid(True, alpha=0.3)

        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=category_colors[cat], alpha=0.7, label=cat.title())
                          for cat in category_colors.keys()]
        axes[1, 1].legend(handles=legend_elements, loc='best')

        plt.tight_layout()
        plt.savefig(analysis_dir / 'generalization_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _analyze_generalization_category(self, category, category_results, analysis_dir):
        """Analyze results for a specific generalization category"""
        # Extract valid results
        valid_results = {name: result for name, result in category_results.items() if 'error' not in result}

        if not valid_results:
            print(f"⚠️ No valid results for category {category}")
            return

        # Create category-specific visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Generalization Analysis: {category.title()}', fontsize=16, fontweight='bold')

        scenario_names = list(valid_results.keys())
        success_rates = [result['success_rate'] * 100 for result in valid_results.values()]
        avg_rewards = [result['avg_reward'] for result in valid_results.values()]
        avg_steps = [result['avg_steps'] for result in valid_results.values()]

        # Success rates
        axes[0, 0].bar(range(len(scenario_names)), success_rates, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Success Rate (%)')
        axes[0, 0].set_xticks(range(len(scenario_names)))
        axes[0, 0].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)

        # Average rewards
        axes[0, 1].bar(range(len(scenario_names)), avg_rewards, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Average Reward')
        axes[0, 1].set_xticks(range(len(scenario_names)))
        axes[0, 1].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)

        # Episode steps
        axes[1, 0].bar(range(len(scenario_names)), avg_steps, color='orange', alpha=0.7)
        axes[1, 0].set_title('Average Episode Steps')
        axes[1, 0].set_xticks(range(len(scenario_names)))
        axes[1, 0].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True, alpha=0.3)

        # Communication effectiveness
        comm_scores = []
        for result in valid_results.values():
            comm_analysis = result.get('communication_analysis', {})
            nmi_score = comm_analysis.get('message_action_nmi', 0)
            comm_scores.append(nmi_score)

        axes[1, 1].bar(range(len(scenario_names)), comm_scores, color='purple', alpha=0.7)
        axes[1, 1].set_title('Communication Effectiveness (NMI)')
        axes[1, 1].set_xticks(range(len(scenario_names)))
        axes[1, 1].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Message-Action NMI')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(analysis_dir / f'{category}_generalization.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _analyze_communication_generalization(self, results, analysis_dir):
        """Analyze how communication patterns change across generalization scenarios"""
        # Collect communication data
        scenario_names = []
        message_diversities = []
        action_diversities = []
        nmi_scores = []

        for category, category_results in results.items():
            for scenario_name, result in category_results.items():
                if 'error' not in result:
                    comm_analysis = result.get('communication_analysis', {})
                    scenario_names.append(f"{category}_{scenario_name}")

                    # Get values with safe defaults and validation
                    msg_div = comm_analysis.get('message_diversity', 0)
                    act_div = comm_analysis.get('action_diversity', 0)
                    nmi_score = comm_analysis.get('message_action_nmi', 0)

                    # Ensure values are finite and valid
                    msg_div = msg_div if np.isfinite(msg_div) else 0.0
                    act_div = act_div if np.isfinite(act_div) else 0.0
                    nmi_score = nmi_score if np.isfinite(nmi_score) else 0.0

                    message_diversities.append(msg_div)
                    action_diversities.append(act_div)
                    nmi_scores.append(nmi_score)

        if not scenario_names:
            return

        # Create communication analysis visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Communication Patterns Across Generalization Scenarios', fontsize=16, fontweight='bold')

        # Message diversity
        axes[0, 0].bar(range(len(scenario_names)), message_diversities, color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Message Diversity')
        axes[0, 0].set_xticks(range(len(scenario_names)))
        axes[0, 0].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Diversity Score')
        axes[0, 0].grid(True, alpha=0.3)

        # Action diversity
        axes[0, 1].bar(range(len(scenario_names)), action_diversities, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Action Diversity')
        axes[0, 1].set_xticks(range(len(scenario_names)))
        axes[0, 1].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Diversity Score')
        axes[0, 1].grid(True, alpha=0.3)

        # Message-Action NMI
        axes[1, 0].bar(range(len(scenario_names)), nmi_scores, color='gold', alpha=0.7)
        axes[1, 0].set_title('Message-Action Correlation (NMI)')
        axes[1, 0].set_xticks(range(len(scenario_names)))
        axes[1, 0].set_xticklabels(scenario_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('NMI Score')
        axes[1, 0].grid(True, alpha=0.3)

        # Correlation between communication and performance
        success_rates = []
        for category, category_results in results.items():
            for scenario_name, result in category_results.items():
                if 'error' not in result:
                    success_rate = result.get('success_rate', 0) * 100
                    # Ensure success rate is finite and valid
                    success_rate = success_rate if np.isfinite(success_rate) else 0.0
                    success_rates.append(success_rate)

        axes[1, 1].scatter(nmi_scores, success_rates, alpha=0.7, color='darkgreen')
        axes[1, 1].set_title('Communication Quality vs Performance')
        axes[1, 1].set_xlabel('Message-Action NMI')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)

        # Add trend line with robust error handling
        if len(nmi_scores) > 1 and len(success_rates) > 1:
            try:
                # Clean data: remove NaN and infinite values
                nmi_array = np.array(nmi_scores)
                success_array = np.array(success_rates)

                # Create mask for valid data points
                valid_mask = (np.isfinite(nmi_array) & np.isfinite(success_array) &
                             (nmi_array != 0) & (success_array != 0))

                if np.sum(valid_mask) > 1:
                    clean_nmi = nmi_array[valid_mask]
                    clean_success = success_array[valid_mask]

                    # Check if there's enough variation in the data
                    if np.std(clean_nmi) > 1e-10 and np.std(clean_success) > 1e-10:
                        z = np.polyfit(clean_nmi, clean_success, 1)
                        p = np.poly1d(z)

                        # Plot trend line over the range of clean data
                        x_trend = np.linspace(np.min(clean_nmi), np.max(clean_nmi), 100)
                        axes[1, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8, label='Trend')
                        axes[1, 1].legend()
                    else:
                        # Add text indicating insufficient variation
                        axes[1, 1].text(0.5, 0.95, 'Insufficient data variation for trend line',
                                       transform=axes[1, 1].transAxes, ha='center', va='top',
                                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                else:
                    # Add text indicating insufficient valid data
                    axes[1, 1].text(0.5, 0.95, 'Insufficient valid data for trend analysis',
                                   transform=axes[1, 1].transAxes, ha='center', va='top',
                                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

            except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
                # Handle numerical errors gracefully
                print(f"Warning: Could not compute trend line due to numerical issues: {e}")
                axes[1, 1].text(0.5, 0.95, 'Trend analysis failed (numerical issues)',
                               transform=axes[1, 1].transAxes, ha='center', va='top',
                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

        plt.tight_layout()
        plt.savefig(analysis_dir / 'communication_generalization.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_generalization_report(self, results, analysis_dir):
        """Generate comprehensive generalization report"""
        # Calculate overall statistics
        all_success_rates = []
        all_rewards = []
        category_performance = {}

        for category, category_results in results.items():
            category_success_rates = []
            category_rewards = []

            for scenario_name, result in category_results.items():
                if 'error' not in result:
                    success_rate = result['success_rate']
                    avg_reward = result['avg_reward']

                    all_success_rates.append(success_rate)
                    all_rewards.append(avg_reward)
                    category_success_rates.append(success_rate)
                    category_rewards.append(avg_reward)

            if category_success_rates:
                category_performance[category] = {
                    'avg_success_rate': np.mean(category_success_rates),
                    'avg_reward': np.mean(category_rewards),
                    'num_scenarios': len(category_success_rates)
                }

        # Generate report text
        report_text = f"""
GENERALIZATION TEST REPORT
=========================
Generated: {datetime.now().isoformat()}
Model: {self.model_path}

OVERALL PERFORMANCE
------------------
Average Success Rate: {np.mean(all_success_rates):.2%}
Average Reward: {np.mean(all_rewards):.2f}
Total Scenarios Tested: {len(all_success_rates)}

PERFORMANCE BY CATEGORY
-----------------------
"""

        for category, perf in category_performance.items():
            report_text += f"""
{category.upper()}:
  Average Success Rate: {perf['avg_success_rate']:.2%}
  Average Reward: {perf['avg_reward']:.2f}
  Scenarios Tested: {perf['num_scenarios']}
"""

        # Find best and worst performing scenarios
        best_scenario = None
        worst_scenario = None
        best_success = -1
        worst_success = 2

        for category, category_results in results.items():
            for scenario_name, result in category_results.items():
                if 'error' not in result:
                    success_rate = result['success_rate']
                    if success_rate > best_success:
                        best_success = success_rate
                        best_scenario = (category, scenario_name, result)
                    if success_rate < worst_success:
                        worst_success = success_rate
                        worst_scenario = (category, scenario_name, result)

        if best_scenario and worst_scenario:
            report_text += f"""
BEST PERFORMING SCENARIO
-----------------------
Category: {best_scenario[0]}
Scenario: {best_scenario[1]}
Success Rate: {best_scenario[2]['success_rate']:.2%}
Average Reward: {best_scenario[2]['avg_reward']:.2f}
Description: {best_scenario[2]['scenario_config']['description']}

WORST PERFORMING SCENARIO
-------------------------
Category: {worst_scenario[0]}
Scenario: {worst_scenario[1]}
Success Rate: {worst_scenario[2]['success_rate']:.2%}
Average Reward: {worst_scenario[2]['avg_reward']:.2f}
Description: {worst_scenario[2]['scenario_config']['description']}
"""

        # Add insights and recommendations
        report_text += f"""
GENERALIZATION INSIGHTS
-----------------------
"""

        if np.mean(all_success_rates) > 0.7:
            report_text += "• Model shows strong generalization capabilities\n"
        elif np.mean(all_success_rates) > 0.4:
            report_text += "• Model shows moderate generalization capabilities\n"
        else:
            report_text += "• Model shows limited generalization capabilities\n"

        # Category-specific insights
        if 'grid_size' in category_performance:
            grid_perf = category_performance['grid_size']['avg_success_rate']
            if grid_perf > 0.6:
                report_text += "• Good generalization to different grid sizes\n"
            else:
                report_text += "• Limited generalization to different grid sizes\n"

        if 'colors' in category_performance:
            color_perf = category_performance['colors']['avg_success_rate']
            if color_perf > 0.6:
                report_text += "• Good generalization to novel colors\n"
            else:
                report_text += "• Limited generalization to novel colors\n"

        if 'complexity' in category_performance:
            complexity_perf = category_performance['complexity']['avg_success_rate']
            if complexity_perf > 0.6:
                report_text += "• Good generalization to increased complexity\n"
            else:
                report_text += "• Limited generalization to increased complexity\n"

        report_text += f"""
RECOMMENDATIONS
---------------
"""

        if np.mean(all_success_rates) < 0.3:
            report_text += "• Consider additional training or architectural improvements\n"

        if 'colors' in category_performance and category_performance['colors']['avg_success_rate'] < 0.4:
            report_text += "• Consider training with more diverse color combinations\n"

        if 'complexity' in category_performance and category_performance['complexity']['avg_success_rate'] < 0.4:
            report_text += "• Consider curriculum learning with gradually increasing complexity\n"

        if best_scenario and worst_scenario:
            success_gap = best_scenario[2]['success_rate'] - worst_scenario[2]['success_rate']
            if success_gap > 0.5:
                report_text += "• Large performance gaps suggest specific weaknesses to address\n"

        # Save report
        with open(analysis_dir / 'generalization_report.txt', 'w') as f:
            f.write(report_text)

        print("\n📋 GENERALIZATION TEST SUMMARY:")
        print(f"Overall Success Rate: {np.mean(all_success_rates):.2%}")
        print(f"Overall Average Reward: {np.mean(all_rewards):.2f}")
        print(f"Best Category: {max(category_performance.items(), key=lambda x: x[1]['avg_success_rate'])[0] if category_performance else 'N/A'}")

def main():
    """Main function for running generalization tests"""
    parser = argparse.ArgumentParser(description='Run generalization tests on trained emergent communication model')
    parser.add_argument('--model_path', type=str,
                       default='results/architect_builder_v1/models/final_model.pt',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes per test scenario')
    parser.add_argument('--output_dir', type=str, default='generalization_results',
                       help='Directory to save results')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        print("Available model files:")
        for root, dirs, files in os.walk('results'):
            for file in files:
                if file.endswith('.pt'):
                    print(f"  {os.path.join(root, file)}")
        return

    # Initialize framework
    framework = GeneralizationTestFramework(args.model_path, args.output_dir)

    # Run generalization tests
    results = framework.run_all_generalization_tests(args.episodes)

    # Analyze results
    framework.analyze_generalization_results(results)

    print(f"\n🎉 Generalization tests completed!")
    print(f"📁 Results and analysis saved to: {args.output_dir}")

if __name__ == "__main__":
    main()