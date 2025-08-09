#!/usr/bin/env python3
"""
Ablation Studies Framework for Emergent Communication
Tests different communication architectures and training configurations
"""

import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import argparse
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from config.config import config
from src.environments.environment import ArchitectBuilderEnv
from src.agents.agents import Architect, Builder
from src.training.mappo import MAPPO
from src.utils.training_logger import TrainingLogger

class AblationStudyFramework:
    """
    Framework for conducting ablation studies on communication architectures
    """

    def __init__(self, base_config=None, output_dir='ablation_results'):
        self.base_config = base_config or config.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define ablation configurations
        self.ablation_configs = self._define_ablation_configs()

        # Results storage
        self.results = {}

    def _define_ablation_configs(self):
        """Define different configurations for ablation studies"""
        configs = {}

        # 1. Message Length Ablation
        configs['message_length'] = {
            'short_messages': {**self.base_config, 'message_length': 1, 'name': 'Short Messages (L=1)'},
            'medium_messages': {**self.base_config, 'message_length': 3, 'name': 'Medium Messages (L=3)'},
            'long_messages': {**self.base_config, 'message_length': 5, 'name': 'Long Messages (L=5)'},
        }

        # 2. Vocabulary Size Ablation
        configs['vocab_size'] = {
            'small_vocab': {**self.base_config, 'vocab_size': 4, 'name': 'Small Vocab (V=4)'},
            'medium_vocab': {**self.base_config, 'vocab_size': 8, 'name': 'Medium Vocab (V=8)'},
            'large_vocab': {**self.base_config, 'vocab_size': 16, 'name': 'Large Vocab (V=16)'},
        }

        # 3. Architecture Ablation
        configs['architecture'] = {
            'small_hidden': {**self.base_config, 'architect_hidden_dim': 32, 'builder_hidden_dim': 32, 'name': 'Small Hidden (32)'},
            'medium_hidden': {**self.base_config, 'architect_hidden_dim': 64, 'builder_hidden_dim': 64, 'name': 'Medium Hidden (64)'},
            'large_hidden': {**self.base_config, 'architect_hidden_dim': 128, 'builder_hidden_dim': 128, 'name': 'Large Hidden (128)'},
        }

        # 4. Communication Mechanism Ablation
        configs['communication'] = {
            'no_communication': {**self.base_config, 'vocab_size': 1, 'message_length': 1, 'name': 'No Communication'},
            'discrete_communication': {**self.base_config, 'gumbel_hard': True, 'name': 'Discrete Communication'},
            'continuous_communication': {**self.base_config, 'gumbel_hard': False, 'name': 'Continuous Communication'},
        }

        # 5. Training Hyperparameter Ablation
        configs['training'] = {
            'low_lr': {**self.base_config, 'learning_rate': 1e-5, 'name': 'Low LR (1e-5)'},
            'medium_lr': {**self.base_config, 'learning_rate': 1e-4, 'name': 'Medium LR (1e-4)'},
            'high_lr': {**self.base_config, 'learning_rate': 1e-3, 'name': 'High LR (1e-3)'},
        }

        # 6. Environment Complexity Ablation
        configs['environment'] = {
            'simple_env': {**self.base_config, 'grid_size': 3, 'block_colors': ['red'], 'name': 'Simple Env (3x3, 1 color)'},
            'medium_env': {**self.base_config, 'grid_size': 4, 'block_colors': ['red', 'green'], 'name': 'Medium Env (4x4, 2 colors)'},
            'complex_env': {**self.base_config, 'grid_size': 5, 'block_colors': ['red', 'green', 'blue'], 'name': 'Complex Env (5x5, 3 colors)'},
        }

        return configs

    def run_single_ablation(self, config_dict, experiment_name, num_episodes=500):
        """Run a single ablation experiment"""
        print(f"\n🔬 Running ablation: {experiment_name}")
        print(f"📋 Config: {config_dict.get('name', experiment_name)}")

        # Create experiment directory
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize environment with config
        env = ArchitectBuilderEnv(
            grid_size=config_dict['grid_size'],
            block_colors=config_dict['block_colors'],
            max_blocks_per_color=config_dict.get('max_blocks_per_color', 1),
            max_episode_steps=config_dict.get('max_episode_steps', 50)
        )

        # Initialize agents with config
        architect = Architect(
            grid_size=config_dict['grid_size'],
            input_channels=6,
            hidden_dim=config_dict['architect_hidden_dim'],
            vocab_size=config_dict['vocab_size'],
            message_length=config_dict['message_length']
        ).to(config_dict['device'])

        builder = Builder(
            grid_size=config_dict['grid_size'],
            input_channels=6,
            hidden_dim=config_dict['builder_hidden_dim'],
            vocab_size=config_dict['vocab_size'],
            message_length=config_dict['message_length'],
            action_dim=config_dict['action_dim']
        ).to(config_dict['device'])

        # Initialize trainer
        trainer = MAPPO(architect, builder, config_dict)

        # Initialize logging
        log_file = exp_dir / 'training_log.json'
        logger = TrainingLogger(str(log_file))

        # Training metrics
        episode_rewards = []
        episode_lengths = []
        success_count = 0

        # Training loop
        progress_bar = tqdm(range(num_episodes), desc=f"Training {experiment_name}")

        for episode in progress_bar:
            # Reset environment
            obs = env.reset()
            architect_obs = obs['architect'].to(config_dict['device'])
            builder_obs = obs['builder'].to(config_dict['device'])
            carrying_state = torch.tensor(obs['builder_carrying']).to(config_dict['device'])

            hidden_state = None
            episode_reward = 0
            episode_length = 0
            done = False

            episode_messages = []
            episode_actions = []

            while not done:
                # Select actions
                actions = trainer.select_actions(
                    architect_obs, builder_obs, carrying_state,
                    hidden_state, training=True
                )

                # Take step in environment
                next_obs, reward, done, info = env.step(actions['action'].cpu().item())

                # Log step data
                logger.log_step(actions['message'], actions['action'])
                episode_messages.append(actions['message'].cpu().detach().numpy())
                episode_actions.append(actions['action'].cpu().item())

                # Add experience to buffer
                trainer.add_experience(
                    architect_obs=architect_obs,
                    builder_obs=builder_obs,
                    carrying_state=carrying_state,
                    message=actions['message'],
                    action=actions['action'],
                    reward=torch.tensor(reward),
                    done=torch.tensor(done),
                    architect_value=actions['architect_value'],
                    builder_value=actions['builder_value'],
                    action_log_prob=actions['action_log_prob'],
                    message_log_prob=actions['message_log_prob']
                )

                # Update observations
                architect_obs = next_obs['architect'].to(config_dict['device'])
                builder_obs = next_obs['builder'].to(config_dict['device'])
                carrying_state = torch.tensor(next_obs['builder_carrying']).to(config_dict['device'])
                hidden_state = actions['hidden_state']

                episode_reward += reward
                episode_length += 1

            # Track episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            episode_success = env._is_blueprint_complete() or reward > 500
            if episode_success:
                success_count += 1

            # Log episode data
            current_success_rate = success_count / (episode + 1)

            # Convert messages and actions for logging
            log_messages = []
            for msg in episode_messages:
                if hasattr(msg, 'tolist'):
                    log_messages.append(msg.tolist())
                elif isinstance(msg, np.ndarray):
                    log_messages.append(msg.tolist())
                else:
                    log_messages.append(msg)

            log_actions = []
            for action in episode_actions:
                if hasattr(action, 'item'):
                    log_actions.append(action.item())
                elif isinstance(action, np.ndarray):
                    log_actions.append(action.item())
                else:
                    log_actions.append(int(action))

            logger.log_episode(
                episode=episode,
                reward=episode_reward,
                success_rate=current_success_rate,
                episode_length=episode_length,
                messages=log_messages,
                actions=log_actions
            )

            # Update networks periodically
            if len(trainer.buffer.experiences) >= config_dict['batch_size']:
                loss = trainer.update()
                if loss is not None:
                    logger.log_loss(episode, loss)

            # Update progress bar
            if episode % 50 == 0:
                recent_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
                recent_success = success_count / min(episode + 1, 50) * 100

                progress_bar.set_postfix({
                    'Reward': f"{np.mean(recent_rewards):.2f}",
                    'Success': f"{recent_success:.1f}%",
                    'Length': f"{np.mean(episode_lengths[-50:]):.1f}"
                })

        # Save final model
        model_path = exp_dir / 'final_model.pt'
        torch.save({
            'architect_state_dict': architect.state_dict(),
            'builder_state_dict': builder.state_dict(),
            'config': config_dict,
            'training_stats': {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'final_success_rate': success_count / num_episodes * 100
            }
        }, model_path)

        # Cleanup logging
        logger.close()

        # Return results
        results = {
            'experiment_name': experiment_name,
            'config': config_dict,
            'final_success_rate': success_count / num_episodes * 100,
            'avg_reward': np.mean(episode_rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'reward_std': np.std(episode_rewards),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'model_path': str(model_path)
        }

        print(f"✅ Completed {experiment_name}")
        print(f"📊 Success Rate: {results['final_success_rate']:.1f}%")
        print(f"🎯 Avg Reward: {results['avg_reward']:.2f}")

        return results

    def run_ablation_category(self, category, num_episodes=500):
        """Run all experiments in a specific ablation category"""
        print(f"\n🔬 Running {category} ablation studies...")

        if category not in self.ablation_configs:
            print(f"❌ Unknown ablation category: {category}")
            return {}

        category_results = {}
        configs = self.ablation_configs[category]

        for exp_name, exp_config in configs.items():
            try:
                result = self.run_single_ablation(exp_config, f"{category}_{exp_name}", num_episodes)
                category_results[exp_name] = result
            except Exception as e:
                print(f"❌ Failed to run {exp_name}: {e}")
                category_results[exp_name] = {'error': str(e)}

        # Save category results
        results_file = self.output_dir / f"{category}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(category_results, f, indent=2, default=str, ensure_ascii=False)

        return category_results

    def run_all_ablations(self, num_episodes=500):
        """Run all ablation studies"""
        print("🚀 Starting comprehensive ablation studies...")

        all_results = {}

        for category in self.ablation_configs.keys():
            print(f"\n{'='*50}")
            print(f"📋 Category: {category.upper()}")
            print(f"{'='*50}")

            category_results = self.run_ablation_category(category, num_episodes)
            all_results[category] = category_results

        # Save all results
        self.results = all_results
        results_file = self.output_dir / 'all_ablation_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)

        print(f"\n✅ All ablation studies completed!")
        print(f"📁 Results saved to: {self.output_dir}")

        return all_results

    def analyze_results(self, results=None):
        """Analyze and visualize ablation study results"""
        if results is None:
            results = self.results

        if not results:
            print("❌ No results to analyze. Run ablation studies first.")
            return

        print("📊 Analyzing ablation study results...")

        # Create analysis directory
        analysis_dir = self.output_dir / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Analyze each category
        for category, category_results in results.items():
            self._analyze_category(category, category_results, analysis_dir)

        # Create summary comparison
        self._create_summary_comparison(results, analysis_dir)

        print(f"📈 Analysis completed! Visualizations saved to: {analysis_dir}")

    def _analyze_category(self, category, category_results, analysis_dir):
        """Analyze results for a specific category"""
        # Extract metrics
        exp_names = []
        success_rates = []
        avg_rewards = []
        avg_lengths = []

        for exp_name, result in category_results.items():
            if 'error' not in result:
                exp_names.append(result.get('config', {}).get('name', exp_name))
                success_rates.append(result['final_success_rate'])
                avg_rewards.append(result['avg_reward'])
                avg_lengths.append(result['avg_episode_length'])

        if not exp_names:
            print(f"⚠️ No valid results for category {category}")
            return

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Ablation Study: {category.title()}', fontsize=16, fontweight='bold')

        # Success rates
        axes[0, 0].bar(range(len(exp_names)), success_rates, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Success Rate (%)')
        axes[0, 0].set_xticks(range(len(exp_names)))
        axes[0, 0].set_xticklabels(exp_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)

        # Average rewards
        axes[0, 1].bar(range(len(exp_names)), avg_rewards, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Average Reward')
        axes[0, 1].set_xticks(range(len(exp_names)))
        axes[0, 1].set_xticklabels(exp_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)

        # Episode lengths
        axes[1, 0].bar(range(len(exp_names)), avg_lengths, color='orange', alpha=0.7)
        axes[1, 0].set_title('Average Episode Length')
        axes[1, 0].set_xticks(range(len(exp_names)))
        axes[1, 0].set_xticklabels(exp_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True, alpha=0.3)

        # Learning curves (if available)
        axes[1, 1].set_title('Learning Curves')
        for i, (exp_name, result) in enumerate(category_results.items()):
            if 'error' not in result and 'episode_rewards' in result:
                rewards = result['episode_rewards']
                # Smooth the curve
                window_size = min(50, len(rewards) // 10)
                if window_size > 1:
                    smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    axes[1, 1].plot(smoothed, label=result.get('config', {}).get('name', exp_name), alpha=0.8)

        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(analysis_dir / f'{category}_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_summary_comparison(self, results, analysis_dir):
        """Create summary comparison across all categories"""
        # Find best performing configuration in each category
        best_configs = {}

        for category, category_results in results.items():
            best_success = -1
            best_config = None

            for exp_name, result in category_results.items():
                if 'error' not in result:
                    success_rate = result['final_success_rate']
                    if success_rate > best_success:
                        best_success = success_rate
                        best_config = {
                            'name': result.get('config', {}).get('name', exp_name),
                            'success_rate': success_rate,
                            'avg_reward': result['avg_reward'],
                            'category': category
                        }

            if best_config:
                best_configs[category] = best_config

        # Create summary visualization
        if best_configs:
            categories = list(best_configs.keys())
            success_rates = [best_configs[cat]['success_rate'] for cat in categories]
            avg_rewards = [best_configs[cat]['avg_reward'] for cat in categories]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Best Configuration per Ablation Category', fontsize=16, fontweight='bold')

            # Success rates
            bars1 = ax1.bar(range(len(categories)), success_rates, color='skyblue', alpha=0.7)
            ax1.set_title('Best Success Rate per Category')
            ax1.set_xticks(range(len(categories)))
            ax1.set_xticklabels([cat.title() for cat in categories], rotation=45, ha='right')
            ax1.set_ylabel('Success Rate (%)')
            ax1.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars1, success_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom')

            # Average rewards
            bars2 = ax2.bar(range(len(categories)), avg_rewards, color='lightgreen', alpha=0.7)
            ax2.set_title('Best Average Reward per Category')
            ax2.set_xticks(range(len(categories)))
            ax2.set_xticklabels([cat.title() for cat in categories], rotation=45, ha='right')
            ax2.set_ylabel('Average Reward')
            ax2.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars2, avg_rewards):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}', ha='center', va='bottom')

            try:
                plt.tight_layout()
            except UserWarning:
                # If tight_layout fails, adjust manually
                plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0.4)
            plt.savefig(analysis_dir / 'summary_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Create summary report
        summary_text = f"""
ABLATION STUDY SUMMARY REPORT
============================
Generated: {datetime.now().isoformat()}
Total Categories: {len(results)}

BEST CONFIGURATIONS PER CATEGORY
--------------------------------
"""

        for category, config in best_configs.items():
            summary_text += f"""
{category.upper()}:
  Best Configuration: {config['name']}
  Success Rate: {config['success_rate']:.1f}%
  Average Reward: {config['avg_reward']:.2f}
"""

        summary_text += f"""
KEY INSIGHTS
------------
"""

        # Add insights based on results
        if best_configs:
            best_overall = max(best_configs.values(), key=lambda x: x['success_rate'])
            summary_text += f"* Best overall performance: {best_overall['name']} ({best_overall['success_rate']:.1f}% success)\n"

            # Find categories with significant differences
            success_rates = [config['success_rate'] for config in best_configs.values()]
            if max(success_rates) - min(success_rates) > 20:
                summary_text += f"* Large performance differences observed (range: {min(success_rates):.1f}% - {max(success_rates):.1f}%)\n"

            # Communication-specific insights
            if 'communication' in best_configs:
                comm_config = best_configs['communication']
                summary_text += f"* Communication mechanism impact: {comm_config['name']} achieved {comm_config['success_rate']:.1f}% success\n"

        # Save summary
        with open(analysis_dir / 'ablation_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary_text)

        print("\n📋 ABLATION STUDY SUMMARY:")
        if best_configs:
            for category, config in best_configs.items():
                print(f"  {category.title()}: {config['name']} ({config['success_rate']:.1f}% success)")

def main():
    """Main function for running ablation studies"""
    parser = argparse.ArgumentParser(description='Run ablation studies on emergent communication')
    parser.add_argument('--category', type=str, choices=['message_length', 'vocab_size', 'architecture',
                                                        'communication', 'training', 'environment', 'all'],
                       default='all', help='Ablation category to run')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes per experiment')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='Directory to save results')

    args = parser.parse_args()

    # Initialize framework
    framework = AblationStudyFramework(output_dir=args.output_dir)

    # Run ablation studies
    if args.category == 'all':
        results = framework.run_all_ablations(args.episodes)
    else:
        results = {args.category: framework.run_ablation_category(args.category, args.episodes)}

    # Analyze results
    framework.analyze_results(results)

    print(f"\n🎉 Ablation studies completed!")
    print(f"📁 Results and analysis saved to: {args.output_dir}")

if __name__ == "__main__":
    main()