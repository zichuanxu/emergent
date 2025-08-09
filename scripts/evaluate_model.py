#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for Emergent Communication
Includes interpretability tools like NMI, consistency tests, and behavioral analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import argparse
import os
import json
import sys
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from environments.environment import ArchitectBuilderEnv
from agents.agents import Architect, Builder
from training.mappo import MAPPO

class EvaluationFramework:
    def __init__(self, model_path, num_episodes=100):
        self.num_episodes = num_episodes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.load_models(model_path)
        
        # Initialize environment
        self.env = ArchitectBuilderEnv()
        
        # Storage for analysis
        self.episode_data = []
        self.communication_patterns = defaultdict(list)
        self.behavioral_metrics = defaultdict(list)
        
    def load_models(self, model_path):
        """Load trained models"""
        print(f"Loading models from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize models with same architecture as training
        # Use the original architecture parameters from training
        self.architect = Architect(
            grid_size=8, 
            input_channels=6, 
            hidden_dim=64,  # Original hidden_dim was 64
            vocab_size=8,   # Original vocab_size was 8
            message_length=3  # Message length was 3
        ).to(self.device)
        
        self.builder = Builder(
            grid_size=8,
            input_channels=6,
            hidden_dim=64,
            vocab_size=8,
            message_length=3,
            action_dim=5
        ).to(self.device)
        
        # Load state dicts with strict=False to handle extra parameters
        self.architect.load_state_dict(checkpoint['architect_state_dict'], strict=False)
        self.builder.load_state_dict(checkpoint['builder_state_dict'], strict=False)
        
        self.architect.eval()
        self.builder.eval()
        
        print("Models loaded successfully!")
    
    def collect_episode_data(self):
        """Collect data from multiple episodes for analysis"""
        print(f"Collecting data from {self.num_episodes} episodes...")
        
        for episode in range(self.num_episodes):
            episode_messages = []
            episode_actions = []
            episode_rewards = []
            episode_states = []
            
            obs = self.env.reset()
            done = False
            step = 0
            
            while not done and step < 50:  # Max 50 steps per episode
                with torch.no_grad():
                    # Get architect's observation and message
                    arch_obs = obs['architect'].unsqueeze(0).to(self.device)  # Don't flatten - keep as tensor
                    message, _, _ = self.architect(arch_obs)  # Returns message, value, logits
                    
                    # Get builder's action
                    builder_obs = obs['builder'].unsqueeze(0).to(self.device)
                    carrying_state = torch.tensor([obs.get('builder_carrying', 0)]).to(self.device)
                    action_logits, _, _ = self.builder(builder_obs, message, carrying_state)
                    action = torch.argmax(action_logits, dim=-1).item()
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                
                # Store data
                episode_messages.append(message.cpu().numpy().flatten())
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_states.append(obs['builder'].clone().detach())
                
                step += 1
            
            # Store episode data
            self.episode_data.append({
                'episode': episode,
                'messages': episode_messages,
                'actions': episode_actions,
                'rewards': episode_rewards,
                'states': episode_states,
                'total_reward': sum(episode_rewards),
                'success': info.get('blueprint_completed', False) if info else False,
                'steps': len(episode_rewards)
            })
            
            if (episode + 1) % 20 == 0:
                print(f"Collected {episode + 1}/{self.num_episodes} episodes")
    
    def compute_nmi_scores(self):
        """Compute Normalized Mutual Information between messages and various factors"""
        print("Computing NMI scores...")
        
        all_messages = []
        all_actions = []
        all_rewards = []
        all_success = []
        
        for episode in self.episode_data:
            all_messages.extend(episode['messages'])
            all_actions.extend(episode['actions'])
            all_rewards.extend(episode['rewards'])
            all_success.extend([episode['success']] * len(episode['actions']))
        
        # Convert messages to discrete labels for NMI computation
        # Use k-means clustering to discretize continuous messages
        messages_array = np.array(all_messages)
        n_clusters = min(10, len(set(map(tuple, all_messages))))  # Max 10 clusters
        
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            message_clusters = kmeans.fit_predict(messages_array)
        else:
            message_clusters = np.zeros(len(all_messages))
        
        # Compute NMI scores
        nmi_scores = {}
        
        if len(set(all_actions)) > 1:
            nmi_scores['message_action'] = normalized_mutual_info_score(
                message_clusters, all_actions
            )
        
        # Discretize rewards for NMI computation
        reward_bins = np.digitize(all_rewards, bins=np.percentile(all_rewards, [25, 50, 75]))
        if len(set(reward_bins)) > 1:
            nmi_scores['message_reward'] = normalized_mutual_info_score(
                message_clusters, reward_bins
            )
        
        if len(set(all_success)) > 1:
            nmi_scores['message_success'] = normalized_mutual_info_score(
                message_clusters, all_success
            )
        
        return nmi_scores
    
    def consistency_tests(self):
        """Perform consistency tests on communication"""
        print("Running consistency tests...")
        
        consistency_results = {}
        
        # Test 1: Message stability (same state -> similar message)
        state_message_pairs = []
        for episode in self.episode_data:
            for i, (state, message) in enumerate(zip(episode['states'], episode['messages'])):
                # Convert tensor state to numpy for comparison
                if hasattr(state, 'numpy'):
                    state_np = state.flatten().numpy()
                else:
                    state_np = np.array(state).flatten()
                state_message_pairs.append((tuple(state_np), message))
        
        # Group by similar states (discretized)
        state_groups = defaultdict(list)
        for state, message in state_message_pairs:
            # Discretize state for grouping
            discretized_state = tuple(np.round(np.array(state), 1))
            state_groups[discretized_state].append(message)
        
        # Compute consistency within each state group
        consistency_scores = []
        for state, messages in state_groups.items():
            if len(messages) > 1:
                messages_array = np.array(messages)
                # Compute pairwise distances
                distances = []
                for i in range(len(messages)):
                    for j in range(i+1, len(messages)):
                        dist = np.linalg.norm(messages_array[i] - messages_array[j])
                        distances.append(dist)
                
                if distances:
                    avg_distance = np.mean(distances)
                    consistency_scores.append(1.0 / (1.0 + avg_distance))  # Higher is more consistent
        
        consistency_results['message_stability'] = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # Test 2: Action consistency (similar messages -> similar actions)
        message_action_pairs = []
        for episode in self.episode_data:
            for message, action in zip(episode['messages'], episode['actions']):
                message_action_pairs.append((tuple(message), action))
        
        # Group by similar messages
        message_groups = defaultdict(list)
        for message, action in message_action_pairs:
            # Discretize message for grouping
            discretized_message = tuple(np.round(np.array(message), 1))
            message_groups[discretized_message].append(action)
        
        action_consistency_scores = []
        for message, actions in message_groups.items():
            if len(actions) > 1:
                # Compute action consistency (fraction of most common action)
                action_counts = Counter(actions)
                most_common_count = action_counts.most_common(1)[0][1]
                consistency = most_common_count / len(actions)
                action_consistency_scores.append(consistency)
        
        consistency_results['action_consistency'] = np.mean(action_consistency_scores) if action_consistency_scores else 0.0
        
        return consistency_results
    
    def behavioral_analysis(self):
        """Analyze behavioral patterns and performance metrics"""
        print("Performing behavioral analysis...")
        
        behavioral_metrics = {}
        
        # Success rate
        success_rate = np.mean([ep['success'] for ep in self.episode_data])
        behavioral_metrics['success_rate'] = success_rate
        
        # Average reward
        avg_reward = np.mean([ep['total_reward'] for ep in self.episode_data])
        behavioral_metrics['avg_total_reward'] = avg_reward
        
        # Episode length statistics
        episode_lengths = [ep['steps'] for ep in self.episode_data]
        behavioral_metrics['avg_episode_length'] = np.mean(episode_lengths)
        behavioral_metrics['std_episode_length'] = np.std(episode_lengths)
        
        # Message diversity
        all_messages = []
        for episode in self.episode_data:
            all_messages.extend(episode['messages'])
        
        message_array = np.array(all_messages)
        unique_messages = len(set(map(tuple, np.round(message_array, 2))))
        total_messages = len(all_messages)
        behavioral_metrics['message_diversity'] = unique_messages / total_messages if total_messages > 0 else 0
        
        # Action diversity
        all_actions = []
        for episode in self.episode_data:
            all_actions.extend(episode['actions'])
        
        unique_actions = len(set(all_actions))
        behavioral_metrics['action_diversity'] = unique_actions / 5  # 5 is total possible actions
        
        # Learning progression (if episodes are in order)
        episode_rewards = [ep['total_reward'] for ep in self.episode_data]
        first_half_avg = np.mean(episode_rewards[:len(episode_rewards)//2])
        second_half_avg = np.mean(episode_rewards[len(episode_rewards)//2:])
        behavioral_metrics['reward_improvement'] = second_half_avg - first_half_avg
        
        return behavioral_metrics
    
    def visualize_communication_patterns(self, save_dir='evaluation_results'):
        """Create visualizations of communication patterns"""
        print("Creating visualizations...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Message space visualization using t-SNE
        all_messages = []
        all_actions = []
        all_rewards = []
        
        for episode in self.episode_data:
            all_messages.extend(episode['messages'])
            all_actions.extend(episode['actions'])
            all_rewards.extend(episode['rewards'])
        
        if len(all_messages) > 50:  # Only create t-SNE if we have enough data
            messages_array = np.array(all_messages)
            
            # t-SNE visualization
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_messages)-1))
            messages_2d = tsne.fit_transform(messages_array)
            
            # Plot messages colored by actions
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(messages_2d[:, 0], messages_2d[:, 1], c=all_actions, cmap='tab10', alpha=0.6)
            plt.colorbar(scatter, label='Action')
            plt.title('Message Space (colored by Action)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            plt.subplot(1, 2, 2)
            scatter = plt.scatter(messages_2d[:, 0], messages_2d[:, 1], c=all_rewards, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Reward')
            plt.title('Message Space (colored by Reward)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'message_space_tsne.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Message-Action correlation heatmap
        message_dims = len(all_messages[0]) if all_messages else 3
        action_counts = np.zeros((5, message_dims))  # 5 actions, message_dims dimensions
        
        for message, action in zip(all_messages, all_actions):
            for dim, value in enumerate(message):
                action_counts[action, dim] += abs(value)  # Use absolute value
        
        # Normalize by action frequency
        action_freq = np.bincount(all_actions, minlength=5)
        for action in range(5):
            if action_freq[action] > 0:
                action_counts[action] /= action_freq[action]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(action_counts, annot=True, fmt='.2f', cmap='viridis',
                   xticklabels=[f'Msg Dim {i}' for i in range(message_dims)],
                   yticklabels=[f'Action {i}' for i in range(5)])
        plt.title('Message-Action Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'message_action_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance over episodes
        episode_rewards = [ep['total_reward'] for ep in self.episode_data]
        episode_success = [ep['success'] for ep in self.episode_data]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards, alpha=0.7)
        plt.plot(np.convolve(episode_rewards, np.ones(min(10, len(episode_rewards))))/min(10, len(episode_rewards)), 
                 'r-', linewidth=2, label='Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward over Episodes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        window_size = min(20, len(episode_success))
        success_rate_over_time = []
        for i in range(window_size, len(episode_success) + 1):
            success_rate_over_time.append(np.mean(episode_success[i-window_size:i]))
        
        if success_rate_over_time:
            plt.plot(range(window_size, len(episode_success) + 1), success_rate_over_time, 'g-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.title(f'Success Rate (Rolling {window_size}-episode window)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_over_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {save_dir}/")
    
    def generate_report(self, save_dir='evaluation_results'):
        """Generate comprehensive evaluation report"""
        print("Generating evaluation report...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Collect all metrics
        nmi_scores = self.compute_nmi_scores()
        consistency_results = self.consistency_tests()
        behavioral_metrics = self.behavioral_analysis()
        
        # Create comprehensive report
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'num_episodes_evaluated': self.num_episodes,
            'nmi_scores': nmi_scores,
            'consistency_tests': consistency_results,
            'behavioral_metrics': behavioral_metrics,
            'summary': {
                'overall_success_rate': behavioral_metrics.get('success_rate', 0),
                'communication_effectiveness': np.mean(list(nmi_scores.values())) if nmi_scores else 0,
                'behavioral_consistency': np.mean(list(consistency_results.values())) if consistency_results else 0,
                'learning_progression': behavioral_metrics.get('reward_improvement', 0)
            }
        }
        
        # Save report as JSON
        with open(os.path.join(save_dir, 'evaluation_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create human-readable summary
        summary_text = f"""
EMERGENT COMMUNICATION EVALUATION REPORT
========================================
Generated: {report['evaluation_timestamp']}
Episodes Evaluated: {self.num_episodes}

PERFORMANCE METRICS
-------------------
Success Rate: {behavioral_metrics.get('success_rate', 0):.2%}
Average Total Reward: {behavioral_metrics.get('avg_total_reward', 0):.2f}
Average Episode Length: {behavioral_metrics.get('avg_episode_length', 0):.1f} steps
Reward Improvement: {behavioral_metrics.get('reward_improvement', 0):.2f}

COMMUNICATION ANALYSIS
----------------------
Message Diversity: {behavioral_metrics.get('message_diversity', 0):.3f}
Action Diversity: {behavioral_metrics.get('action_diversity', 0):.3f}

NMI SCORES (Communication Effectiveness)
----------------------------------------
"""
        
        for metric, score in nmi_scores.items():
            summary_text += f"{metric.replace('_', ' ').title()}: {score:.3f}\n"
        
        summary_text += f"""
CONSISTENCY TESTS
-----------------
"""
        
        for metric, score in consistency_results.items():
            summary_text += f"{metric.replace('_', ' ').title()}: {score:.3f}\n"
        
        summary_text += f"""
INTERPRETABILITY INSIGHTS
--------------------------
- Communication-Action Alignment: {'High' if nmi_scores.get('message_action', 0) > 0.3 else 'Moderate' if nmi_scores.get('message_action', 0) > 0.1 else 'Low'}
- Message Stability: {'High' if consistency_results.get('message_stability', 0) > 0.7 else 'Moderate' if consistency_results.get('message_stability', 0) > 0.4 else 'Low'}
- Learning Evidence: {'Yes' if behavioral_metrics.get('reward_improvement', 0) > 5 else 'Limited'}

RECOMMENDATIONS
---------------
"""
        
        if behavioral_metrics.get('success_rate', 0) < 0.1:
            summary_text += "- Consider extending training or adjusting reward structure\n"
        
        if nmi_scores.get('message_action', 0) < 0.2:
            summary_text += "- Communication may need more structure or training\n"
        
        if behavioral_metrics.get('message_diversity', 0) < 0.1:
            summary_text += "- Message space may be too constrained or needs regularization\n"
        
        if consistency_results.get('message_stability', 0) < 0.5:
            summary_text += "- Consider stability improvements in communication protocol\n"
        
        # Save summary
        with open(os.path.join(save_dir, 'evaluation_summary.txt'), 'w') as f:
            f.write(summary_text)
        
        print(f"Evaluation report saved to {save_dir}/")
        print("\nSUMMARY:")
        print(f"Success Rate: {behavioral_metrics.get('success_rate', 0):.2%}")
        print(f"Avg Reward: {behavioral_metrics.get('avg_total_reward', 0):.2f}")
        print(f"Communication Effectiveness: {np.mean(list(nmi_scores.values())) if nmi_scores else 0:.3f}")
        
        return report
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("Starting comprehensive evaluation...")
        
        # Collect episode data
        self.collect_episode_data()
        
        # Analyze the data
        nmi_scores = self.compute_nmi_scores()
        consistency_results = self.consistency_tests()
        behavioral_metrics = self.behavioral_analysis()
        
        # Print results
        print("\n=== EVALUATION RESULTS ===")
        print(f"Success Rate: {behavioral_metrics.get('success_rate', 0):.2%}")
        print(f"Average Total Reward: {behavioral_metrics.get('avg_total_reward', 0):.2f}")
        print(f"Average Episode Length: {behavioral_metrics.get('avg_episode_length', 0):.1f} steps")
        print(f"Message Diversity: {behavioral_metrics.get('message_diversity', 0):.3f}")
        print(f"Action Diversity: {behavioral_metrics.get('action_diversity', 0):.3f}")
        
        print("\nNMI Scores (Communication Effectiveness):")
        for metric, score in nmi_scores.items():
            print(f"  {metric}: {score:.3f}")
        
        print("\nConsistency Tests:")
        for metric, score in consistency_results.items():
            print(f"  {metric}: {score:.3f}")
        
        # Create simple report
        report = {
            'nmi_scores': nmi_scores,
            'consistency_tests': consistency_results,
            'behavioral_metrics': behavioral_metrics
        }
        
        print("\nEvaluation completed successfully!")
        return report

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained emergent communication model')
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
        return
    
    # Run evaluation
    evaluator = EvaluationFramework(args.model_path, args.num_episodes)
    evaluator.run_full_evaluation()

if __name__ == "__main__":
    main()
