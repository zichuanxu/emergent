#!/usr/bin/env python3
"""
Enhanced Real-time Training Dashboard with Communication Analysis
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import seaborn as sns
import numpy as np
import sys
import os
import json
from collections import deque, defaultdict
import time
from datetime import datetime
import argparse
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class EnhancedRealTimeVisualizer:
    def __init__(self, log_file='training_log.json', update_interval=2000):
        self.log_file = log_file
        self.update_interval = update_interval

        # Create figure with subplots
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('Real-time Training Dashboard - Emergent Communication', fontsize=16, fontweight='bold')

        # Create grid layout
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Training progress plots
        self.ax_reward = self.fig.add_subplot(gs[0, 0])
        self.ax_success = self.fig.add_subplot(gs[0, 1])
        self.ax_length = self.fig.add_subplot(gs[0, 2])
        self.ax_loss = self.fig.add_subplot(gs[0, 3])

        # Communication analysis plots
        self.ax_messages = self.fig.add_subplot(gs[1, 0])
        self.ax_nmi = self.fig.add_subplot(gs[1, 1])
        self.ax_diversity = self.fig.add_subplot(gs[1, 2])
        self.ax_consistency = self.fig.add_subplot(gs[1, 3])

        # Advanced analysis plots
        self.ax_heatmap = self.fig.add_subplot(gs[2, 0:2])
        self.ax_distribution = self.fig.add_subplot(gs[2, 2])
        self.ax_efficiency = self.fig.add_subplot(gs[2, 3])

        # Data storage for analysis
        self.data_history = {
            'episodes': [],
            'rewards': [],
            'success_rates': [],
            'episode_lengths': [],
            'losses': [],
            'messages': [],
            'actions': [],
            'nmi_scores': [],
            'message_diversity': [],
            'consistency_scores': [],
            'communication_efficiency': []
        }

        # Rolling windows for smooth plotting
        self.window_size = 50
        self.last_update_time = time.time()

        print(f"📊 Enhanced Dashboard initialized")
        print(f"📁 Monitoring: {self.log_file}")
        print(f"🔄 Update interval: {self.update_interval}ms")

    def load_data(self):
        """Load and parse training data from log file"""
        try:
            if not os.path.exists(self.log_file):
                return False

            with open(self.log_file, 'r') as f:
                data = json.load(f)

            # Update data history
            for key in self.data_history.keys():
                if key in data:
                    self.data_history[key] = data[key]

            return True

        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            return False

    def compute_communication_metrics(self):
        """Compute real-time communication analysis metrics"""
        if not self.data_history['messages'] or not self.data_history['actions']:
            return

        try:
            # Get recent data
            recent_messages = self.data_history['messages'][-500:] if len(self.data_history['messages']) > 500 else self.data_history['messages']
            recent_actions = self.data_history['actions'][-500:] if len(self.data_history['actions']) > 500 else self.data_history['actions']

            if len(recent_messages) < 10 or len(recent_actions) < 10:
                return

            # Convert to numpy arrays with proper handling
            try:
                messages_array = np.array(recent_messages)
                actions_array = np.array(recent_actions)

                # Ensure messages are 2D
                if messages_array.ndim == 1:
                    messages_array = messages_array.reshape(-1, 1)
                elif messages_array.ndim > 2:
                    # Flatten extra dimensions
                    messages_array = messages_array.reshape(messages_array.shape[0], -1)
            except ValueError as e:
                # Handle irregular message shapes
                print(f"Warning: Irregular message shapes, skipping metrics computation: {e}")
                return

            # Compute NMI between messages and actions
            if messages_array.shape[1] > 1:
                # Use clustering for multi-dimensional messages
                n_clusters = min(8, len(set(map(tuple, messages_array.round(2)))))
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    message_clusters = kmeans.fit_predict(messages_array)
                    nmi_score = normalized_mutual_info_score(message_clusters, actions_array)
                else:
                    nmi_score = 0.0
            else:
                # For 1D messages, discretize directly
                message_bins = np.digitize(messages_array.flatten(), bins=np.percentile(messages_array.flatten(), [25, 50, 75]))
                nmi_score = normalized_mutual_info_score(message_bins, actions_array)

            # Message diversity
            unique_messages = len(set(map(tuple, messages_array.round(2))))
            diversity = unique_messages / len(messages_array)

            # Communication efficiency (entropy-based)
            message_entropy = 0
            for dim in range(messages_array.shape[1]):
                hist, _ = np.histogram(messages_array[:, dim], bins=10)
                hist = hist / np.sum(hist)
                dim_entropy = -np.sum(hist * np.log(hist + 1e-8))
                message_entropy += dim_entropy

            efficiency = message_entropy / messages_array.shape[1]  # Average entropy per dimension

            # Update metrics
            self.data_history['nmi_scores'].append(nmi_score)
            self.data_history['message_diversity'].append(diversity)
            self.data_history['communication_efficiency'].append(efficiency)

            # Consistency score (simplified)
            if len(recent_messages) > 20:
                # Split into two halves and compare
                mid = len(recent_messages) // 2
                first_half = np.array(recent_messages[:mid])
                second_half = np.array(recent_messages[mid:])

                first_mean = np.mean(first_half, axis=0)
                second_mean = np.mean(second_half, axis=0)

                consistency = 1.0 / (1.0 + np.linalg.norm(first_mean - second_mean))
                self.data_history['consistency_scores'].append(consistency)

        except Exception as e:
            print(f"Warning: Communication metrics computation failed: {e}")

    def update_plots(self, frame):
        """Update all plots with latest data"""
        # Load new data
        if not self.load_data():
            # Show "waiting for data" message
            for ax in [self.ax_reward, self.ax_success, self.ax_length, self.ax_loss]:
                ax.clear()
                ax.text(0.5, 0.5, 'Waiting for training data...',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title('No Data')
            return

        # Compute communication metrics
        self.compute_communication_metrics()

        # Update training progress plots
        self.update_training_plots()

        # Update communication analysis plots
        self.update_communication_plots()

        # Update advanced analysis plots
        self.update_advanced_plots()

        # Update timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        self.fig.suptitle(f'Real-time Training Dashboard - Emergent Communication (Updated: {current_time})',
                         fontsize=16, fontweight='bold')

    def update_training_plots(self):
        """Update basic training progress plots"""
        episodes = self.data_history['episodes']

        if not episodes:
            return

        # Reward plot
        self.ax_reward.clear()
        rewards = self.data_history['rewards']
        if rewards:
            self.ax_reward.plot(episodes, rewards, 'b-', alpha=0.6, linewidth=1)
            if len(rewards) > self.window_size:
                # Add moving average
                window_rewards = np.convolve(rewards, np.ones(self.window_size)/self.window_size, mode='valid')
                window_episodes = episodes[self.window_size-1:]
                self.ax_reward.plot(window_episodes, window_rewards, 'r-', linewidth=2, label=f'MA({self.window_size})')
                self.ax_reward.legend()

        self.ax_reward.set_title('Episode Rewards')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True, alpha=0.3)

        # Success rate plot
        self.ax_success.clear()
        success_rates = self.data_history['success_rates']
        if success_rates:
            # Convert to percentage and smooth
            success_pct = [s * 100 for s in success_rates]
            self.ax_success.plot(episodes, success_pct, 'g-', linewidth=2)
            if len(success_pct) > 10:
                # Add trend line
                z = np.polyfit(episodes[-50:], success_pct[-50:], 1)
                p = np.poly1d(z)
                self.ax_success.plot(episodes[-50:], p(episodes[-50:]), 'r--', alpha=0.8, label='Trend')
                self.ax_success.legend()

        self.ax_success.set_title('Success Rate')
        self.ax_success.set_xlabel('Episode')
        self.ax_success.set_ylabel('Success Rate (%)')
        self.ax_success.set_ylim(0, 100)
        self.ax_success.grid(True, alpha=0.3)

        # Episode length plot
        self.ax_length.clear()
        lengths = self.data_history['episode_lengths']
        if lengths:
            self.ax_length.plot(episodes, lengths, 'orange', alpha=0.7)
            if len(lengths) > self.window_size:
                window_lengths = np.convolve(lengths, np.ones(self.window_size)/self.window_size, mode='valid')
                window_episodes = episodes[self.window_size-1:]
                self.ax_length.plot(window_episodes, window_lengths, 'darkred', linewidth=2)

        self.ax_length.set_title('Episode Length')
        self.ax_length.set_xlabel('Episode')
        self.ax_length.set_ylabel('Steps')
        self.ax_length.grid(True, alpha=0.3)

        # Loss plot (if available)
        self.ax_loss.clear()
        losses = self.data_history['losses']
        if losses:
            self.ax_loss.semilogy(episodes, losses, 'purple', alpha=0.7)
            self.ax_loss.set_title('Training Loss')
            self.ax_loss.set_xlabel('Episode')
            self.ax_loss.set_ylabel('Loss (log scale)')
            self.ax_loss.grid(True, alpha=0.3)
        else:
            self.ax_loss.text(0.5, 0.5, 'Loss data\nnot available',
                             ha='center', va='center', transform=self.ax_loss.transAxes)
            self.ax_loss.set_title('Training Loss')

    def update_communication_plots(self):
        """Update communication analysis plots"""
        episodes = self.data_history['episodes']

        # Message patterns plot
        self.ax_messages.clear()
        messages = self.data_history['messages']
        if messages and len(messages) > 0:
            try:
                # Handle messages that might have different shapes
                if isinstance(messages[0], list):
                    # Multi-dimensional messages
                    messages_array = np.array(messages)
                    if messages_array.ndim > 1 and messages_array.shape[1] > 1:
                        # Plot message dimensions over time
                        for dim in range(min(3, messages_array.shape[1])):  # Show up to 3 dimensions
                            self.ax_messages.plot(messages_array[-200:, dim],
                                                label=f'Dim {dim}', alpha=0.7)
                        self.ax_messages.legend()
                        self.ax_messages.set_title('Message Patterns (Recent 200)')
                    else:
                        # Flatten if needed
                        flat_messages = [msg[0] if isinstance(msg, list) and len(msg) > 0 else msg for msg in messages[-200:]]
                        self.ax_messages.plot(flat_messages, 'b-', alpha=0.7)
                        self.ax_messages.set_title('Message Values (Recent 200)')
                else:
                    # Single dimension messages
                    self.ax_messages.plot(messages[-200:], 'b-', alpha=0.7)
                    self.ax_messages.set_title('Message Values (Recent 200)')
            except Exception as e:
                self.ax_messages.text(0.5, 0.5, f'Message data\nformat error:\n{str(e)[:50]}...',
                                    ha='center', va='center', transform=self.ax_messages.transAxes, fontsize=8)
                self.ax_messages.set_title('Message Patterns (Error)')
        else:
            self.ax_messages.text(0.5, 0.5, 'No message\ndata yet',
                                ha='center', va='center', transform=self.ax_messages.transAxes)
            self.ax_messages.set_title('Message Patterns')

        self.ax_messages.set_xlabel('Recent Steps')
        self.ax_messages.set_ylabel('Message Value')
        self.ax_messages.grid(True, alpha=0.3)

        # NMI scores plot
        self.ax_nmi.clear()
        nmi_scores = self.data_history['nmi_scores']
        if nmi_scores:
            self.ax_nmi.plot(nmi_scores, 'r-', linewidth=2)
            self.ax_nmi.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Good threshold')
            self.ax_nmi.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Poor threshold')
            self.ax_nmi.legend()
            self.ax_nmi.set_ylim(0, 1)

        self.ax_nmi.set_title('Message-Action NMI')
        self.ax_nmi.set_xlabel('Update')
        self.ax_nmi.set_ylabel('NMI Score')
        self.ax_nmi.grid(True, alpha=0.3)

        # Message diversity plot
        self.ax_diversity.clear()
        diversity = self.data_history['message_diversity']
        if diversity:
            self.ax_diversity.plot(diversity, 'g-', linewidth=2)
            self.ax_diversity.set_ylim(0, 1)

        self.ax_diversity.set_title('Message Diversity')
        self.ax_diversity.set_xlabel('Update')
        self.ax_diversity.set_ylabel('Diversity Score')
        self.ax_diversity.grid(True, alpha=0.3)

        # Consistency plot
        self.ax_consistency.clear()
        consistency = self.data_history['consistency_scores']
        if consistency:
            self.ax_consistency.plot(consistency, 'purple', linewidth=2)
            self.ax_consistency.set_ylim(0, 1)

        self.ax_consistency.set_title('Communication Consistency')
        self.ax_consistency.set_xlabel('Update')
        self.ax_consistency.set_ylabel('Consistency Score')
        self.ax_consistency.grid(True, alpha=0.3)

    def update_advanced_plots(self):
        """Update advanced analysis plots"""
        # Message-Action correlation heatmap
        self.ax_heatmap.clear()
        messages = self.data_history['messages']
        actions = self.data_history['actions']

        if messages and actions and len(messages) > 50:
            try:
                messages_array = np.array(messages[-200:])  # Recent 200 samples
                actions_array = np.array(actions[-200:])

                if messages_array.ndim > 1:
                    # Compute correlation matrix
                    n_actions = len(set(actions_array))
                    n_dims = messages_array.shape[1]
                    correlation_matrix = np.zeros((n_actions, n_dims))

                    for action in range(n_actions):
                        action_mask = actions_array == action
                        if np.sum(action_mask) > 0:
                            action_messages = messages_array[action_mask]
                            correlation_matrix[action] = np.mean(np.abs(action_messages), axis=0)

                    sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                               cmap='viridis', ax=self.ax_heatmap,
                               xticklabels=[f'Msg{i}' for i in range(n_dims)],
                               yticklabels=[f'Act{i}' for i in range(n_actions)])
                    self.ax_heatmap.set_title('Message-Action Correlation Matrix')
                else:
                    self.ax_heatmap.text(0.5, 0.5, 'Single dimension\nmessages',
                                       ha='center', va='center', transform=self.ax_heatmap.transAxes)
                    self.ax_heatmap.set_title('Message-Action Correlation')
            except Exception as e:
                self.ax_heatmap.text(0.5, 0.5, f'Correlation\ncomputation\nfailed',
                                   ha='center', va='center', transform=self.ax_heatmap.transAxes)
                self.ax_heatmap.set_title('Message-Action Correlation')
        else:
            self.ax_heatmap.text(0.5, 0.5, 'Insufficient data\nfor correlation\nanalysis',
                               ha='center', va='center', transform=self.ax_heatmap.transAxes)
            self.ax_heatmap.set_title('Message-Action Correlation')

        # Action distribution
        self.ax_distribution.clear()
        if actions:
            action_counts = np.bincount(actions[-500:])  # Recent 500 actions
            self.ax_distribution.bar(range(len(action_counts)), action_counts,
                                   color='skyblue', alpha=0.7)
            self.ax_distribution.set_title('Action Distribution\n(Recent 500)')
            self.ax_distribution.set_xlabel('Action')
            self.ax_distribution.set_ylabel('Count')

        # Communication efficiency
        self.ax_efficiency.clear()
        efficiency = self.data_history['communication_efficiency']
        if efficiency:
            self.ax_efficiency.plot(efficiency, 'brown', linewidth=2)
            self.ax_efficiency.set_title('Communication Efficiency')
            self.ax_efficiency.set_xlabel('Update')
            self.ax_efficiency.set_ylabel('Entropy Score')
            self.ax_efficiency.grid(True, alpha=0.3)

    def run(self):
        """Start the real-time dashboard"""
        print("🚀 Starting Enhanced Real-time Dashboard...")
        print("📊 Dashboard will update automatically as training progresses")
        print("❌ Close the window to stop monitoring")

        # Create animation
        ani = FuncAnimation(self.fig, self.update_plots, interval=self.update_interval, cache_frame_data=False)

        # Show the dashboard
        plt.show()

        return ani

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Real-time Training Dashboard')
    parser.add_argument('--log_file', type=str, default='training_log.json',
                       help='Path to training log file')
    parser.add_argument('--update_interval', type=int, default=2000,
                       help='Update interval in milliseconds')

    args = parser.parse_args()

    # Create and run dashboard
    dashboard = EnhancedRealTimeVisualizer(
        log_file=args.log_file,
        update_interval=args.update_interval
    )

    try:
        ani = dashboard.run()
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")

if __name__ == '__main__':
    main()
