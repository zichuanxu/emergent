#!/usr/bin/env python3
"""
Enhanced Real-time Dashboard for Emergent Communication Research
Includes ablation studies monitoring and generalization test visualization
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import numpy as np
import sys
import os
import json
from collections import deque, defaultdict
import time
from datetime import datetime
import argparse
from pathlib import Path
import threading
import queue
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class EnhancedResearchDashboard:
    """
    Enhanced dashboard for monitoring training, ablation studies, and generalization tests
    """

    def __init__(self, mode='training', data_sources=None, update_interval=3000):
        self.mode = mode  # 'training', 'ablation', 'generalization', or 'comparison'
        self.data_sources = data_sources or {}
        self.update_interval = update_interval

        # Initialize based on mode
        if mode == 'training':
            self._init_training_dashboard()
        elif mode == 'ablation':
            self._init_ablation_dashboard()
        elif mode == 'generalization':
            self._init_generalization_dashboard()
        elif mode == 'comparison':
            self._init_comparison_dashboard()

        # Data storage
        self.data_cache = {}
        self.last_update_time = time.time()

        print(f"🚀 Enhanced Research Dashboard initialized in {mode} mode")

    def _init_training_dashboard(self):
        """Initialize training monitoring dashboard"""
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.suptitle('Enhanced Training Dashboard - Emergent Communication',
                         fontsize=16, fontweight='bold')

        # Create comprehensive grid layout
        gs = self.fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

        # Training progress (row 1)
        self.ax_reward = self.fig.add_subplot(gs[0, 0])
        self.ax_success = self.fig.add_subplot(gs[0, 1])
        self.ax_length = self.fig.add_subplot(gs[0, 2])
        self.ax_loss = self.fig.add_subplot(gs[0, 3])

        # Communication analysis (row 2)
        self.ax_messages = self.fig.add_subplot(gs[1, 0])
        self.ax_nmi = self.fig.add_subplot(gs[1, 1])
        self.ax_diversity = self.fig.add_subplot(gs[1, 2])
        self.ax_consistency = self.fig.add_subplot(gs[1, 3])

        # Advanced analysis (row 3)
        self.ax_heatmap = self.fig.add_subplot(gs[2, 0:2])
        self.ax_distribution = self.fig.add_subplot(gs[2, 2])
        self.ax_efficiency = self.fig.add_subplot(gs[2, 3])

        # Research insights (row 4)
        self.ax_emergence = self.fig.add_subplot(gs[3, 0])
        self.ax_generalization = self.fig.add_subplot(gs[3, 1])
        self.ax_stability = self.fig.add_subplot(gs[3, 2])
        self.ax_insights = self.fig.add_subplot(gs[3, 3])

        # Data storage for training
        self.training_data = {
            'episodes': [], 'rewards': [], 'success_rates': [], 'episode_lengths': [],
            'losses': [], 'messages': [], 'actions': [], 'nmi_scores': [],
            'message_diversity': [], 'consistency_scores': [], 'communication_efficiency': [],
            'emergence_indicators': [], 'generalization_scores': [], 'stability_scores': []
        }

    def _init_ablation_dashboard(self):
        """Initialize ablation studies monitoring dashboard"""
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.suptitle('Ablation Studies Dashboard - Real-time Comparison',
                         fontsize=16, fontweight='bold')

        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Comparison plots
        self.ax_ablation_success = self.fig.add_subplot(gs[0, 0])
        self.ax_ablation_reward = self.fig.add_subplot(gs[0, 1])
        self.ax_ablation_efficiency = self.fig.add_subplot(gs[0, 2])

        # Category analysis
        self.ax_category_comparison = self.fig.add_subplot(gs[1, :])

        # Detailed analysis
        self.ax_learning_curves = self.fig.add_subplot(gs[2, 0])
        self.ax_communication_comparison = self.fig.add_subplot(gs[2, 1])
        self.ax_ablation_insights = self.fig.add_subplot(gs[2, 2])

        # Data storage for ablation
        self.ablation_data = defaultdict(dict)

    def _init_generalization_dashboard(self):
        """Initialize generalization tests dashboard"""
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.suptitle('Generalization Tests Dashboard - Zero-shot Performance',
                         fontsize=16, fontweight='bold')

        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Generalization performance
        self.ax_gen_overview = self.fig.add_subplot(gs[0, :])

        # Category-specific analysis
        self.ax_grid_gen = self.fig.add_subplot(gs[1, 0])
        self.ax_color_gen = self.fig.add_subplot(gs[1, 1])
        self.ax_complexity_gen = self.fig.add_subplot(gs[1, 2])

        # Communication generalization
        self.ax_comm_gen = self.fig.add_subplot(gs[2, 0])
        self.ax_pattern_gen = self.fig.add_subplot(gs[2, 1])
        self.ax_gen_insights = self.fig.add_subplot(gs[2, 2])

        # Data storage for generalization
        self.generalization_data = defaultdict(dict)

    def _init_comparison_dashboard(self):
        """Initialize comparison dashboard for multiple experiments"""
        self.fig = plt.figure(figsize=(20, 16))
        self.fig.suptitle('Multi-Experiment Comparison Dashboard',
                         fontsize=16, fontweight='bold')

        gs = self.fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

        # Performance comparison
        self.ax_perf_comparison = self.fig.add_subplot(gs[0, :])

        # Training curves comparison
        self.ax_reward_curves = self.fig.add_subplot(gs[1, 0:2])
        self.ax_success_curves = self.fig.add_subplot(gs[1, 2:])

        # Communication comparison
        self.ax_comm_comparison = self.fig.add_subplot(gs[2, 0:2])
        self.ax_diversity_comparison = self.fig.add_subplot(gs[2, 2:])

        # Research insights comparison
        self.ax_emergence_comparison = self.fig.add_subplot(gs[3, 0])
        self.ax_generalization_comparison = self.fig.add_subplot(gs[3, 1])
        self.ax_efficiency_comparison = self.fig.add_subplot(gs[3, 2])
        self.ax_overall_insights = self.fig.add_subplot(gs[3, 3])

        # Data storage for comparison
        self.comparison_data = defaultdict(dict)

    def load_training_data(self, log_file):
        """Load training data from log file"""
        try:
            if not os.path.exists(log_file):
                return False

            with open(log_file, 'r') as f:
                data = json.load(f)

            # Update training data
            for key in self.training_data.keys():
                if key in data:
                    self.training_data[key] = data[key]

            return True
        except Exception as e:
            return False

    def load_ablation_data(self, results_dir):
        """Load ablation study results"""
        try:
            results_dir = Path(results_dir)
            if not results_dir.exists():
                return False

            # Load all category results
            for category_file in results_dir.glob('*_results.json'):
                category = category_file.stem.replace('_results', '')

                with open(category_file, 'r') as f:
                    category_data = json.load(f)

                self.ablation_data[category] = category_data

            return True
        except Exception as e:
            return False

    def load_generalization_data(self, results_file):
        """Load generalization test results"""
        try:
            if not os.path.exists(results_file):
                return False

            with open(results_file, 'r') as f:
                data = json.load(f)

            self.generalization_data = data
            return True
        except Exception as e:
            return False

    def update_training_dashboard(self, frame):
        """Update training dashboard"""
        # Load data
        log_file = self.data_sources.get('training_log', 'training_log.json')
        if not self.load_training_data(log_file):
            self._show_waiting_message("Waiting for training data...")
            return

        # Compute advanced metrics
        self._compute_advanced_training_metrics()

        # Update all plots
        self._update_training_plots()
        self._update_communication_plots()
        self._update_advanced_analysis()
        self._update_research_insights()

        # Update timestamp
        current_time = datetime.now().strftime("%H:%M:%S")
        self.fig.suptitle(f'Enhanced Training Dashboard - Emergent Communication (Updated: {current_time})',
                         fontsize=16, fontweight='bold')

    def update_ablation_dashboard(self, frame):
        """Update ablation studies dashboard"""
        results_dir = self.data_sources.get('ablation_results', 'ablation_results')
        if not self.load_ablation_data(results_dir):
            self._show_waiting_message("Waiting for ablation study results...")
            return

        self._update_ablation_plots()

        current_time = datetime.now().strftime("%H:%M:%S")
        self.fig.suptitle(f'Ablation Studies Dashboard - Real-time Comparison (Updated: {current_time})',
                         fontsize=16, fontweight='bold')

    def update_generalization_dashboard(self, frame):
        """Update generalization tests dashboard"""
        results_file = self.data_sources.get('generalization_results', 'generalization_results/generalization_results.json')
        if not self.load_generalization_data(results_file):
            self._show_waiting_message("Waiting for generalization test results...")
            return

        self._update_generalization_plots()

        current_time = datetime.now().strftime("%H:%M:%S")
        self.fig.suptitle(f'Generalization Tests Dashboard - Zero-shot Performance (Updated: {current_time})',
                         fontsize=16, fontweight='bold')

    def _compute_advanced_training_metrics(self):
        """Compute advanced metrics for training analysis"""
        if not self.training_data['messages'] or not self.training_data['actions']:
            return

        try:
            # Recent data for analysis
            recent_messages = self.training_data['messages'][-200:] if len(self.training_data['messages']) > 200 else self.training_data['messages']
            recent_actions = self.training_data['actions'][-200:] if len(self.training_data['actions']) > 200 else self.training_data['actions']

            if len(recent_messages) < 10:
                return

            # Convert to arrays
            messages_array = np.array(recent_messages)
            actions_array = np.array(recent_actions)

            if messages_array.ndim == 1:
                messages_array = messages_array.reshape(-1, 1)

            # Emergence indicators
            emergence_score = self._compute_emergence_score(messages_array, actions_array)
            self.training_data['emergence_indicators'].append(emergence_score)

            # Generalization proxy (message consistency across different contexts)
            generalization_score = self._compute_generalization_proxy(messages_array, actions_array)
            self.training_data['generalization_scores'].append(generalization_score)

            # Stability score (temporal consistency)
            stability_score = self._compute_stability_score(messages_array)
            self.training_data['stability_scores'].append(stability_score)

        except Exception as e:
            print(f"Warning: Advanced metrics computation failed: {e}")

    def _compute_emergence_score(self, messages, actions):
        """Compute emergence score based on systematicity and compositionality"""
        try:
            # Systematicity: similar messages -> similar actions
            if messages.shape[1] > 1:
                # Compute message similarities
                message_distances = []
                action_distances = []

                for i in range(min(50, len(messages))):
                    for j in range(i+1, min(50, len(messages))):
                        msg_dist = np.linalg.norm(messages[i] - messages[j])
                        act_dist = abs(actions[i] - actions[j])

                        message_distances.append(msg_dist)
                        action_distances.append(act_dist)

                if message_distances and action_distances:
                    correlation = np.corrcoef(message_distances, action_distances)[0, 1]
                    return abs(correlation) if not np.isnan(correlation) else 0.0

            return 0.0
        except:
            return 0.0

    def _compute_generalization_proxy(self, messages, actions):
        """Compute generalization proxy based on message-action consistency"""
        try:
            # Group similar messages and check action consistency
            from sklearn.cluster import KMeans

            n_clusters = min(8, len(set(map(tuple, messages.round(2)))))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(messages)

                # Compute action consistency within clusters
                consistencies = []
                for cluster_id in range(n_clusters):
                    cluster_actions = actions[clusters == cluster_id]
                    if len(cluster_actions) > 1:
                        from collections import Counter
                        action_counts = Counter(cluster_actions)
                        most_common_count = action_counts.most_common(1)[0][1]
                        consistency = most_common_count / len(cluster_actions)
                        consistencies.append(consistency)

                return np.mean(consistencies) if consistencies else 0.0

            return 0.0
        except:
            return 0.0

    def _compute_stability_score(self, messages):
        """Compute temporal stability of messages"""
        try:
            if len(messages) < 20:
                return 0.0

            # Split into early and late periods
            mid = len(messages) // 2
            early_messages = messages[:mid]
            late_messages = messages[mid:]

            # Compute mean difference
            early_mean = np.mean(early_messages, axis=0)
            late_mean = np.mean(late_messages, axis=0)

            stability = 1.0 / (1.0 + np.linalg.norm(early_mean - late_mean))
            return stability
        except:
            return 0.0

    def _update_training_plots(self):
        """Update basic training plots"""
        episodes = self.training_data['episodes']
        if not episodes:
            return

        # Reward plot with advanced analysis
        self.ax_reward.clear()
        rewards = self.training_data['rewards']
        if rewards:
            self.ax_reward.plot(episodes, rewards, 'b-', alpha=0.6, linewidth=1, label='Raw')

            # Add trend analysis
            if len(rewards) > 50:
                window_size = min(50, len(rewards) // 5)
                smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                smooth_episodes = episodes[window_size-1:]
                self.ax_reward.plot(smooth_episodes, smoothed, 'r-', linewidth=2, label=f'Trend (MA{window_size})')

                # Add performance zones
                recent_avg = np.mean(rewards[-50:])
                self.ax_reward.axhline(y=recent_avg, color='green', linestyle='--', alpha=0.7, label=f'Recent Avg: {recent_avg:.1f}')

            self.ax_reward.legend()

        self.ax_reward.set_title('Episode Rewards with Trend Analysis')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True, alpha=0.3)

        # Enhanced success rate plot
        self.ax_success.clear()
        success_rates = self.training_data['success_rates']
        if success_rates:
            success_pct = [s * 100 for s in success_rates]
            self.ax_success.plot(episodes, success_pct, 'g-', linewidth=2, label='Success Rate')

            # Add performance thresholds
            self.ax_success.axhline(y=80, color='gold', linestyle='--', alpha=0.7, label='Good (80%)')
            self.ax_success.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Fair (50%)')
            self.ax_success.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Poor (20%)')

            # Add trend
            if len(success_pct) > 20:
                z = np.polyfit(episodes[-50:], success_pct[-50:], 1)
                p = np.poly1d(z)
                trend_line = p(episodes[-50:])
                self.ax_success.plot(episodes[-50:], trend_line, 'r--', alpha=0.8, label='Trend')

            self.ax_success.legend()

        self.ax_success.set_title('Success Rate with Performance Zones')
        self.ax_success.set_xlabel('Episode')
        self.ax_success.set_ylabel('Success Rate (%)')
        self.ax_success.set_ylim(0, 100)
        self.ax_success.grid(True, alpha=0.3)

        # Episode length with efficiency analysis
        self.ax_length.clear()
        lengths = self.training_data['episode_lengths']
        if lengths:
            self.ax_length.plot(episodes, lengths, 'orange', alpha=0.7, label='Episode Length')

            # Add efficiency indicator
            if len(lengths) > 20:
                recent_avg = np.mean(lengths[-20:])
                overall_avg = np.mean(lengths)
                efficiency = (overall_avg - recent_avg) / overall_avg * 100 if overall_avg > 0 else 0

                self.ax_length.axhline(y=recent_avg, color='red', linestyle='--', alpha=0.7,
                                     label=f'Recent Avg: {recent_avg:.1f}')
                self.ax_length.text(0.02, 0.98, f'Efficiency Gain: {efficiency:.1f}%',
                                  transform=self.ax_length.transAxes, va='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            self.ax_length.legend()

        self.ax_length.set_title('Episode Length & Efficiency')
        self.ax_length.set_xlabel('Episode')
        self.ax_length.set_ylabel('Steps')
        self.ax_length.grid(True, alpha=0.3)

        # Loss plot with convergence analysis
        self.ax_loss.clear()
        losses = self.training_data['losses']
        if losses:
            self.ax_loss.semilogy(episodes, losses, 'purple', alpha=0.7, label='Training Loss')

            # Add convergence indicator
            if len(losses) > 50:
                recent_losses = losses[-20:]
                loss_std = np.std(recent_losses)
                convergence_status = "Converged" if loss_std < 0.1 else "Converging" if loss_std < 0.5 else "Unstable"

                self.ax_loss.text(0.02, 0.98, f'Status: {convergence_status}',
                                transform=self.ax_loss.transAxes, va='top',
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            self.ax_loss.legend()
            self.ax_loss.set_title('Training Loss & Convergence')
        else:
            self.ax_loss.text(0.5, 0.5, 'Loss data\nnot available', ha='center', va='center',
                            transform=self.ax_loss.transAxes)
            self.ax_loss.set_title('Training Loss')

        self.ax_loss.set_xlabel('Episode')
        self.ax_loss.set_ylabel('Loss (log scale)')
        self.ax_loss.grid(True, alpha=0.3)

    def _update_communication_plots(self):
        """Update communication analysis plots"""
        # Enhanced message patterns
        self.ax_messages.clear()
        messages = self.training_data['messages']
        if messages and len(messages) > 0:
            try:
                messages_array = np.array(messages)
                if messages_array.ndim > 1 and messages_array.shape[1] > 1:
                    # Show message evolution over time
                    recent_messages = messages_array[-100:]  # Last 100 messages

                    for dim in range(min(3, messages_array.shape[1])):
                        self.ax_messages.plot(recent_messages[:, dim],
                                            label=f'Dim {dim}', alpha=0.8, linewidth=2)

                    # Add message space coverage indicator
                    message_range = np.ptp(recent_messages, axis=0)  # Peak-to-peak range
                    avg_range = np.mean(message_range)

                    self.ax_messages.text(0.02, 0.98, f'Space Coverage: {avg_range:.2f}',
                                        transform=self.ax_messages.transAxes, va='top',
                                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

                    self.ax_messages.legend()
                    self.ax_messages.set_title('Message Evolution & Space Coverage')
                else:
                    flat_messages = [msg[0] if isinstance(msg, list) and len(msg) > 0 else msg for msg in messages[-100:]]
                    self.ax_messages.plot(flat_messages, 'b-', alpha=0.7, linewidth=2)
                    self.ax_messages.set_title('Message Values (Recent 100)')
            except Exception as e:
                self.ax_messages.text(0.5, 0.5, f'Message visualization\nerror: {str(e)[:30]}...',
                                    ha='center', va='center', transform=self.ax_messages.transAxes)
                self.ax_messages.set_title('Message Patterns (Error)')

        self.ax_messages.set_xlabel('Recent Steps')
        self.ax_messages.set_ylabel('Message Value')
        self.ax_messages.grid(True, alpha=0.3)

        # Enhanced NMI plot with quality assessment
        self.ax_nmi.clear()
        nmi_scores = self.training_data['nmi_scores']
        if nmi_scores:
            self.ax_nmi.plot(nmi_scores, 'r-', linewidth=2, label='NMI Score')

            # Add quality zones
            self.ax_nmi.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Excellent (>0.5)')
            self.ax_nmi.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Good (>0.3)')
            self.ax_nmi.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Poor (>0.1)')

            # Current quality assessment
            if nmi_scores:
                current_nmi = nmi_scores[-1]
                quality = "Excellent" if current_nmi > 0.5 else "Good" if current_nmi > 0.3 else "Fair" if current_nmi > 0.1 else "Poor"

                self.ax_nmi.text(0.02, 0.98, f'Current: {quality}',
                               transform=self.ax_nmi.transAxes, va='top',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

            self.ax_nmi.legend()
            self.ax_nmi.set_ylim(0, 1)

        self.ax_nmi.set_title('Communication Quality (NMI)')
        self.ax_nmi.set_xlabel('Update')
        self.ax_nmi.set_ylabel('NMI Score')
        self.ax_nmi.grid(True, alpha=0.3)

        # Message diversity with optimal range
        self.ax_diversity.clear()
        diversity = self.training_data['message_diversity']
        if diversity:
            self.ax_diversity.plot(diversity, 'g-', linewidth=2, label='Diversity')

            # Add optimal range
            self.ax_diversity.axhspan(0.3, 0.7, alpha=0.2, color='green', label='Optimal Range')

            # Current status
            if diversity:
                current_div = diversity[-1]
                status = "Optimal" if 0.3 <= current_div <= 0.7 else "Too Low" if current_div < 0.3 else "Too High"

                self.ax_diversity.text(0.02, 0.98, f'Status: {status}',
                                     transform=self.ax_diversity.transAxes, va='top',
                                     bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

            self.ax_diversity.legend()
            self.ax_diversity.set_ylim(0, 1)

        self.ax_diversity.set_title('Message Diversity & Optimal Range')
        self.ax_diversity.set_xlabel('Update')
        self.ax_diversity.set_ylabel('Diversity Score')
        self.ax_diversity.grid(True, alpha=0.3)

        # Consistency with stability indicator
        self.ax_consistency.clear()
        consistency = self.training_data['consistency_scores']
        if consistency:
            self.ax_consistency.plot(consistency, 'purple', linewidth=2, label='Consistency')

            # Add stability assessment
            if len(consistency) > 20:
                recent_std = np.std(consistency[-20:])
                stability = "Stable" if recent_std < 0.1 else "Moderate" if recent_std < 0.2 else "Unstable"

                self.ax_consistency.text(0.02, 0.98, f'Stability: {stability}',
                                       transform=self.ax_consistency.transAxes, va='top',
                                       bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

            self.ax_consistency.legend()
            self.ax_consistency.set_ylim(0, 1)

        self.ax_consistency.set_title('Communication Consistency')
        self.ax_consistency.set_xlabel('Update')
        self.ax_consistency.set_ylabel('Consistency Score')
        self.ax_consistency.grid(True, alpha=0.3)

    def _update_advanced_analysis(self):
        """Update advanced analysis plots"""
        # Enhanced heatmap with statistical significance
        self.ax_heatmap.clear()
        messages = self.training_data['messages']
        actions = self.training_data['actions']

        if messages and actions and len(messages) > 50:
            try:
                messages_array = np.array(messages[-200:])
                actions_array = np.array(actions[-200:])

                if messages_array.ndim > 1:
                    n_actions = len(set(actions_array))
                    n_dims = messages_array.shape[1]
                    correlation_matrix = np.zeros((n_actions, n_dims))
                    significance_matrix = np.zeros((n_actions, n_dims))

                    for action in range(n_actions):
                        action_mask = actions_array == action
                        if np.sum(action_mask) > 5:  # Minimum samples for significance
                            action_messages = messages_array[action_mask]
                            correlation_matrix[action] = np.mean(np.abs(action_messages), axis=0)

                            # Simple significance test (variance-based)
                            significance_matrix[action] = 1.0 / (1.0 + np.var(action_messages, axis=0))

                    # Create enhanced heatmap
                    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='viridis',
                               ax=self.ax_heatmap, alpha=0.8,
                               xticklabels=[f'Msg{i}' for i in range(n_dims)],
                               yticklabels=[f'Act{i}' for i in range(n_actions)])

                    # Add significance overlay
                    for i in range(n_actions):
                        for j in range(n_dims):
                            if significance_matrix[i, j] > 0.7:  # High significance
                                self.ax_heatmap.add_patch(plt.Rectangle((j, i), 1, 1,
                                                                      fill=False, edgecolor='red', lw=2))

                    self.ax_heatmap.set_title('Message-Action Correlation (Red=Significant)')
                else:
                    self.ax_heatmap.text(0.5, 0.5, 'Single dimension\nmessages',
                                       ha='center', va='center', transform=self.ax_heatmap.transAxes)
                    self.ax_heatmap.set_title('Message-Action Correlation')
            except Exception as e:
                self.ax_heatmap.text(0.5, 0.5, f'Correlation analysis\nfailed: {str(e)[:30]}...',
                                   ha='center', va='center', transform=self.ax_heatmap.transAxes)
                self.ax_heatmap.set_title('Message-Action Correlation')

        # Enhanced action distribution with entropy
        self.ax_distribution.clear()
        if actions:
            recent_actions = actions[-500:]
            action_counts = np.bincount(recent_actions, minlength=5)

            bars = self.ax_distribution.bar(range(len(action_counts)), action_counts,
                                          color='skyblue', alpha=0.7)

            # Add entropy calculation
            action_probs = action_counts / np.sum(action_counts)
            action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
            max_entropy = np.log(len(action_counts))
            entropy_ratio = action_entropy / max_entropy

            # Color bars based on frequency
            for i, (bar, count) in enumerate(zip(bars, action_counts)):
                if count == max(action_counts):
                    bar.set_color('gold')
                elif count == min(action_counts):
                    bar.set_color('lightcoral')

            self.ax_distribution.text(0.02, 0.98, f'Entropy: {entropy_ratio:.2f}',
                                    transform=self.ax_distribution.transAxes, va='top',
                                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            self.ax_distribution.set_title('Action Distribution & Entropy')
            self.ax_distribution.set_xlabel('Action')
            self.ax_distribution.set_ylabel('Count')

        # Enhanced communication efficiency
        self.ax_efficiency.clear()
        efficiency = self.training_data['communication_efficiency']
        if efficiency:
            self.ax_efficiency.plot(efficiency, 'brown', linewidth=2, label='Efficiency')

            # Add efficiency zones
            if efficiency:
                recent_eff = np.mean(efficiency[-10:]) if len(efficiency) >= 10 else efficiency[-1]
                status = "High" if recent_eff > 2.0 else "Medium" if recent_eff > 1.0 else "Low"

                self.ax_efficiency.text(0.02, 0.98, f'Current: {status}',
                                      transform=self.ax_efficiency.transAxes, va='top',
                                      bbox=dict(boxstyle='round', facecolor='bisque', alpha=0.8))

            self.ax_efficiency.legend()
            self.ax_efficiency.set_title('Communication Efficiency')
            self.ax_efficiency.set_xlabel('Update')
            self.ax_efficiency.set_ylabel('Entropy Score')
            self.ax_efficiency.grid(True, alpha=0.3)

    def _update_research_insights(self):
        """Update research insights plots"""
        # Emergence indicators
        self.ax_emergence.clear()
        emergence = self.training_data['emergence_indicators']
        if emergence:
            self.ax_emergence.plot(emergence, 'darkgreen', linewidth=2, label='Emergence Score')

            # Add emergence phases
            if len(emergence) > 20:
                recent_trend = np.polyfit(range(len(emergence[-20:])), emergence[-20:], 1)[0]
                phase = "Emerging" if recent_trend > 0.01 else "Stable" if abs(recent_trend) < 0.01 else "Declining"

                self.ax_emergence.text(0.02, 0.98, f'Phase: {phase}',
                                     transform=self.ax_emergence.transAxes, va='top',
                                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

            self.ax_emergence.legend()

        self.ax_emergence.set_title('Emergence Indicators')
        self.ax_emergence.set_xlabel('Update')
        self.ax_emergence.set_ylabel('Emergence Score')
        self.ax_emergence.grid(True, alpha=0.3)

        # Generalization proxy
        self.ax_generalization.clear()
        generalization = self.training_data['generalization_scores']
        if generalization:
            self.ax_generalization.plot(generalization, 'navy', linewidth=2, label='Generalization Proxy')

            # Add generalization assessment
            if generalization:
                current_gen = generalization[-1]
                assessment = "Strong" if current_gen > 0.7 else "Moderate" if current_gen > 0.4 else "Weak"

                self.ax_generalization.text(0.02, 0.98, f'Assessment: {assessment}',
                                          transform=self.ax_generalization.transAxes, va='top',
                                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

            self.ax_generalization.legend()

        self.ax_generalization.set_title('Generalization Proxy')
        self.ax_generalization.set_xlabel('Update')
        self.ax_generalization.set_ylabel('Generalization Score')
        self.ax_generalization.grid(True, alpha=0.3)

        # Stability indicators
        self.ax_stability.clear()
        stability = self.training_data['stability_scores']
        if stability:
            self.ax_stability.plot(stability, 'darkorange', linewidth=2, label='Stability Score')

            # Add stability zones
            self.ax_stability.axhspan(0.8, 1.0, alpha=0.2, color='green', label='High Stability')
            self.ax_stability.axhspan(0.6, 0.8, alpha=0.2, color='yellow', label='Medium Stability')
            self.ax_stability.axhspan(0.0, 0.6, alpha=0.2, color='red', label='Low Stability')

            self.ax_stability.legend()
            self.ax_stability.set_ylim(0, 1)

        self.ax_stability.set_title('Communication Stability')
        self.ax_stability.set_xlabel('Update')
        self.ax_stability.set_ylabel('Stability Score')
        self.ax_stability.grid(True, alpha=0.3)

        # Research insights summary
        self.ax_insights.clear()
        self.ax_insights.axis('off')

        # Compile insights
        insights_text = "RESEARCH INSIGHTS\n" + "="*20 + "\n\n"

        # Performance insights
        if self.training_data['success_rates']:
            recent_success = self.training_data['success_rates'][-1] * 100
            insights_text += f"Performance: {recent_success:.1f}%\n"

        # Communication insights
        if self.training_data['nmi_scores']:
            recent_nmi = self.training_data['nmi_scores'][-1]
            comm_quality = "Excellent" if recent_nmi > 0.5 else "Good" if recent_nmi > 0.3 else "Fair"
            insights_text += f"Comm Quality: {comm_quality}\n"

        # Emergence insights
        if emergence:
            recent_emergence = emergence[-1]
            emergence_level = "High" if recent_emergence > 0.6 else "Medium" if recent_emergence > 0.3 else "Low"
            insights_text += f"Emergence: {emergence_level}\n"

        # Stability insights
        if stability:
            recent_stability = stability[-1]
            stability_level = "Stable" if recent_stability > 0.7 else "Moderate" if recent_stability > 0.5 else "Unstable"
            insights_text += f"Stability: {stability_level}\n"

        # Add recommendations
        insights_text += "\nRECOMMENDATIONS:\n"

        if self.training_data['success_rates'] and self.training_data['success_rates'][-1] < 0.3:
            insights_text += "• Extend training\n"

        if self.training_data['nmi_scores'] and self.training_data['nmi_scores'][-1] < 0.2:
            insights_text += "• Improve comm protocol\n"

        if emergence and emergence[-1] < 0.3:
            insights_text += "• Encourage emergence\n"

        self.ax_insights.text(0.05, 0.95, insights_text, transform=self.ax_insights.transAxes,
                            va='top', ha='left', fontsize=10, fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        self.ax_insights.set_title('Research Insights & Recommendations')

    def _update_ablation_plots(self):
        """Update ablation studies plots"""
        if not self.ablation_data:
            return

        # Success rate comparison
        self.ax_ablation_success.clear()
        categories = []
        best_success_rates = []

        for category, experiments in self.ablation_data.items():
            if experiments:
                success_rates = []
                for exp_name, result in experiments.items():
                    if 'error' not in result:
                        success_rates.append(result.get('final_success_rate', 0))

                if success_rates:
                    categories.append(category.title())
                    best_success_rates.append(max(success_rates))

        if categories:
            bars = self.ax_ablation_success.bar(range(len(categories)), best_success_rates,
                                              color='skyblue', alpha=0.7)

            # Highlight best performing category
            if best_success_rates:
                best_idx = np.argmax(best_success_rates)
                bars[best_idx].set_color('gold')

            self.ax_ablation_success.set_title('Best Success Rate per Category')
            self.ax_ablation_success.set_xticks(range(len(categories)))
            self.ax_ablation_success.set_xticklabels(categories, rotation=45, ha='right')
            self.ax_ablation_success.set_ylabel('Success Rate (%)')
            self.ax_ablation_success.grid(True, alpha=0.3)

    def _update_generalization_plots(self):
        """Update generalization test plots"""
        if not self.generalization_data:
            return

        # Overview of generalization performance
        self.ax_gen_overview.clear()

        scenario_names = []
        success_rates = []
        categories = []

        for category, scenarios in self.generalization_data.items():
            for scenario_name, result in scenarios.items():
                if 'error' not in result:
                    scenario_names.append(f"{category}_{scenario_name}")
                    success_rates.append(result['success_rate'] * 100)
                    categories.append(category)

        if scenario_names:
            # Color by category
            colors = plt.cm.Set3(np.linspace(0, 1, len(set(categories))))
            category_colors = {cat: colors[i] for i, cat in enumerate(set(categories))}
            bar_colors = [category_colors[cat] for cat in categories]

            bars = self.ax_gen_overview.bar(range(len(scenario_names)), success_rates,
                                          color=bar_colors, alpha=0.7)

            self.ax_gen_overview.set_title('Generalization Performance Overview')
            self.ax_gen_overview.set_xticks(range(len(scenario_names)))
            self.ax_gen_overview.set_xticklabels(scenario_names, rotation=45, ha='right')
            self.ax_gen_overview.set_ylabel('Success Rate (%)')
            self.ax_gen_overview.grid(True, alpha=0.3)

            # Add legend
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=category_colors[cat],
                                           alpha=0.7, label=cat.title())
                             for cat in category_colors.keys()]
            self.ax_gen_overview.legend(handles=legend_elements, loc='upper right')

    def _show_waiting_message(self, message):
        """Show waiting message on all axes"""
        for ax in self.fig.get_axes():
            ax.clear()
            ax.text(0.5, 0.5, message, ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Waiting for Data')

    def run(self):
        """Start the dashboard"""
        print(f"🚀 Starting Enhanced Research Dashboard in {self.mode} mode...")

        # Select update function based on mode
        if self.mode == 'training':
            update_func = self.update_training_dashboard
        elif self.mode == 'ablation':
            update_func = self.update_ablation_dashboard
        elif self.mode == 'generalization':
            update_func = self.update_generalization_dashboard
        else:
            update_func = self.update_training_dashboard  # Default

        # Create animation
        ani = FuncAnimation(self.fig, update_func, interval=self.update_interval,
                           cache_frame_data=False)

        # Show dashboard
        plt.show()

        return ani

def main():
    """Main function with enhanced command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Research Dashboard for Emergent Communication')
    parser.add_argument('--mode', type=str,
                       choices=['training', 'ablation', 'generalization', 'comparison'],
                       default='training', help='Dashboard mode')
    parser.add_argument('--training_log', type=str, default='training_log.json',
                       help='Path to training log file')
    parser.add_argument('--ablation_results', type=str, default='ablation_results',
                       help='Path to ablation results directory')
    parser.add_argument('--generalization_results', type=str,
                       default='generalization_results/generalization_results.json',
                       help='Path to generalization results file')
    parser.add_argument('--update_interval', type=int, default=3000,
                       help='Update interval in milliseconds')

    args = parser.parse_args()

    # Prepare data sources
    data_sources = {
        'training_log': args.training_log,
        'ablation_results': args.ablation_results,
        'generalization_results': args.generalization_results
    }

    # Create and run dashboard
    dashboard = EnhancedResearchDashboard(
        mode=args.mode,
        data_sources=data_sources,
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