"""
Advanced Interpretability Analysis for Emergent Communication
Provides additional tools for analyzing communication structure and emergence
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from itertools import combinations

class InterpretabilityAnalyzer:
    def __init__(self, messages, actions, states=None, rewards=None, contexts=None):
        """
        Initialize with collected communication data

        Args:
            messages: List/array of communication messages
            actions: List/array of corresponding actions
            states: Optional list/array of environment states
            rewards: Optional list/array of rewards
            contexts: Optional list/array of contexts
        """
        self.messages = np.array(messages)
        self.actions = np.array(actions)
        self.states = np.array(states) if states is not None else None
        self.rewards = np.array(rewards) if rewards is not None else None
        self.contexts = np.array(contexts) if contexts is not None else None

        # Ensure messages are 2D
        if self.messages.ndim == 1:
            self.messages = self.messages.reshape(-1, 1)

    def analyze_compositionality(self, n_components=None):
        """
        Analyze compositional structure in messages
        Tests if different message dimensions encode different types of information
        """
        if self.messages.shape[1] < 2:
            return {'compositionality_score': 0.0, 'component_analysis': {}}

        n_components = n_components or min(self.messages.shape[1], 5)
        component_analysis = {}

        # Analyze each message dimension
        for dim in range(min(n_components, self.messages.shape[1])):
            dim_values = self.messages[:, dim]

            # Compute NMI with different targets
            dim_analysis = {}

            if self.actions is not None:
                dim_analysis['action_nmi'] = normalized_mutual_info_score(
                    np.digitize(dim_values, bins=np.percentile(dim_values, [25, 50, 75])),
                    self.actions
                )

            if self.states is not None:
                # Use first few state dimensions for analysis
                state_flat = self.states.reshape(len(self.states), -1)
                for state_dim in range(min(3, state_flat.shape[1])):
                    state_values = state_flat[:, state_dim]
                    dim_analysis[f'state_dim_{state_dim}_nmi'] = normalized_mutual_info_score(
                        np.digitize(dim_values, bins=np.percentile(dim_values, [25, 50, 75])),
                        np.digitize(state_values, bins=np.percentile(state_values, [25, 50, 75]))
                    )

            if self.rewards is not None:
                dim_analysis['reward_nmi'] = normalized_mutual_info_score(
                    np.digitize(dim_values, bins=np.percentile(dim_values, [25, 50, 75])),
                    np.digitize(self.rewards, bins=np.percentile(self.rewards, [25, 50, 75]))
                )

            component_analysis[f'dimension_{dim}'] = dim_analysis

        # Compute overall compositionality score
        # Higher score means different dimensions encode different information
        all_nmis = []
        for dim_analysis in component_analysis.values():
            all_nmis.extend(dim_analysis.values())

        compositionality_score = np.std(all_nmis) if all_nmis else 0.0

        return {
            'compositionality_score': compositionality_score,
            'component_analysis': component_analysis
        }

    def analyze_message_structure(self):
        """
        Analyze the structure and patterns in messages
        """
        structure_analysis = {}

        # Message diversity
        unique_messages = len(set(map(tuple, self.messages.round(2))))
        structure_analysis['message_diversity'] = unique_messages / len(self.messages)

        # Message entropy (information content)
        message_entropies = []
        for dim in range(self.messages.shape[1]):
            dim_values = self.messages[:, dim]
            hist, _ = np.histogram(dim_values, bins=10)
            hist = hist / np.sum(hist)
            dim_entropy = entropy(hist + 1e-8)
            message_entropies.append(dim_entropy)

        structure_analysis['avg_dimension_entropy'] = np.mean(message_entropies)
        structure_analysis['dimension_entropies'] = message_entropies

        # Message clustering analysis
        n_clusters = min(8, unique_messages)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            message_clusters = kmeans.fit_predict(self.messages)

            # Analyze cluster-action relationships
            cluster_action_nmi = normalized_mutual_info_score(message_clusters, self.actions)
            structure_analysis['cluster_action_alignment'] = cluster_action_nmi

            # Cluster sizes (uniformity)
            cluster_sizes = np.bincount(message_clusters)
            cluster_uniformity = 1.0 - np.std(cluster_sizes) / np.mean(cluster_sizes)
            structure_analysis['cluster_uniformity'] = cluster_uniformity

        return structure_analysis

    def analyze_communication_efficiency(self):
        """
        Analyze how efficiently the communication system uses its capacity
        """
        efficiency_metrics = {}

        # Theoretical maximum information (uniform distribution)
        vocab_size = len(set(map(tuple, self.messages.round(2))))
        max_entropy = np.log2(vocab_size) if vocab_size > 1 else 0

        # Actual information content
        message_tuples = [tuple(msg.round(2)) for msg in self.messages]
        message_counts = Counter(message_tuples)
        message_probs = np.array(list(message_counts.values())) / len(self.messages)
        actual_entropy = entropy(message_probs, base=2)

        # Efficiency ratio
        efficiency_metrics['entropy_efficiency'] = actual_entropy / max_entropy if max_entropy > 0 else 0

        # Task-relevant information (NMI with actions)
        if self.actions is not None:
            # Discretize messages for NMI computation
            kmeans = KMeans(n_clusters=min(8, vocab_size), random_state=42, n_init=10)
            message_labels = kmeans.fit_predict(self.messages)

            task_relevance = normalized_mutual_info_score(message_labels, self.actions)
            efficiency_metrics['task_relevance'] = task_relevance

            # Communication success rate (how often same message leads to same action)
            message_action_consistency = []
            message_groups = defaultdict(list)
            for msg_label, action in zip(message_labels, self.actions):
                message_groups[msg_label].append(action)

            for actions_for_msg in message_groups.values():
                if len(actions_for_msg) > 1:
                    most_common = Counter(actions_for_msg).most_common(1)[0][1]
                    consistency = most_common / len(actions_for_msg)
                    message_action_consistency.append(consistency)

            efficiency_metrics['message_action_consistency'] = np.mean(message_action_consistency) if message_action_consistency else 0.0

        return efficiency_metrics

    def analyze_emergence_indicators(self):
        """
        Analyze indicators of emergent communication properties
        """
        emergence_metrics = {}

        # Systematicity: Do similar contexts produce similar messages?
        if self.contexts is not None or self.states is not None:
            contexts = self.contexts if self.contexts is not None else self.states
            context_flat = contexts.reshape(len(contexts), -1)

            # Compute context similarities
            context_distances = pdist(context_flat, metric='euclidean')
            context_sim_matrix = 1.0 / (1.0 + squareform(context_distances))

            # Compute message similarities
            message_distances = pdist(self.messages, metric='euclidean')
            message_sim_matrix = 1.0 / (1.0 + squareform(message_distances))

            # Correlation between context and message similarities
            context_sim_flat = context_sim_matrix[np.triu_indices_from(context_sim_matrix, k=1)]
            message_sim_flat = message_sim_matrix[np.triu_indices_from(message_sim_matrix, k=1)]

            systematicity = np.corrcoef(context_sim_flat, message_sim_flat)[0, 1]
            emergence_metrics['systematicity'] = systematicity if not np.isnan(systematicity) else 0.0

        # Productivity: Are messages used in novel combinations?
        if self.messages.shape[1] > 1:
            # Analyze co-occurrence patterns in message dimensions
            dimension_combinations = []
            for i, j in combinations(range(self.messages.shape[1]), 2):
                dim_i_discrete = np.digitize(self.messages[:, i], bins=np.percentile(self.messages[:, i], [33, 67]))
                dim_j_discrete = np.digitize(self.messages[:, j], bins=np.percentile(self.messages[:, j], [33, 67]))

                # Count unique combinations
                combinations_seen = set(zip(dim_i_discrete, dim_j_discrete))
                total_possible = len(set(dim_i_discrete)) * len(set(dim_j_discrete))

                productivity = len(combinations_seen) / total_possible if total_possible > 0 else 0
                dimension_combinations.append(productivity)

            emergence_metrics['productivity'] = np.mean(dimension_combinations) if dimension_combinations else 0.0

        # Stability: Do messages remain consistent over time?
        if len(self.messages) > 10:
            # Split into early and late periods
            split_point = len(self.messages) // 2
            early_messages = self.messages[:split_point]
            late_messages = self.messages[split_point:]

            # Compare message distributions
            early_mean = np.mean(early_messages, axis=0)
            late_mean = np.mean(late_messages, axis=0)

            stability = 1.0 / (1.0 + np.linalg.norm(early_mean - late_mean))
            emergence_metrics['temporal_stability'] = stability

        return emergence_metrics

    def generate_interpretability_report(self):
        """
        Generate comprehensive interpretability report
        """
        print("Generating interpretability analysis...")

        # Run all analyses
        compositionality = self.analyze_compositionality()
        structure = self.analyze_message_structure()
        efficiency = self.analyze_communication_efficiency()
        emergence = self.analyze_emergence_indicators()

        # Compile report
        report = {
            'compositionality_analysis': compositionality,
            'message_structure': structure,
            'communication_efficiency': efficiency,
            'emergence_indicators': emergence,
            'summary_scores': {
                'overall_interpretability': np.mean([
                    compositionality.get('compositionality_score', 0),
                    structure.get('message_diversity', 0),
                    efficiency.get('task_relevance', 0),
                    emergence.get('systematicity', 0)
                ]),
                'communication_quality': np.mean([
                    efficiency.get('task_relevance', 0),
                    efficiency.get('message_action_consistency', 0),
                    structure.get('cluster_action_alignment', 0)
                ]),
                'emergence_strength': np.mean([
                    emergence.get('systematicity', 0),
                    emergence.get('productivity', 0),
                    emergence.get('temporal_stability', 0)
                ])
            }
        }

        return report

    def visualize_interpretability(self, save_path=None):
        """
        Create visualizations for interpretability analysis
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Message space visualization
        if self.messages.shape[1] >= 2:
            axes[0, 0].scatter(self.messages[:, 0], self.messages[:, 1],
                             c=self.actions, cmap='tab10', alpha=0.6)
            axes[0, 0].set_title('Message Space (colored by Action)')
            axes[0, 0].set_xlabel('Message Dim 0')
            axes[0, 0].set_ylabel('Message Dim 1')

        # 2. Message-Action correlation heatmap
        if self.messages.shape[1] > 1:
            correlation_matrix = np.zeros((len(set(self.actions)), self.messages.shape[1]))
            for action in set(self.actions):
                action_mask = self.actions == action
                action_messages = self.messages[action_mask]
                if len(action_messages) > 0:
                    correlation_matrix[action] = np.mean(np.abs(action_messages), axis=0)

            sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                       ax=axes[0, 1], cmap='viridis')
            axes[0, 1].set_title('Action-Message Dimension Correlation')
            axes[0, 1].set_xlabel('Message Dimension')
            axes[0, 1].set_ylabel('Action')

        # 3. Message entropy by dimension
        entropies = []
        for dim in range(self.messages.shape[1]):
            hist, _ = np.histogram(self.messages[:, dim], bins=10)
            hist = hist / np.sum(hist)
            dim_entropy = entropy(hist + 1e-8)
            entropies.append(dim_entropy)

        axes[0, 2].bar(range(len(entropies)), entropies)
        axes[0, 2].set_title('Information Content by Message Dimension')
        axes[0, 2].set_xlabel('Message Dimension')
        axes[0, 2].set_ylabel('Entropy')

        # 4. Message clustering
        n_clusters = min(8, len(set(map(tuple, self.messages.round(2)))))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(self.messages)

            if self.messages.shape[1] >= 2:
                scatter = axes[1, 0].scatter(self.messages[:, 0], self.messages[:, 1],
                                           c=clusters, cmap='tab10', alpha=0.6)
                axes[1, 0].set_title('Message Clusters')
                axes[1, 0].set_xlabel('Message Dim 0')
                axes[1, 0].set_ylabel('Message Dim 1')

        # 5. Action distribution by message cluster
        if n_clusters > 1:
            cluster_action_matrix = np.zeros((n_clusters, len(set(self.actions))))
            for cluster in range(n_clusters):
                cluster_mask = clusters == cluster
                cluster_actions = self.actions[cluster_mask]
                for action in cluster_actions:
                    cluster_action_matrix[cluster, action] += 1

            # Normalize by cluster size
            cluster_sizes = np.bincount(clusters)
            for cluster in range(n_clusters):
                if cluster_sizes[cluster] > 0:
                    cluster_action_matrix[cluster] /= cluster_sizes[cluster]

            sns.heatmap(cluster_action_matrix, annot=True, fmt='.2f',
                       ax=axes[1, 1], cmap='Blues')
            axes[1, 1].set_title('Action Distribution by Message Cluster')
            axes[1, 1].set_xlabel('Action')
            axes[1, 1].set_ylabel('Message Cluster')

        # 6. Communication efficiency over time
        if len(self.messages) > 20:
            window_size = len(self.messages) // 10
            efficiency_over_time = []

            for i in range(window_size, len(self.messages), window_size):
                window_messages = self.messages[i-window_size:i]
                window_actions = self.actions[i-window_size:i]

                # Compute NMI for this window
                kmeans_window = KMeans(n_clusters=min(5, len(set(map(tuple, window_messages.round(2))))),
                                     random_state=42, n_init=10)
                window_clusters = kmeans_window.fit_predict(window_messages)
                window_nmi = normalized_mutual_info_score(window_clusters, window_actions)
                efficiency_over_time.append(window_nmi)

            axes[1, 2].plot(efficiency_over_time, 'b-', linewidth=2)
            axes[1, 2].set_title('Communication Efficiency Over Time')
            axes[1, 2].set_xlabel('Time Window')
            axes[1, 2].set_ylabel('Message-Action NMI')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Interpretability visualizations saved to {save_path}")

        return fig