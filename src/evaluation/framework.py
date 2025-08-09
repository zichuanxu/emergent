import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

class EvaluationFramework:
    def __init__(self, model, data_loader, vocab_size=8, message_length=3):
        self.model = model
        self.data_loader = data_loader
        self.vocab_size = vocab_size
        self.message_length = message_length
        self.collected_data = []

    def collect_interaction_data(self, num_episodes=100):
        """Collect communication data from model interactions"""
        print(f"Collecting data from {num_episodes} episodes...")

        self.collected_data = []
        for episode in range(num_episodes):
            episode_data = {
                'messages': [],
                'actions': [],
                'states': [],
                'rewards': [],
                'contexts': []
            }

            for batch_idx, batch in enumerate(self.data_loader):
                if batch_idx >= num_episodes:
                    break

                with torch.no_grad():
                    if isinstance(batch, dict):
                        inputs = batch.get('inputs', batch.get('state'))
                        targets = batch.get('targets', batch.get('actions'))
                        context = batch.get('context', None)
                    else:
                        inputs, targets = batch
                        context = None

                    # Get model outputs (messages, actions, etc.)
                    outputs = self.model(inputs)

                    if isinstance(outputs, tuple):
                        messages = outputs[0] if len(outputs) > 0 else None
                        actions = outputs[1] if len(outputs) > 1 else targets
                    else:
                        messages = outputs
                        actions = targets

                    # Store data
                    if messages is not None:
                        episode_data['messages'].extend(messages.cpu().numpy())
                    if actions is not None:
                        episode_data['actions'].extend(actions.cpu().numpy() if torch.is_tensor(actions) else actions)
                    if inputs is not None:
                        episode_data['states'].extend(inputs.cpu().numpy())
                    if context is not None:
                        episode_data['contexts'].extend(context.cpu().numpy() if torch.is_tensor(context) else context)

            if episode_data['messages']:  # Only add if we have data
                self.collected_data.append(episode_data)

        print(f"Collected data from {len(self.collected_data)} episodes")

    def compute_nmi(self, messages, targets, target_type='actions'):
        """
        Compute Normalized Mutual Information between messages and targets

        Args:
            messages: Array of communication messages
            targets: Array of target variables (actions, states, rewards, etc.)
            target_type: Type of target for appropriate discretization
        """
        if len(messages) == 0 or len(targets) == 0:
            return 0.0

        # Convert messages to discrete labels
        messages_array = np.array(messages)
        if messages_array.ndim > 1:
            # For multi-dimensional messages, use clustering
            n_clusters = min(10, len(set(map(tuple, messages_array.round(2)))))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                message_labels = kmeans.fit_predict(messages_array)
            else:
                message_labels = np.zeros(len(messages))
        else:
            # For 1D messages, discretize directly
            message_labels = np.digitize(messages_array, bins=np.percentile(messages_array, [25, 50, 75]))

        # Discretize targets based on type
        if target_type == 'actions':
            target_labels = np.array(targets).astype(int)
        elif target_type == 'rewards':
            target_labels = np.digitize(targets, bins=np.percentile(targets, [25, 50, 75]))
        elif target_type == 'states':
            # For states, use clustering
            states_array = np.array(targets)
            if states_array.ndim > 1:
                states_flat = states_array.reshape(len(states_array), -1)
                n_clusters = min(8, len(set(map(tuple, states_flat.round(1)))))
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    target_labels = kmeans.fit_predict(states_flat)
                else:
                    target_labels = np.zeros(len(states_array))
            else:
                target_labels = np.digitize(states_array, bins=np.percentile(states_array, [25, 50, 75]))
        else:
            target_labels = np.array(targets)

        # Compute NMI
        try:
            if len(set(message_labels)) > 1 and len(set(target_labels)) > 1:
                nmi = normalized_mutual_info_score(message_labels, target_labels)
                return nmi
            else:
                return 0.0
        except Exception as e:
            print(f"Warning: NMI computation failed: {e}")
            return 0.0

    def consistency_test(self):
        """
        Perform comprehensive consistency tests on communication
        """
        if not self.collected_data:
            print("No data collected. Running data collection first...")
            self.collect_interaction_data()

        consistency_results = {}

        # Test 1: Message Stability (same context -> similar message)
        context_message_pairs = []
        for episode in self.collected_data:
            contexts = episode.get('contexts', episode.get('states', []))
            messages = episode.get('messages', [])

            for context, message in zip(contexts, messages):
                if context is not None and message is not None:
                    # Discretize context for grouping
                    if isinstance(context, np.ndarray):
                        context_key = tuple(np.round(context.flatten(), 1))
                    else:
                        context_key = tuple(np.round(np.array(context).flatten(), 1))
                    context_message_pairs.append((context_key, message))

        # Group by context and compute message consistency
        context_groups = defaultdict(list)
        for context, message in context_message_pairs:
            context_groups[context].append(message)

        stability_scores = []
        for context, messages in context_groups.items():
            if len(messages) > 1:
                messages_array = np.array(messages)
                # Compute pairwise cosine similarities
                if messages_array.ndim > 1:
                    similarities = []
                    for i in range(len(messages)):
                        for j in range(i+1, len(messages)):
                            sim = np.dot(messages_array[i], messages_array[j]) / (
                                np.linalg.norm(messages_array[i]) * np.linalg.norm(messages_array[j]) + 1e-8
                            )
                            similarities.append(sim)
                    if similarities:
                        stability_scores.append(np.mean(similarities))

        consistency_results['message_stability'] = np.mean(stability_scores) if stability_scores else 0.0

        # Test 2: Action Consistency (similar messages -> similar actions)
        message_action_pairs = []
        for episode in self.collected_data:
            messages = episode.get('messages', [])
            actions = episode.get('actions', [])

            for message, action in zip(messages, actions):
                if message is not None and action is not None:
                    message_key = tuple(np.round(np.array(message).flatten(), 1))
                    message_action_pairs.append((message_key, action))

        # Group by message and compute action consistency
        message_groups = defaultdict(list)
        for message, action in message_action_pairs:
            message_groups[message].append(action)

        action_consistency_scores = []
        for message, actions in message_groups.items():
            if len(actions) > 1:
                # For discrete actions, compute mode frequency
                if all(isinstance(a, (int, np.integer)) for a in actions):
                    action_counts = Counter(actions)
                    most_common_count = action_counts.most_common(1)[0][1]
                    consistency = most_common_count / len(actions)
                else:
                    # For continuous actions, compute variance
                    actions_array = np.array(actions)
                    consistency = 1.0 / (1.0 + np.var(actions_array))
                action_consistency_scores.append(consistency)

        consistency_results['action_consistency'] = np.mean(action_consistency_scores) if action_consistency_scores else 0.0

        # Test 3: Compositional Consistency (message parts should be consistent)
        if self.message_length > 1:
            compositional_scores = []
            all_messages = []
            for episode in self.collected_data:
                all_messages.extend(episode.get('messages', []))

            if all_messages:
                messages_array = np.array(all_messages)
                if messages_array.ndim > 1 and messages_array.shape[1] >= 2:
                    # Check if different parts of messages are used consistently
                    for dim in range(messages_array.shape[1]):
                        dim_values = messages_array[:, dim]
                        # Compute entropy (lower entropy = more consistent usage)
                        hist, _ = np.histogram(dim_values, bins=10)
                        hist = hist / np.sum(hist)
                        dim_entropy = entropy(hist + 1e-8)
                        # Convert to consistency score (higher = more consistent)
                        consistency = 1.0 / (1.0 + dim_entropy)
                        compositional_scores.append(consistency)

            consistency_results['compositional_consistency'] = np.mean(compositional_scores) if compositional_scores else 0.0

        return consistency_results

    def compute_interpretability_metrics(self):
        """Compute comprehensive interpretability metrics"""
        if not self.collected_data:
            self.collect_interaction_data()

        metrics = {}

        # Collect all data
        all_messages = []
        all_actions = []
        all_states = []
        all_rewards = []

        for episode in self.collected_data:
            all_messages.extend(episode.get('messages', []))
            all_actions.extend(episode.get('actions', []))
            all_states.extend(episode.get('states', []))
            all_rewards.extend(episode.get('rewards', []))

        # NMI Scores
        if all_messages and all_actions:
            metrics['nmi_message_action'] = self.compute_nmi(all_messages, all_actions, 'actions')

        if all_messages and all_states:
            metrics['nmi_message_state'] = self.compute_nmi(all_messages, all_states, 'states')

        if all_messages and all_rewards:
            metrics['nmi_message_reward'] = self.compute_nmi(all_messages, all_rewards, 'rewards')

        # Message Diversity
        if all_messages:
            messages_array = np.array(all_messages)
            unique_messages = len(set(map(tuple, messages_array.round(2))))
            metrics['message_diversity'] = unique_messages / len(all_messages)

        # Communication Efficiency (how much information is conveyed)
        if all_messages and all_actions:
            # Compute mutual information between messages and actions
            messages_array = np.array(all_messages)
            if messages_array.ndim > 1:
                # Use PCA to reduce dimensionality for efficiency computation
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(3, messages_array.shape[1]))
                messages_reduced = pca.fit_transform(messages_array)
                metrics['communication_efficiency'] = self.compute_nmi(messages_reduced, all_actions, 'actions')
            else:
                metrics['communication_efficiency'] = self.compute_nmi(all_messages, all_actions, 'actions')

        return metrics

    def evaluate(self):
        """Main evaluation function"""
        print("Running comprehensive evaluation...")

        # Collect data if not already done
        if not self.collected_data:
            self.collect_interaction_data()

        # Compute all metrics
        nmi_metrics = self.compute_interpretability_metrics()
        consistency_metrics = self.consistency_test()

        # Combine results
        results = {
            'interpretability_metrics': nmi_metrics,
            'consistency_metrics': consistency_metrics,
            'overall_score': {
                'communication_effectiveness': np.mean([v for v in nmi_metrics.values() if 'nmi' in str(v)]) if nmi_metrics else 0.0,
                'behavioral_consistency': np.mean(list(consistency_metrics.values())) if consistency_metrics else 0.0
            }
        }

        return results