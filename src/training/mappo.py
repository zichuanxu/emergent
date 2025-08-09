import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque

# Experience tuple for storing trajectory data
Experience = namedtuple('Experience', [
    'architect_obs', 'builder_obs', 'carrying_state',
    'message', 'action', 'reward', 'done',
    'architect_value', 'builder_value',
    'action_log_prob', 'message_log_prob'
])

class RolloutBuffer:
    """Buffer for storing and processing MAPPO rollouts."""

    def __init__(self, buffer_size, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        """Reset the buffer."""
        self.experiences = []
        self.returns = []
        self.advantages = []

    def add(self, experience):
        """Add an experience to the buffer."""
        self.experiences.append(experience)

    def compute_returns_and_advantages(self):
        """Compute returns and advantages using GAE."""
        rewards = [exp.reward for exp in self.experiences]
        values = [exp.architect_value + exp.builder_value for exp in self.experiences]  # Joint value
        dones = [exp.done for exp in self.experiences]

        # Compute GAE advantages
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]

            # Convert boolean done to float for arithmetic operations
            done_mask = float(dones[t]) if isinstance(dones[t], bool) else dones[t].float()

            delta = rewards[t] + self.gamma * next_value * (1 - done_mask) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - done_mask) * gae
            advantages.insert(0, gae)

        # Compute returns
        returns = []
        for t in range(len(rewards)):
            returns.append(advantages[t] + values[t])

        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size):
        """Get mini-batches for training."""
        indices = np.random.permutation(len(self.experiences))

        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            batch_experiences = [self.experiences[i] for i in batch_indices]
            batch_advantages = self.advantages[batch_indices]
            batch_returns = self.returns[batch_indices]

            yield batch_experiences, batch_advantages, batch_returns

class MAPPO:
    """Multi-Agent Proximal Policy Optimization trainer."""

    def __init__(self, architect, builder, config):
        self.architect = architect
        self.builder = builder
        self.config = config

        # Optimizers
        self.architect_optimizer = torch.optim.Adam(
            architect.parameters(),
            lr=config['learning_rate']
        )
        self.builder_optimizer = torch.optim.Adam(
            builder.parameters(),
            lr=config['learning_rate']
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=config['batch_size'] * 4,
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda']
        )

        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)

    def select_actions(self, architect_obs, builder_obs, carrying_state,
                      hidden_state=None, training=True):
        """Select actions for both agents."""

        # Architect generates message
        message, architect_value, message_logits = self.architect(
            architect_obs.unsqueeze(0),
            temperature=self.config['gumbel_temperature'],
            hard=self.config['gumbel_hard'] and not training
        )

        # Builder selects action based on message and observation
        action_logits, builder_value, new_hidden_state = self.builder(
            builder_obs.unsqueeze(0),
            message,
            carrying_state.unsqueeze(0) if carrying_state is not None else None,
            hidden_state
        )

        if training:
            # Check for NaN in logits and handle
            if torch.isnan(action_logits).any():
                print(f"Warning: NaN detected in action_logits, resetting...")
                action_logits = torch.zeros_like(action_logits)

            # Sample actions with numerical stability
            action_logits_clamped = torch.clamp(action_logits, min=-20, max=20)
            action_probs = F.softmax(action_logits_clamped, dim=-1)

            # Add epsilon and renormalize to ensure valid probability distribution
            eps = 1e-8
            action_probs = action_probs + eps
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

            # Double-check for NaN after processing
            if torch.isnan(action_probs).any():
                print(f"Warning: NaN detected in action_probs, using uniform distribution")
                action_probs = torch.ones_like(action_probs) / action_probs.shape[-1]

            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            action_log_prob = action_dist.log_prob(action)

            # Message log probability (for Gumbel-Softmax)
            if torch.isnan(message_logits).any():
                print(f"Warning: NaN detected in message_logits, resetting...")
                message_logits = torch.zeros_like(message_logits)

            message_logits_clamped = torch.clamp(message_logits, min=-20, max=20)
            message_probs = F.softmax(message_logits_clamped, dim=-1) + eps
            message_probs = message_probs / message_probs.sum(dim=-1, keepdim=True)

            # Approximate log prob for Gumbel-Softmax
            message_log_prob = torch.sum(message * torch.log(message_probs + eps), dim=-1).sum()
        else:
            # Deterministic actions for evaluation
            action = torch.argmax(action_logits, dim=-1)
            action_log_prob = torch.zeros_like(action, dtype=torch.float32)
            message_log_prob = torch.zeros(1, dtype=torch.float32)

        return {
            'message': message.squeeze(0),
            'action': action.squeeze(0),
            'architect_value': architect_value.squeeze(0),
            'builder_value': builder_value.squeeze(0),
            'action_log_prob': action_log_prob.squeeze(0),
            'message_log_prob': message_log_prob,
            'hidden_state': new_hidden_state
        }

    def update(self):
        """Update both agents using MAPPO."""

        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages()

        # Track losses for logging
        total_losses = []

        # Training loop
        for epoch in range(self.config['ppo_epochs']):
            for batch_experiences, batch_advantages, batch_returns in self.buffer.get_batches(
                self.config['batch_size']
            ):
                loss = self._update_batch(batch_experiences, batch_advantages, batch_returns)
                if loss is not None:
                    total_losses.append(loss)

        # Clear buffer
        self.buffer.reset()

        # Return average loss for logging
        return np.mean(total_losses) if total_losses else None

    def _update_batch(self, experiences, advantages, returns):
        """Update networks on a batch of experiences."""

        # Prepare batch data
        device = experiences[0].architect_obs.device  # Get device from first experience
        architect_obs = torch.stack([exp.architect_obs for exp in experiences])
        builder_obs = torch.stack([exp.builder_obs for exp in experiences])
        carrying_states = torch.stack([
            exp.carrying_state.clone().detach() if exp.carrying_state is not None else torch.tensor(0, device=device)
            for exp in experiences
        ]).to(device)

        # Move advantages and returns to the same device
        advantages = advantages.to(device)
        returns = returns.to(device)

        old_messages = torch.stack([exp.message.detach() for exp in experiences])
        old_actions = torch.stack([exp.action for exp in experiences])
        old_action_log_probs = torch.stack([exp.action_log_prob.detach() for exp in experiences])
        old_message_log_probs = torch.stack([exp.message_log_prob.detach() for exp in experiences])

        # Forward pass for Architect
        new_messages, architect_values, message_logits = self.architect(
            architect_obs,
            temperature=self.config['gumbel_temperature'],
            hard=False  # Use soft during training
        )

        # Forward pass for Builder
        action_logits, builder_values, _ = self.builder(
            builder_obs,
            old_messages,  # Use old messages for stable training
            carrying_states
        )

        # Compute new log probabilities with NaN handling
        action_logits_safe = torch.clamp(action_logits, min=-10, max=10)
        if torch.isnan(action_logits_safe).any():
            print("Warning: NaN in batch action_logits, using uniform distribution")
            action_logits_safe = torch.zeros_like(action_logits_safe)

        action_probs = F.softmax(action_logits_safe, dim=-1) + 1e-8
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        if torch.isnan(action_probs).any():
            print("Warning: NaN in batch action_probs, using uniform distribution")
            action_probs = torch.ones_like(action_probs) / action_probs.shape[-1]

        action_dist = torch.distributions.Categorical(action_probs)
        new_action_log_probs = action_dist.log_prob(old_actions)

        # Compute policy ratios
        action_ratio = torch.exp(new_action_log_probs - old_action_log_probs)

        # PPO objective for Builder
        builder_policy_loss_1 = advantages * action_ratio
        builder_policy_loss_2 = advantages * torch.clamp(
            action_ratio,
            1 - self.config['clip_epsilon'],
            1 + self.config['clip_epsilon']
        )
        builder_policy_loss = -torch.min(builder_policy_loss_1, builder_policy_loss_2).mean()

        # Entropy bonus for exploration
        action_entropy = action_dist.entropy().mean()

        # Communication loss (simplified - encourage diverse communication)
        message_entropy = -torch.sum(F.softmax(message_logits, dim=-1) *
                                   F.log_softmax(message_logits, dim=-1), dim=-1).mean()

        # Separate value losses to avoid gradient conflicts
        architect_value_loss = F.mse_loss(architect_values.squeeze(-1), returns)
        builder_value_loss = F.mse_loss(builder_values.squeeze(-1), returns)

        # Total losses
        architect_loss = architect_value_loss * self.config['value_loss_coef'] + \
                        message_entropy * self.config['entropy_coef']

        builder_loss = builder_policy_loss + \
                      builder_value_loss * self.config['value_loss_coef'] - \
                      action_entropy * self.config['entropy_coef']

        # Update Architect first
        self.architect_optimizer.zero_grad()
        architect_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.architect.parameters(), self.config['max_grad_norm'])
        self.architect_optimizer.step()

        # Update Builder separately
        self.builder_optimizer.zero_grad()
        builder_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.builder.parameters(), self.config['max_grad_norm'])
        self.builder_optimizer.step()

        # Return combined loss for logging
        return (architect_loss + builder_loss).item()

    def add_experience(self, architect_obs, builder_obs, carrying_state,
                      message, action, reward, done, architect_value,
                      builder_value, action_log_prob, message_log_prob):
        """Add experience to the rollout buffer."""
        experience = Experience(
            architect_obs=architect_obs,
            builder_obs=builder_obs,
            carrying_state=carrying_state,
            message=message,
            action=action,
            reward=reward,
            done=done,
            architect_value=architect_value,
            builder_value=builder_value,
            action_log_prob=action_log_prob,
            message_log_prob=message_log_prob
        )
        self.buffer.add(experience)

    def get_training_stats(self):
        """Get current training statistics."""
        return {
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'success_rate': np.mean(self.success_rate) if self.success_rate else 0,
        }
