#!/usr/bin/env python3
"""
Training script for the Architect-Builder emergent communication project.
"""

import sys
import os
import torch
import numpy as np
import random
from pathlib import Path
import argparse
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import config
from src.environments.environment import ArchitectBuilderEnv
from src.agents.agents import Architect, Builder
from src.training.mappo import MAPPO
from src.utils.training_logger import TrainingLogger, DashboardManager

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_directories():
    """Create necessary directories for saving results."""
    experiment_dir = Path(config['log_dir']) / config['experiment_name']

    # Create directories
    (experiment_dir / 'models').mkdir(parents=True, exist_ok=True)
    (experiment_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (experiment_dir / 'plots').mkdir(parents=True, exist_ok=True)

    return experiment_dir

def train():
    """Main training function."""
    print("🚀 Starting Architect-Builder Training")
    print(f"Device: {config['device']}")
    print(f"Experiment: {config['experiment_name']}")

    # Set random seed
    set_seed(config['seed'])

    # Create directories
    experiment_dir = create_directories()

    # Initialize logging and dashboard
    log_file = experiment_dir / 'logs' / 'training_log.json'
    logger = TrainingLogger(str(log_file))
    dashboard_manager = DashboardManager(str(log_file))

    # Initialize environment
    env = ArchitectBuilderEnv(
        grid_size=config['grid_size'],
        block_colors=config['block_colors'],
        max_blocks_per_color=config['max_blocks_per_color'],
        max_episode_steps=config['max_episode_steps']
    )

    # Initialize agents
    architect = Architect(
        grid_size=config['grid_size'],
        input_channels=6,  # Number of color channels
        hidden_dim=config['architect_hidden_dim'],
        vocab_size=config['vocab_size'],
        message_length=config['message_length']
    ).to(config['device'])

    builder = Builder(
        grid_size=config['grid_size'],
        input_channels=6,
        hidden_dim=config['builder_hidden_dim'],
        vocab_size=config['vocab_size'],
        message_length=config['message_length'],
        action_dim=config['action_dim']
    ).to(config['device'])

    # Initialize trainer
    trainer = MAPPO(architect, builder, config)

    print(f"Architect parameters: {sum(p.numel() for p in architect.parameters()):,}")
    print(f"Builder parameters: {sum(p.numel() for p in builder.parameters()):,}")

    # Start dashboard (optional)
    try:
        dashboard_manager.start_dashboard(update_interval=3000)
    except Exception as e:
        print(f"⚠️ Could not start dashboard: {e}")
        print("💡 You can manually start it with: python visualization_dashboard.py")

    # Training loop
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    # Storage for logging communication data
    episode_messages = []
    episode_actions = []

    progress_bar = tqdm(range(config['num_episodes']), desc="Training")

    for episode in progress_bar:
        # Reset environment
        obs = env.reset()
        architect_obs = obs['architect'].to(config['device'])
        builder_obs = obs['builder'].to(config['device'])
        carrying_state = torch.tensor(obs['builder_carrying']).to(config['device'])

        hidden_state = None
        episode_reward = 0
        episode_length = 0
        done = False

        # Reset episode communication data
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

            # Log step data for real-time analysis
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
            architect_obs = next_obs['architect'].to(config['device'])
            builder_obs = next_obs['builder'].to(config['device'])
            carrying_state = torch.tensor(next_obs['builder_carrying']).to(config['device'])
            hidden_state = actions['hidden_state']

            episode_reward += reward
            episode_length += 1

        # Track episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Check if blueprint was completed successfully
        episode_success = env._is_blueprint_complete() or reward > 500
        if episode_success:
            success_count += 1

        trainer.episode_rewards.append(episode_reward)
        trainer.episode_lengths.append(episode_length)
        trainer.success_rate.append(1 if episode_success else 0)

        # Log episode data
        current_success_rate = success_count / (episode + 1)

        # Convert messages and actions to proper format for logging
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
        if len(trainer.buffer.experiences) >= config['batch_size']:
            loss = trainer.update()
            if loss is not None:
                logger.log_loss(episode, loss)

        # Update progress bar
        if episode % 100 == 0:
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            recent_success = success_count / min(episode + 1, 100) * 100

            progress_bar.set_postfix({
                'Reward': f"{np.mean(recent_rewards):.2f}",
                'Success': f"{recent_success:.1f}%",
                'Length': f"{np.mean(episode_lengths[-100:]):.1f}"
            })

        # Save model periodically
        if episode % config['save_freq'] == 0 and episode > 0:
            checkpoint_path = experiment_dir / 'models' / f'checkpoint_{episode}.pt'
            torch.save({
                'episode': episode,
                'architect_state_dict': architect.state_dict(),
                'builder_state_dict': builder.state_dict(),
                'architect_optimizer': trainer.architect_optimizer.state_dict(),
                'builder_optimizer': trainer.builder_optimizer.state_dict(),
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint at episode {episode}")

        # Evaluation
        if episode % config['eval_freq'] == 0 and episode > 0:
            print(f"\n🔍 Evaluation at episode {episode}")
            eval_success_rate = evaluate_agents(env, architect, builder, trainer, num_episodes=10)
            print(f"Evaluation success rate: {eval_success_rate:.1f}%")

    # Save final model
    final_model_path = experiment_dir / 'models' / 'final_model.pt'
    torch.save({
        'architect_state_dict': architect.state_dict(),
        'builder_state_dict': builder.state_dict(),
        'config': config,
        'training_stats': {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_success_rate': success_count / config['num_episodes'] * 100
        }
    }, final_model_path)

    print(f"\n🎉 Training completed!")
    print(f"Final success rate: {success_count / config['num_episodes'] * 100:.1f}%")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Model saved to: {final_model_path}")

    # Cleanup logging and dashboard
    logger.close()
    dashboard_manager.stop_dashboard()

    print("📊 Training log and dashboard closed")

def evaluate_agents(env, architect, builder, trainer, num_episodes=10):
    """Evaluate trained agents."""
    architect.eval()
    builder.eval()

    success_count = 0

    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()
            architect_obs = obs['architect'].to(config['device'])
            builder_obs = obs['builder'].to(config['device'])
            carrying_state = torch.tensor(obs['builder_carrying']).to(config['device'])

            hidden_state = None
            done = False

            while not done:
                actions = trainer.select_actions(
                    architect_obs, builder_obs, carrying_state,
                    hidden_state, training=False
                )

                next_obs, reward, done, info = env.step(actions['action'].cpu().item())

                architect_obs = next_obs['architect'].to(config['device'])
                builder_obs = next_obs['builder'].to(config['device'])
                carrying_state = torch.tensor(next_obs['builder_carrying']).to(config['device'])
                hidden_state = actions['hidden_state']

            if reward > 50:  # Success threshold
                success_count += 1

    architect.train()
    builder.train()

    return success_count / num_episodes * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Architect-Builder agents')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes')
    parser.add_argument('--device', type=str, default=None,
                       help='Training device (cuda/cpu)')

    args = parser.parse_args()

    # Override config if arguments provided
    if args.episodes:
        config['num_episodes'] = args.episodes
    if args.device:
        config['device'] = args.device

    train()
