#!/usr/bin/env python3
"""
Visualization script for the Architect-Builder environment.
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import config
from src.environments.environment import ArchitectBuilderEnv
from src.agents.agents import Architect, Builder
from src.training.mappo import MAPPO

def visualize_episode():
    """Visualize a single episode with random agents."""
    print("🎮 Visualizing Architect-Builder Environment")
    
    # Initialize environment
    env = ArchitectBuilderEnv(
        grid_size=config['grid_size'],
        block_colors=config['block_colors'],
        max_blocks_per_color=config['max_blocks_per_color'],
        max_episode_steps=config['max_episode_steps']
    )
    
    # Reset environment
    obs = env.reset()
    
    print(f"Environment initialized:")
    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Colors: {env.block_colors}")
    print(f"Max episode steps: {env.max_episode_steps}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    step = 0
    done = False
    
    while not done and step < 10:  # Show first 10 steps
        # Display current state
        ax1.clear()
        ax2.clear()
        
        # Blueprint (what Architect sees)
        ax1.imshow(env.blueprint, cmap='tab10', vmin=0, vmax=5)
        ax1.set_title(f'Blueprint (Architect View) - Step {step}')
        ax1.grid(True, alpha=0.3)
        
        # Current environment (what Builder sees)
        current_display = env.current_grid.copy()
        current_display[env.builder_pos[0], env.builder_pos[1]] = 5  # Mark builder position
        ax2.imshow(current_display, cmap='tab10', vmin=0, vmax=5)
        ax2.set_title(f'Current State (Builder View) - Step {step}')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar legend
        if step == 0:
            colors = ['Empty', 'Red', 'Green', 'Blue', 'Yellow', 'Builder']
            handles = [plt.Rectangle((0,0),1,1, color=plt.cm.tab10(i/10)) for i in range(6)]
            ax2.legend(handles, colors, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.pause(0.5)  # Pause to show the step
        
        # Take random action
        action = np.random.randint(0, 5)
        next_obs, reward, done, info = env.step(action)
        
        print(f"Step {step}: Action={action}, Reward={reward:.2f}, Done={done}")
        print(f"Builder position: {env.builder_pos}, Carrying: {env.builder_carrying}")
        
        step += 1
    
    plt.show()
    print("Episode completed!")

def test_agents():
    """Test trained agents on the environment."""
    print("🤖 Testing Trained Agents")
    
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
        input_channels=6,
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
    
    # Load trained model if it exists
    model_path = "results/architect_builder_v1/models/final_model.pt"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=config['device'], weights_only=False)
        architect.load_state_dict(checkpoint['architect_state_dict'])
        
        # Handle Builder state dict with potential dynamic layers
        builder_state_dict = checkpoint['builder_state_dict']
        # Remove dynamic layer keys that might not exist in current model
        filtered_state_dict = {k: v for k, v in builder_state_dict.items() 
                              if not k.startswith('vision_adjust')}
        builder.load_state_dict(filtered_state_dict, strict=False)
        print("Model loaded successfully!")
    else:
        print("No trained model found, using random agents")
    
    # Initialize trainer for action selection
    trainer = MAPPO(architect, builder, config)
    
    # Run test episode
    obs = env.reset()
    architect_obs = obs['architect'].to(config['device'])
    builder_obs = obs['builder'].to(config['device'])
    carrying_state = torch.tensor(obs['builder_carrying']).to(config['device'])
    
    hidden_state = None
    step = 0
    done = False
    total_reward = 0
    
    print("\nRunning episode with trained agents...")
    
    while not done:
        # Get actions from trained agents
        with torch.no_grad():
            actions = trainer.select_actions(
                architect_obs, builder_obs, carrying_state,
                hidden_state, training=False
            )
        
        # Take step
        next_obs, reward, done, info = env.step(actions['action'].cpu().item())
        
        print(f"Step {step}: Action={actions['action'].item()}, Reward={reward:.2f}")
        print(f"Message: {torch.argmax(actions['message'], dim=-1).cpu().numpy()}")
        
        # Update observations
        architect_obs = next_obs['architect'].to(config['device'])
        builder_obs = next_obs['builder'].to(config['device'])
        carrying_state = torch.tensor(next_obs['builder_carrying']).to(config['device'])
        hidden_state = actions['hidden_state']
        
        total_reward += reward
        step += 1
        
        if step >= 10:  # Limit output
            break
    
    print(f"\nEpisode completed! Total reward: {total_reward:.2f}")
    success = env._is_blueprint_complete()
    print(f"Blueprint completed: {success}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize Architect-Builder Environment')
    parser.add_argument('--mode', choices=['env', 'agents'], default='env',
                       help='Visualization mode: env (random) or agents (trained)')
    
    args = parser.parse_args()
    
    if args.mode == 'env':
        visualize_episode()
    else:
        test_agents()

