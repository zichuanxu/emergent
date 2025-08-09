import torch

config = {
    # Environment settings
    'grid_size': 4,  # Start smaller for easier learning
    'block_colors': ['red', 'green'],  # Fewer colors initially
    'max_blocks_per_color': 1,
    'max_episode_steps': 50,  # More steps to complete task
    
    # Agent settings
    'vocab_size': 8,  # Smaller vocabulary to start
    'message_length': 3,  # Shorter messages
    'architect_hidden_dim': 64,  # Smaller networks
    'builder_hidden_dim': 64,
    'architect_vision_channels': 4,  # RGB + object type
    'builder_vision_channels': 4,
    'action_dim': 5,  # up, down, left, right, pick/place
    
    # Learning settings - MAPPO (conservative for stability)
    'learning_rate': 1e-4,  # Conservative learning rate
    'batch_size': 16,  # Smaller batch size for stability
    'num_episodes': 1000,
    'ppo_epochs': 2,  # Fewer epochs to prevent instability
    'clip_epsilon': 0.1,  # Smaller clipping for stable updates
    'value_loss_coef': 0.25,  # Lower value loss coefficient
    'entropy_coef': 0.02,  # Moderate entropy for controlled exploration
    'gamma': 0.99,  # Standard discount factor
    'gae_lambda': 0.95,
    'max_grad_norm': 0.1,  # Very conservative gradient clipping
    
    # Communication settings
    'gumbel_temperature': 1.0,
    'gumbel_hard': True,
    
    # Evaluation settings
    'eval_episodes': 100,
    'eval_freq': 1000,  # Evaluate every N episodes
    'save_freq': 5000,  # Save model every N episodes
    
    # Visualization settings
    'render_mode': 'human',  # 'human' or 'rgb_array'
    'fps': 5,
    
    # Device settings
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    
    # Experiment settings
    'experiment_name': 'architect_builder_v1',
    'log_dir': 'results',
    'checkpoint_dir': 'models',
    'plot_dir': 'plots',
}
