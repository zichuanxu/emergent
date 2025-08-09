import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class ArchitectBuilderEnv:
    """
    Architect-Builder cooperative environment.
    
    The Architect can see the blueprint but cannot act.
    The Builder can act but cannot see the blueprint.
    Communication is necessary for success.
    """
    
    def __init__(self, grid_size=8, block_colors=None, max_blocks_per_color=2, max_episode_steps=50):
        self.grid_size = grid_size
        self.block_colors = block_colors or ['red', 'green', 'blue', 'yellow']
        self.max_blocks_per_color = max_blocks_per_color
        self.max_episode_steps = max_episode_steps
        
        # Color mapping for visualization
        self.color_map = {
            'empty': 0,
            'red': 1,
            'green': 2, 
            'blue': 3,
            'yellow': 4,
            'builder': 5
        }
        
        # Action space: up, down, left, right, pick/place
        self.action_space_size = 5
        
        self.reset()
    
    def reset(self):
        """Reset environment and generate new episode."""
        # Initialize empty grid
        self.current_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Generate random blueprint (target configuration)
        self.blueprint = self._generate_blueprint()
        
        # Place initial blocks randomly
        self._place_initial_blocks()
        
        # Initialize builder position
        self.builder_pos = self._get_random_empty_position()
        
        # Reset episode state
        self.episode_steps = 0
        self.builder_carrying = None
        
        return self._get_observations()
    
    def _generate_blueprint(self):
        """Generate a random target configuration."""
        blueprint = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Place blocks according to max_blocks_per_color
        for color_name in self.block_colors:
            color_id = self.color_map[color_name]
            num_blocks = random.randint(1, self.max_blocks_per_color)
            
            for _ in range(num_blocks):
                pos = self._get_random_empty_position(blueprint)
                if pos is not None:
                    blueprint[pos[0], pos[1]] = color_id
        
        return blueprint
    
    def _place_initial_blocks(self):
        """Place initial blocks randomly (different from blueprint)."""
        # Count blocks needed from blueprint
        color_counts = {}
        for color_name in self.block_colors:
            color_id = self.color_map[color_name]
            color_counts[color_id] = np.sum(self.blueprint == color_id)
        
        # Place blocks randomly
        for color_id, count in color_counts.items():
            for _ in range(count):
                pos = self._get_random_empty_position()
                if pos is not None:
                    self.current_grid[pos[0], pos[1]] = color_id
    
    def _get_random_empty_position(self, grid=None):
        """Get a random empty position on the grid."""
        if grid is None:
            grid = self.current_grid
            
        empty_positions = np.where(grid == 0)
        if len(empty_positions[0]) == 0:
            return None
        
        idx = random.randint(0, len(empty_positions[0]) - 1)
        return (empty_positions[0][idx], empty_positions[1][idx])
    
    def step(self, action):
        """Execute builder action and return next state."""
        self.episode_steps += 1
        
        # Parse action
        if action == 0:  # up
            new_pos = (self.builder_pos[0] - 1, self.builder_pos[1])
        elif action == 1:  # down
            new_pos = (self.builder_pos[0] + 1, self.builder_pos[1])
        elif action == 2:  # left
            new_pos = (self.builder_pos[0], self.builder_pos[1] - 1)
        elif action == 3:  # right
            new_pos = (self.builder_pos[0], self.builder_pos[1] + 1)
        elif action == 4:  # pick/place
            new_pos = self.builder_pos
            self._pick_place_action()
        else:
            new_pos = self.builder_pos
        
        # Check bounds and update position
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            self.builder_pos = new_pos
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = (self._is_blueprint_complete() or 
                self.episode_steps >= self.max_episode_steps)
        
        return self._get_observations(), reward, done, {}
    
    def _pick_place_action(self):
        """Handle pick up or place down action."""
        builder_x, builder_y = self.builder_pos
        
        if self.builder_carrying is None:
            # Try to pick up block at current position
            if self.current_grid[builder_x, builder_y] != 0:
                self.builder_carrying = self.current_grid[builder_x, builder_y]
                self.current_grid[builder_x, builder_y] = 0
        else:
            # Try to place block at current position
            if self.current_grid[builder_x, builder_y] == 0:
                self.current_grid[builder_x, builder_y] = self.builder_carrying
                self.builder_carrying = None
    
    def _calculate_reward(self):
        """Calculate dense reward to guide learning."""
        reward = 0.0
        
        # Huge reward for completing the blueprint
        if self._is_blueprint_complete():
            return 1000.0
        
        # Dense rewards for progress
        correct_placements = 0
        total_blocks = 0
        
        # Count correct block placements
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.blueprint[i, j] != 0:  # There should be a block here
                    total_blocks += 1
                    if self.current_grid[i, j] == self.blueprint[i, j]:
                        correct_placements += 1
        
        # Reward for correct placements (0 to 50 points)
        if total_blocks > 0:
            placement_ratio = correct_placements / total_blocks
            reward += placement_ratio * 50.0
        
        # Reward for picking up blocks that need to be moved
        if self.builder_carrying is not None:
            # Check if this block belongs somewhere else
            current_pos = self.builder_pos
            target_color = self.blueprint[current_pos[0], current_pos[1]]
            
            if target_color != self.builder_carrying:
                # Carrying a block that doesn't belong here
                reward += 5.0
            else:
                # Carrying the right block for this position
                reward += 10.0
        
        # Small reward for being near blocks that need to be moved
        builder_x, builder_y = self.builder_pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = builder_x + dx, builder_y + dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                    current_block = self.current_grid[nx, ny]
                    target_block = self.blueprint[nx, ny]
                    
                    # If there's a misplaced block nearby
                    if current_block != 0 and current_block != target_block:
                        reward += 1.0
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.5
        
        # Penalty for invalid actions (hitting walls)
        if hasattr(self, '_invalid_move') and self._invalid_move:
            reward -= 2.0
            self._invalid_move = False
        
        return reward
    
    
    def _is_blueprint_complete(self):
        """Check if current configuration matches blueprint."""
        return np.array_equal(self.current_grid, self.blueprint)
    
    def _get_observations(self):
        """Get observations for both agents."""
        # Architect observation: blueprint (what should be built)
        architect_obs = self._grid_to_tensor(self.blueprint)
        
        # Builder observation: current grid + builder position
        builder_grid = self.current_grid.copy()
        builder_grid[self.builder_pos[0], self.builder_pos[1]] = self.color_map['builder']
        builder_obs = self._grid_to_tensor(builder_grid)
        
        return {
            'architect': architect_obs,
            'builder': builder_obs,
            'builder_carrying': self.builder_carrying or 0
        }
    
    def _grid_to_tensor(self, grid):
        """Convert grid to tensor representation."""
        # One-hot encode the grid
        tensor = torch.zeros(len(self.color_map), self.grid_size, self.grid_size)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                tensor[grid[i, j], i, j] = 1.0
        return tensor
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Render blueprint
            ax1.imshow(self.blueprint, cmap='viridis')
            ax1.set_title('Blueprint (Architect View)')
            ax1.grid(True)
            
            # Render current state
            current_display = self.current_grid.copy()
            current_display[self.builder_pos[0], self.builder_pos[1]] = self.color_map['builder']
            ax2.imshow(current_display, cmap='viridis')
            ax2.set_title('Current State (Builder View)')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        return self.current_grid

