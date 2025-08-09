#!/usr/bin/env python3
"""
Training Logger for Real-time Dashboard
Handles logging of training metrics and communication data
"""

import json
import os
import time
import threading
from collections import deque
import numpy as np
from datetime import datetime

def safe_tensor_to_numpy(tensor):
    """
    Safely convert tensor to numpy array, handling CUDA tensors

    Args:
        tensor: Input tensor (torch.Tensor, numpy array, or other)

    Returns:
        numpy array or original input if not a tensor
    """
    if tensor is None:
        return None

    # Handle torch tensors
    if hasattr(tensor, 'detach') and hasattr(tensor, 'cpu') and hasattr(tensor, 'numpy'):
        # This is a torch tensor, detach gradients and move to CPU
        return tensor.detach().cpu().numpy()
    elif hasattr(tensor, 'numpy'):
        # This might be a numpy-compatible object
        try:
            return tensor.numpy()
        except (RuntimeError, TypeError):
            # If numpy() fails, try other methods
            if hasattr(tensor, 'cpu'):
                return tensor.cpu().detach().numpy()
            else:
                return tensor
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        # Not a tensor, return as-is
        return tensor

class TrainingLogger:
    """
    Logger that writes training data to JSON file for real-time dashboard consumption
    """

    def __init__(self, log_file='training_log.json', max_history=1000):
        self.log_file = log_file
        self.max_history = max_history

        # Data storage
        self.data = {
            'episodes': [],
            'rewards': [],
            'success_rates': [],
            'episode_lengths': [],
            'losses': [],
            'messages': [],
            'actions': [],
            'timestamps': [],
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'last_update': None,
                'total_episodes': 0
            }
        }

        # Thread-safe writing
        self.lock = threading.Lock()
        self.write_pending = False

        # Initialize log file
        self._write_to_file()

        print(f"📝 Training logger initialized: {log_file}")

    def log_episode(self, episode, reward, success_rate, episode_length,
                   messages=None, actions=None, loss=None):
        """
        Log data from a completed episode

        Args:
            episode: Episode number
            reward: Total episode reward
            success_rate: Success rate (0.0 to 1.0)
            episode_length: Number of steps in episode
            messages: List of messages from the episode
            actions: List of actions from the episode
            loss: Training loss (optional)
        """
        with self.lock:
            # Add episode data
            self.data['episodes'].append(episode)
            self.data['rewards'].append(float(reward))
            self.data['success_rates'].append(float(success_rate))
            self.data['episode_lengths'].append(int(episode_length))
            self.data['timestamps'].append(time.time())

            if loss is not None:
                self.data['losses'].append(float(loss))

            # Add communication data
            if messages is not None:
                # Convert messages to serializable format
                messages = safe_tensor_to_numpy(messages)

                if isinstance(messages, np.ndarray):
                    messages = messages.tolist()

                self.data['messages'].extend(messages)

            if actions is not None:
                # Convert actions to serializable format
                actions = safe_tensor_to_numpy(actions)

                if isinstance(actions, np.ndarray):
                    actions = actions.tolist()
                elif not isinstance(actions, list):
                    actions = [actions]

                self.data['actions'].extend(actions)

            # Maintain maximum history
            self._trim_history()

            # Update metadata
            self.data['metadata']['last_update'] = datetime.now().isoformat()
            self.data['metadata']['total_episodes'] = episode + 1

            # Schedule write
            if not self.write_pending:
                self.write_pending = True
                # Write immediately for real-time updates
                self._write_to_file()
                self.write_pending = False

    def log_step(self, message, action):
        """
        Log data from a single step (for real-time communication analysis)

        Args:
            message: Message tensor/array from architect
            action: Action taken by builder
        """
        with self.lock:
            # Convert message to serializable format
            message = safe_tensor_to_numpy(message)

            if isinstance(message, np.ndarray):
                if message.ndim > 1:
                    message = message.flatten()
                message = message.tolist()

            # Convert action to serializable format
            if hasattr(action, 'item'):
                action = action.item()
            else:
                action = safe_tensor_to_numpy(action)
                if isinstance(action, np.ndarray):
                    action = action.item()

            # Add to data
            self.data['messages'].append(message)
            self.data['actions'].append(int(action))

            # Maintain maximum history for real-time data
            if len(self.data['messages']) > self.max_history:
                self.data['messages'] = self.data['messages'][-self.max_history:]
            if len(self.data['actions']) > self.max_history:
                self.data['actions'] = self.data['actions'][-self.max_history:]

    def log_loss(self, episode, loss):
        """
        Log training loss

        Args:
            episode: Current episode
            loss: Loss value
        """
        with self.lock:
            # Ensure losses list is same length as episodes
            while len(self.data['losses']) < len(self.data['episodes']):
                self.data['losses'].append(0.0)

            if len(self.data['losses']) > 0:
                self.data['losses'][-1] = float(loss)

    def _trim_history(self):
        """Trim data to maintain maximum history length"""
        for key in ['episodes', 'rewards', 'success_rates', 'episode_lengths', 'timestamps']:
            if len(self.data[key]) > self.max_history:
                self.data[key] = self.data[key][-self.max_history:]

        # Trim losses to match episodes
        if len(self.data['losses']) > len(self.data['episodes']):
            self.data['losses'] = self.data['losses'][-len(self.data['episodes']):]

        # Trim communication data
        max_comm_history = self.max_history * 10  # Keep more communication data
        for key in ['messages', 'actions']:
            if len(self.data[key]) > max_comm_history:
                self.data[key] = self.data[key][-max_comm_history:]

    def _write_to_file(self):
        """Write data to JSON file"""
        try:
            # Create backup of existing file
            if os.path.exists(self.log_file):
                backup_file = f"{self.log_file}.backup"
                try:
                    os.rename(self.log_file, backup_file)
                except:
                    pass

            # Write new data
            with open(self.log_file, 'w') as f:
                json.dump(self.data, f, indent=2)

            # Remove backup if write was successful
            backup_file = f"{self.log_file}.backup"
            if os.path.exists(backup_file):
                try:
                    os.remove(backup_file)
                except:
                    pass

        except Exception as e:
            print(f"Warning: Failed to write training log: {e}")
            # Restore backup if available
            backup_file = f"{self.log_file}.backup"
            if os.path.exists(backup_file):
                try:
                    os.rename(backup_file, self.log_file)
                except:
                    pass

    def get_stats(self):
        """Get current training statistics"""
        with self.lock:
            if not self.data['episodes']:
                return {}

            return {
                'total_episodes': len(self.data['episodes']),
                'latest_reward': self.data['rewards'][-1] if self.data['rewards'] else 0,
                'latest_success_rate': self.data['success_rates'][-1] if self.data['success_rates'] else 0,
                'avg_reward_last_100': np.mean(self.data['rewards'][-100:]) if len(self.data['rewards']) >= 100 else np.mean(self.data['rewards']),
                'avg_success_rate_last_100': np.mean(self.data['success_rates'][-100:]) if len(self.data['success_rates']) >= 100 else np.mean(self.data['success_rates']),
                'total_messages': len(self.data['messages']),
                'total_actions': len(self.data['actions'])
            }

    def close(self):
        """Close logger and perform final write"""
        with self.lock:
            self.data['metadata']['end_time'] = datetime.now().isoformat()
            self._write_to_file()
        print(f"📝 Training logger closed: {self.log_file}")

class DashboardManager:
    """
    Manager for launching and monitoring the dashboard
    """

    def __init__(self, log_file='training_log.json'):
        self.log_file = log_file
        self.dashboard_process = None

    def start_dashboard(self, update_interval=2000):
        """
        Start the dashboard in a separate process

        Args:
            update_interval: Update interval in milliseconds
        """
        import subprocess
        import sys

        try:
            # Launch dashboard as separate process
            cmd = [
                sys.executable,
                'visualization_dashboard.py',
                '--log_file', self.log_file,
                '--update_interval', str(update_interval)
            ]

            self.dashboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            print(f"🚀 Dashboard started (PID: {self.dashboard_process.pid})")
            print(f"📊 Monitoring: {self.log_file}")

        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
            print("💡 You can manually start it with: python visualization_dashboard.py")

    def stop_dashboard(self):
        """Stop the dashboard process"""
        if self.dashboard_process:
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=5)
                print("🛑 Dashboard stopped")
            except:
                try:
                    self.dashboard_process.kill()
                    print("🛑 Dashboard force stopped")
                except:
                    print("⚠️ Could not stop dashboard process")

            self.dashboard_process = None

    def is_dashboard_running(self):
        """Check if dashboard is still running"""
        if self.dashboard_process:
            return self.dashboard_process.poll() is None
        return False

# Example usage
if __name__ == "__main__":
    # Test the logger
    logger = TrainingLogger('test_training_log.json')

    # Simulate some training data
    for episode in range(10):
        reward = np.random.randn() * 10 + 50
        success_rate = min(1.0, episode / 10.0)
        episode_length = np.random.randint(20, 100)

        # Simulate messages and actions
        messages = np.random.randn(episode_length, 3).tolist()
        actions = np.random.randint(0, 5, episode_length).tolist()

        logger.log_episode(episode, reward, success_rate, episode_length, messages, actions)

        print(f"Logged episode {episode}")
        time.sleep(0.1)

    print("Test data generated. Check test_training_log.json")
    logger.close()