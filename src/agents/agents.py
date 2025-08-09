import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Architect(nn.Module):
    """
    Architect agent that generates symbolic messages based on the blueprint.
    
    Architecture:
    - CNN encoder for blueprint vision
    - Transformer for generating communication sequence
    - Gumbel-Softmax for differentiable discrete symbol selection
    """
    
    def __init__(self, grid_size=8, input_channels=6, hidden_dim=128, 
                 vocab_size=16, message_length=5, num_heads=4):
        super(Architect, self).__init__()
        
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.message_length = message_length
        
        # Vision encoder - CNN to process blueprint
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Reduce to manageable size
        )
        
        # Flatten vision features
        self.vision_projection = nn.Linear(hidden_dim * 4 * 4, hidden_dim)
        
        # Simpler message generation (replace Transformer with MLP)
        self.message_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_length * vocab_size)
        )
        
        # Value network for MAPPO
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def forward(self, blueprint_obs, temperature=1.0, hard=False):
        """
        Forward pass to generate communication message.
        
        Args:
            blueprint_obs: Tensor of shape (batch, channels, height, width)
            temperature: Temperature for Gumbel-Softmax
            hard: Whether to use hard Gumbel-Softmax
        
        Returns:
            message: Generated symbolic message
            value: State value estimate
        """
        batch_size = blueprint_obs.shape[0]
        
        # Encode blueprint through CNN
        vision_features = self.vision_encoder(blueprint_obs)
        vision_features = vision_features.view(batch_size, -1)
        vision_embedding = self.vision_projection(vision_features)
        
        # Generate message through MLP
        message_flat = self.message_generator(vision_embedding)
        message_logits = message_flat.view(batch_size, self.message_length, self.vocab_size)
        
        # Apply Gumbel-Softmax for differentiable discrete sampling
        message = F.gumbel_softmax(message_logits, tau=temperature, hard=hard, dim=-1)
        
        # Compute value estimate
        value = self.value_network(vision_embedding)
        
        return message, value, message_logits
    
    def _init_weights(self):
        """Initialize network weights conservatively."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Very small gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 0.1  # Scale down conv weights
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

class Builder(nn.Module):
    """
    Builder agent that interprets messages and takes actions in the environment.
    
    Architecture:
    - CNN encoder for environment vision
    - RNN for processing communication sequence
    - Policy and value networks for MAPPO
    """
    
    def __init__(self, grid_size=8, input_channels=6, hidden_dim=128, 
                 vocab_size=16, message_length=5, action_dim=5):
        super(Builder, self).__init__()
        
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.message_length = message_length
        self.action_dim = action_dim
        
        # Vision encoder - CNN to process current environment state
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.vision_projection = nn.Linear(hidden_dim * 4 * 4, hidden_dim)
        
        # Message processing - embedding + RNN
        self.message_embedding = nn.Linear(vocab_size, hidden_dim)
        self.message_processor = nn.GRU(
            hidden_dim, hidden_dim, 
            batch_first=True, num_layers=2
        )
        
        # Fusion layer to combine vision and communication
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Policy network for action selection
        self.policy_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network for MAPPO
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Additional input for carrying state
        self.carrying_embedding = nn.Embedding(6, hidden_dim // 4)  # 6 possible carrying states
        
        # Initialize weights
        self._init_weights()
    
    def forward(self, env_obs, message, carrying_state=None, hidden_state=None):
        """
        Forward pass to select action based on environment and message.
        
        Args:
            env_obs: Environment observation tensor
            message: Communication message from architect
            carrying_state: What the builder is carrying (if anything)
            hidden_state: Previous RNN hidden state
        
        Returns:
            action_logits: Action probability distribution
            value: State value estimate
            new_hidden_state: Updated RNN hidden state
        """
        batch_size = env_obs.shape[0]
        
        # Encode environment through CNN
        vision_features = self.vision_encoder(env_obs)
        vision_features = vision_features.view(batch_size, -1)
        vision_embedding = self.vision_projection(vision_features)
        
        # Process communication message
        # If message is one-hot, convert to soft embeddings
        if message.dim() == 3:  # (batch, seq_len, vocab_size)
            # Reshape message for matrix multiplication: (batch, seq_len, vocab_size) @ (vocab_size, hidden_dim)
            message_embedded = torch.matmul(message, self.message_embedding.weight.T)
        else:
            message_embedded = self.message_embedding(message)
        
        # Process message sequence through RNN
        message_output, new_hidden_state = self.message_processor(
            message_embedded, hidden_state
        )
        
        # Use final RNN output as message representation
        message_representation = message_output[:, -1, :]
        
        # Add carrying state information if provided
        if carrying_state is not None:
            carrying_embed = self.carrying_embedding(carrying_state)
            vision_embedding = torch.cat([vision_embedding, carrying_embed], dim=-1)
            # Adjust fusion layer input size if needed
            if hasattr(self, 'vision_adjust'):
                vision_embedding = self.vision_adjust(vision_embedding)
            else:
                # Create adjustment layer on first use
                input_size = vision_embedding.shape[-1]
                self.vision_adjust = nn.Linear(input_size, self.hidden_dim).to(vision_embedding.device)
                vision_embedding = self.vision_adjust(vision_embedding)
        
        # Fuse vision and communication information
        fused_features = torch.cat([vision_embedding, message_representation], dim=-1)
        fused_embedding = self.fusion_layer(fused_features)
        
        # Generate action logits and value estimate
        action_logits = self.policy_network(fused_embedding)
        value = self.value_network(fused_embedding)
        
        return action_logits, value, new_hidden_state
    
    def _init_weights(self):
        """Initialize network weights conservatively."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Very small gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 0.1  # Scale down conv weights
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

class GumbelSoftmax(nn.Module):
    """
    Gumbel-Softmax module for differentiable discrete sampling.
    """
    
    def __init__(self, temperature=1.0, hard=False):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.hard = hard
    
    def forward(self, logits):
        return F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=-1)

