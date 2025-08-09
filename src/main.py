import torch
import torch.optim as optim
from src.environments.environment import GridWorld
from src.agents.agents import Architect, Builder
from config import config


def main():
    env = GridWorld(size=config['grid_size'], block_colors=config['block_colors'])
    architect = Architect(input_dim=3, hidden_dim=config['architect_hidden_dim'], vocab_size=16)
    builder = Builder(input_dim=3, hidden_dim=config['builder_hidden_dim'], action_dim=4)

    optimizer_architect = optim.Adam(architect.parameters(), lr=config['learning_rate'])
    optimizer_builder = optim.Adam(builder.parameters(), lr=config['learning_rate'])

    for episode in range(config['num_episodes']):
        state = env.reset()
        done = False
        while not done:
            # Generate message from Architect
            # Interpret message by Builder & take action
            # Compute reward and optimize model
            # Architect generates message
            message = architect(state.unsqueeze(0))
            
            # Decode message using Gumbel-Softmax
            symbols = F.gumbel_softmax(message, tau=config['gumbel_temperature'], hard=config['gumbel_hard'])[0]
            
            # Builder takes action based on message
            action, hidden_state = builder(state.unsqueeze(0), None)
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            
            # Compute and apply gradients
            optimizer_architect.zero_grad()
            optimizer_builder.zero_grad()
            loss = -reward
            loss.backward()
            optimizer_architect.step()
            optimizer_builder.step()

            state = next_state

if __name__ == '__main__':
    main()

