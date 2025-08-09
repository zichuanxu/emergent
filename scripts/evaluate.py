import torch
import numpy as np
from config import config
from src.environments.environment import GridWorld
from src.agents.agents import Architect, Builder


def evaluate():
    env = GridWorld(size=config['grid_size'], block_colors=config['block_colors'])
    architect = Architect(input_dim=3, hidden_dim=config['architect_hidden_dim'], vocab_size=16)
    builder = Builder(input_dim=3, hidden_dim=config['builder_hidden_dim'], action_dim=4)

    model_path = "results/experiment_001/models/final_model.pt"
    architect.load_state_dict(torch.load(model_path))
    builder.load_state_dict(torch.load(model_path))

    for _ in range(config['eval_blueprints']):
        state = env.reset()
        done = False
        while not done:
            # Evaluation loop logic to be implemented
            pass

if __name__ == "__main__":
    evaluate()

