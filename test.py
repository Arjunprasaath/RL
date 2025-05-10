import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions.categorical import Categorical


# Helper Functions
def get_policy_dis(logits):
    return Categorical(logits=logits)

def get_action(policy_dis):
    return policy_dis.sample()

epochs = 5001

# Actor Network
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 10)
        self.layer2 = nn.Linear(10, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return x

# Loading
actor_model = torch.load(f"models/ppo_actor_model_Adam_{epochs}.pt", weights_only= False)
# critic_model = torch.load(f"ac_critic_model_Adam_{epochs}.pt", weights_only= False)

# Testing
env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()
done = False

while not done:
    obs_tensor = torch.FloatTensor(observation)
    logits = actor_model(obs_tensor)
    policy_dis = get_policy_dis(logits)
    action = get_action(policy_dis)

    observation, reward, terminated, truncated, info = env.step(action.item())
    done = terminated or truncated
print(f"Reward: {reward}")