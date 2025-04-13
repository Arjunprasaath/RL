# Implementation of Policy Gradient

# Imports
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch.optim import SGD, Adam
from torch.distributions.categorical import Categorical

# Neural Network Policy
class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 10)
        self.layer2 = nn.Linear(10, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Helper Functions
def get_policy_dis(logits):
    return Categorical(logits= logits)

def get_action(policy_dis):
    return policy_dis.sample().item()

def compute_loss(policy, observations, actions, rewards):
    observations = torch.FloatTensor(observations)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)

    logits = policy(observations)

    policy_distribution = get_policy_dis(logits)
    return - (policy_distribution.log_prob(actions) * rewards).mean()

# Initializing environment and policy
env = gym.make("LunarLander-v3")
policy_model = Policy(env.observation_space.shape[0], env.action_space.n)
optim = Adam(policy_model.parameters(), lr= 0.1)
epochs = 500

# Training
for i in range(epochs):
    batch_obs = []
    batch_action = []
    batch_reward = []

    observation, info = env.reset()

    # flag to terminate
    done = False

    # run until done
    while not done:
        batch_obs.append(observation.copy())

        # Convert observation to tensor for the forward pass
        obs_tensor = torch.FloatTensor(observation)
        
        # Get action from policy
        logits = policy_model(obs_tensor)
        policy_dis = get_policy_dis(logits)
        action = get_action(policy_dis)
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        batch_action.append(action)
        batch_reward.append(reward)

        done = terminated or truncated
    
    print(f"Episode {i+1} - Steps: {len(batch_obs)}, Total Reward: {sum(batch_reward):.2f}")
    
    # Compute loss
    loss = compute_loss(policy_model, batch_obs, batch_action, batch_reward)
    print(f"Loss: {loss.item():.4f}")
    
    # Backpropagation and optimization
    optim.zero_grad()
    loss.backward()
    optim.step()

    env.close()

# Saving
torch.save(policy_model, f"models/pg_model_Adam_{epochs}.pt")
policy_model = torch.load(f"models/pg_model_Adam_{epochs}.pt", weights_only= False)

# Testing
env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()
done = False

while not done:
    obs_tensor = torch.FloatTensor(observation)
    logits = policy_model(obs_tensor)
    policy_dis = get_policy_dis(logits)
    action = get_action(policy_dis)

    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()