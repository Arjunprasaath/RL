# Implementation of Proximal Policy Optimization
# Imports
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.distributions.categorical import Categorical

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
    
# Critic Network
class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 10)
        self.layer2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return x
    
# Helper Functions
def get_policy_dis(logits):
    return Categorical(logits=logits)

def get_action(policy_dis):
    return policy_dis.sample()


def generalized_advantage_estimate(rewards, values, next_values, dones, gamma, lambda_gae):
    advantages = torch.zeros_like(rewards)
    last_advantage = 0.0

    for i in reversed(range(len(rewards))):
        # Mask is 0 if done, else 1
        mask = 1.0 - dones[i]
        # Calculate TD error: delta_i = r_i + gemma * Vs(i + 1) * mask - Vs(i)
        delta = rewards[i] + gamma * mask * next_values[i] - values[i]
        # Calculate Advantage: A_i = delta_i + gamma * lambda * A(i + 1) * mask
        advantages[i] = delta + gamma * lambda_gae * last_advantage * mask
        last_advantage = advantages[i]
    
    mean_adv = torch.mean(advantages)
    std_adv = torch.std(advantages) + 1e-8
    advantages = (advantages - mean_adv) / std_adv
    return advantages


# Initializing Environment and Hyperparameters
GAMMA = 0.99
LAMBDA = 0.95
EPSILON = 0.2
STATE_VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.01


total_entropy = 0.0
total_value_loss = 0.0
total_policy_loss = 0.0

env = gym.make("LunarLander-v3", render_mode= "rgb_array")
# print(env.observation_space.shape, env.action_space.n) -> (8, ) 4
actor = Actor(input_dim=env.observation_space.shape[0], output_dim=env.action_space.n)
critic = Critic(input_dim=env.observation_space.shape[0])
actor_optim = Adam(actor.parameters(), lr = 0.01)
critic_optim = Adam(critic.parameters(), lr = 0.01)

epochs = 200

for i in range(epochs):
    batch_obs = []
    batch_action = []
    batch_reward = []
    batch_dones = []
    batch_log_probs = []
    batch_state_values = []

    observation, info = env.reset()
    done = False
    
    while not done:
        
        batch_obs.append(observation.copy())
        obs_tensor = torch.FloatTensor(observation)

        # Get action from actor
        logits = actor(obs_tensor)
        actor_dis = get_policy_dis(logits)
        action = get_action(actor_dis)
        # Get state value from critic
        state_value = critic(obs_tensor).squeeze()

        observation, reward, terminated, truncated, info = env.step(action.item())

        batch_reward.append(reward)
        batch_action.append(action)
        batch_state_values.append(state_value)
        batch_log_probs.append(actor_dis.log_prob(action))
        batch_dones.append(float(done))

        done = truncated or terminated
        if done:
            batch_state_values.append(torch.tensor(0).float())
    
    # print(f"Batch obs:\n {len(batch_obs)} \nBatch action:\n {len(batch_action)} \nBatch reward:\n {len(batch_reward)} \nBatch state value:\n {len(batch_state_values)} \nBatch log probs:\n {len(batch_log_probs)}")
    batch_obs_tensor = torch.tensor(np.array(batch_obs), dtype=torch.float32)
    batch_reward_tensor = torch.tensor(np.array(batch_reward), dtype=torch.float32)
    batch_dones_tensor = torch.tensor(np.array(batch_dones), dtype=torch.float32)
    batch_action_tensor = torch.stack(batch_action)
    batch_log_probs_tensor = torch.stack(batch_log_probs)
    batch_state_value_tensor = torch.stack(batch_state_values)

    # Estimate Advantage & Reward-to-go
    advantage = generalized_advantage_estimate(batch_reward_tensor, batch_state_value_tensor[:-1], batch_state_value_tensor[1:], batch_dones_tensor, GAMMA, LAMBDA)
    # print(advantage.shape, batch_state_value_tensor.shape)
    reward_to_go = advantage + batch_state_value_tensor[1:] # check what to do with the state value tensor
    # print(advantage.shape, reward_to_go.shape)

    # Update the models using PPO
    advantage = advantage.detach()
    reward_to_go = reward_to_go.detach()
    old_log_probs = batch_log_probs_tensor.detach()

    logits = actor(batch_obs_tensor)
    policy_distribution = get_policy_dis(logits)
    new_log_prob = policy_distribution.log_prob(batch_action_tensor)
    entropy = policy_distribution.entropy().mean()
    # print(f"Logits: {logits.shape}, New Log Probability: {new_log_prob.shape}, Entropy: {entropy.shape}")

    # Calculate ration r_t(theta)
    ratio = torch.exp(new_log_prob - old_log_probs)
    # print(ratio)

    # Calculate surrogate objective
    policy_loss = - torch.min(ratio * advantage, torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * advantage).mean() - ENTROPY_COEFF * entropy

    # Update actor
    actor_optim.zero_grad()
    policy_loss.backward()
    actor_optim.step()

    value_pred = critic(batch_obs_tensor).squeeze()
    value_loss = F.mse_loss(value_pred, reward_to_go)

    # Update critic
    critic_optim.zero_grad()
    value_loss.backward()
    critic_optim.step()

    total_policy_loss += policy_loss.item()
    total_value_loss += value_loss.item()
    total_entropy += entropy.item()

    print(f"Epoch {i}: Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Entropy: {entropy.item():.4f}")

env.close()
