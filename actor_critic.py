# Implementation of Actor Critic
# Imports
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

# Neural Network Actor
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 10)
        self.layer2 = nn.Linear(10, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
# Neural Network Critic
class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 10)
        self.layer2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Helper Functions
def get_policy_dis(logits):
    return Categorical(logits= logits)

def get_action(policy_dis):
    return policy_dis.sample()

# Reward to Go implementation
def reward_to_go(reward):
    n = len(reward)
    rtg_array = np.zeros_like(reward)
    for i in reversed(range(n)):
        rtg_array[i] = reward[i] + (rtg_array[i + 1] if i + 1 < n else 0)
    return rtg_array


# Initializing environment and policy
discount_factor = 0.99

env = gym.make("LunarLander-v3")
actor = Actor(env.observation_space.shape[0], env.action_space.n)
critic = Critic(env.observation_space.shape[0], 1)
optim_actor = Adam(actor.parameters(), lr= 0.01)
optim_critic = Adam(critic.parameters(), lr = 0.01)

epochs = 1001

# Training
for i in range(epochs):
    batch_obs = []
    batch_reward = []
    batch_action = []
    batch_log_probs = []
    batch_state_value = []
    batch_discount_factor = []

    observation, info = env.reset()
    done = False

    while not done:

        batch_obs.append(observation.copy())

        obs_tensor = torch.FloatTensor(observation)

        # Get action from actor
        logits = actor(obs_tensor)
        actor_dis = get_policy_dis(logits)
        action = get_action(actor_dis)
        state_value = critic(obs_tensor).squeeze()

        observation, reward, terminated, truncated, info = env.step(action.item())

        batch_reward.append(reward)
        batch_action.append(action)
        batch_log_probs.append(actor_dis.log_prob(action))
        batch_state_value.append(state_value)

        done = terminated or truncated
        if done:
            batch_state_value.append(torch.tensor(0).float())
    
    # print(f"Batch obs:\n {len(batch_obs)} \nBatch action:\n {len(batch_action)} \nBatch reward:\n {len(batch_reward)} \nBatch state value:\n {len(batch_state_value)} \nBatch log probs:\n {len(batch_log_probs)}")
    batch_rtg_tensor = torch.tensor(reward_to_go(batch_reward), dtype=torch.float32)
    batch_obs_tensor = torch.tensor(np.array(batch_obs), dtype=torch.float32)
    batch_action_tensor = torch.stack(batch_action)
    batch_log_probs_tensor = torch.stack(batch_log_probs)
    batch_state_value_tensor = torch.stack(batch_state_value)

    # Critic loss
    critic_loss = F.mse_loss(batch_state_value_tensor[:-1], batch_rtg_tensor)

    # Advantage function
    # A(s) = R(s) + V(s`) - V(s)
    # detach state values so that actor gradient doesn't influence the critic update
    batch_advantage = batch_rtg_tensor + discount_factor * batch_state_value_tensor[1:].detach() - batch_state_value_tensor[:-1].detach()

    # Actor loss
    actor_loss = - (batch_log_probs_tensor *  batch_advantage).mean()

    # Update actor and critic
    optim_critic.zero_grad()
    critic_loss.backward()
    optim_critic.step()

    optim_actor.zero_grad()
    actor_loss.backward()
    optim_actor.step()

    # Logging
    if i % 50 == 0:
        print(f"Episode: {i + 1} Total steps: {len(batch_obs)} Total reward: {sum(batch_reward):.3f} Average reward: {sum(batch_reward) / len(batch_reward):.3f} Actor loss: {actor_loss:.3f} Critic loss: {critic_loss:.3f}")

    env.close()

# Saving
torch.save(actor, f"actor_model_Adam_{epochs}.pt")
torch.save(critic, f"critic_model_Adam_{epochs}.pt")
actor_model = torch.load(f"actor_model_Adam_{epochs}.pt", weights_only= False)
# critic_model = torch.load(f"critic_model_Adam_{epochs}.pt", weights_only= False)

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