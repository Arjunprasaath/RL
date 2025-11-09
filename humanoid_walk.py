import torch
import wandb
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch.optim import AdamW
from torch.distributions.normal import Normal

from rl_algo.ppo import PPO

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 400),
            nn.Tanh(),
            nn.Linear(400, 400),
            nn.Tanh(),
            nn.Linear(400, 200),
            nn.Tanh(),
        )

        self.mean = nn.Linear(200, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = self.shared_network(x)
        mu = self.mean(x)
        std = torch.exp(self.log_std)
        return mu, std
        
class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


LR = 1e-4
GAMMA = 0.98
LAMBDA = 0.95
EPSILON = 0.3
ENTROPY_COEFF = 0.01

epochs = 50000
episodes = 10
total_actor_loss = 0.0
total_critic_loss = 0.0


env = gym.make("Humanoid-v5")#, render_mode = "human")
actor_model = Policy(input_dim=env.observation_space.shape[0], output_dim=env.action_space.shape[0])
actor_optim = AdamW(actor_model.parameters(), lr = LR)

critic_model = Critic(input_dim=env.observation_space.shape[0])
critic_optim = AdamW(critic_model.parameters(), lr = LR)


print(f"Observation space: {env.observation_space}")
# print(f"Sample observation: {env.observation_space.sample()}") 
print(f"Action space: {env.action_space}")
print(f"Sample action: {env.action_space.sample()}")

config = {"learning rate": LR, "total epochs": epochs, "episodes": episodes}
run = wandb.init(project="Humanoid V5 RL training", name= "PPO run", config=config)

for epoch in range(epochs):
    epoch_total_reward = 0.0
    
    batch_obs = []
    batch_action = []
    batch_reward = []
    batch_dones = []
    batch_log_probs = []
    batch_state_values = []

    for i in range(episodes):
        observation, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            batch_obs.append(observation.copy())
            obs_tensor = torch.FloatTensor(observation)

            with torch.no_grad():
                mean, std = actor_model(obs_tensor)
                actor_distribution = Normal(mean, std)
                action = actor_distribution.sample()
                action_tensor = (torch.tanh(action) * 0.4).numpy()
                # print(action)
                # action = torch.clamp(action, -0.4, 4.0).numpy()
                state_value = critic_model(obs_tensor).squeeze()

            observation, reward, terminated, truncated, info = env.step(action_tensor)

            done = terminated or truncated

            batch_action.append(action)
            batch_reward.append(reward)
            batch_dones.append(float(done))
            batch_log_probs.append(actor_distribution.log_prob(action).sum(axis = -1))
            batch_state_values.append(state_value)

            episode_reward += reward
        
        epoch_total_reward += episode_reward

    batch_state_values.append(torch.tensor(0.0))
    batch_obs_tensor = torch.tensor(np.array(batch_obs), dtype=torch.float32)
    batch_action_tensor = torch.stack(batch_action)
    batch_reward_tensor = torch.tensor(np.array(batch_reward), dtype=torch.float32)
    batch_dones_tensor = torch.tensor(np.array(batch_dones), dtype=torch.float32)
    batch_log_probs_tensor = torch.stack(batch_log_probs)
    batch_state_values_tensor = torch.stack(batch_state_values)

    ppo = PPO(batch_obs_tensor, batch_reward_tensor, batch_action_tensor, batch_state_values_tensor, batch_dones_tensor, batch_log_probs_tensor, GAMMA, LAMBDA, EPSILON, ENTROPY_COEFF)

    actor_loss, critic_loss = ppo(actor_model, critic_model)

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    total_actor_loss += actor_loss.item()
    total_critic_loss += critic_loss.item()
    print(f"Epoch {epoch}: Actor Loss: {actor_loss.item():.4f}, critic Loss: {critic_loss.item():.4f}")
    run.log({"Model loss": actor_loss, "Value loss": critic_loss, "Reward": epoch_total_reward})
env.close()

torch.save(actor_model, f"models/ppo_actor_adamw_humanoid_{epochs}.pt")
torch.save(critic_model, f"models/ppo_critic_adamw_humanoid_rtg_{epochs}.pt")

policy_model = torch.load(f"models/ppo_actor_adamw_humanoid_{epochs}.pt", weights_only=False)
# critic_model = torch.load(f"models/ppo_critic_adamw_humanoid_rtg_{epochs}.pt", weights_only=False)

env = gym.make("Humanoid-v5", render_mode = "human")
observation, info = env.reset()
done = False
episode_reward = 0

while not done:
    obs_tensor = torch.FloatTensor(observation)

    mu, std = policy_model(obs_tensor)
    policy_distribution = Normal(mu, std)
    action = policy_distribution.sample()
    action_tensor = (torch.tanh(action) * 0.4).numpy()

    observation, reward, terminated, truncated, info = env.step(action_tensor)
    episode_reward += reward
    done = terminated or truncated
    
env.close()
print(f"total reward: {episode_reward}")
