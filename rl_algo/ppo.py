import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class PPO():
    def __init__(self, obs, rewards, actions, state_values, dones, old_log_probs, gamma, lambda_gae, epsilon, entropy_coefficient):
        self.obs = obs
        self.rewards = rewards
        self.actions = actions
        self.state_values = state_values.detach()
        self.dones = dones
        self.old_log_probs = old_log_probs.detach()
        
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.epsilon = epsilon
        self.entropy_coefficient = entropy_coefficient

    def _generalized_advantage_estimate(self):
        self.advantages = torch.zeros_like(self.rewards)
        last_advantage = 0.0

        for i in reversed(range(len(self.rewards))):
            # mask is zero when done, else 1
            mask = 1.0 - self.dones[i]

            # calculate TD error: delta[i] = reward[i] + gamma * Vs(i + 1) * mask - Vs(i)
            delta = self.rewards[i] + self.gamma * self.state_values[i + 1] * mask - self.state_values[i]

            # calculate advantage: A_i = delta_i + gamma * lambda * A(i + 1) *  mask
            self.advantages[i] = delta + self.gamma * self.lambda_gae * last_advantage * mask
            last_advantage = self.advantages[i]
        
        # mean_adv = torch.mean(self.advantages)
        # std_adv = torch.std(self.advantages) + 1e-8

        # self.advantages = (self.advantages - mean_adv) / std_adv
        return self.advantages
    
    def __call__(self, actor, critic):
        advantage = self._generalized_advantage_estimate()
        rtg = advantage + self.state_values[:-1]
        # print(advantage, rtg)

        # advantage = advantage.detach()
        # rtg = rtg.detach()
        # old_log_probs = self.old_log_probs.detach()

        mu, std = actor(self.obs)
        actor_distribution = Normal(mu, std)
        new_log_probs = actor_distribution.log_prob(self.actions).sum(axis = -1)
        entropy = actor_distribution.entropy().mean()
        ratio = torch.exp(new_log_probs - self.old_log_probs)

        actor_loss = - torch.min(ratio * advantage, torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage).mean() - self.entropy_coefficient * entropy
        
        value_pred = critic(self.obs).squeeze()
        critic_loss = F.mse_loss(value_pred, rtg) 
        return actor_loss, critic_loss
    

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import Categorical, kl_divergence

# # Assume we have a simple policy network
# class PolicyNetwork(nn.Module):
#     def __init__(self, obs_size, act_size):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_size, 64),
#             nn.Tanh(),
#             nn.Linear(64, act_size),
#         )

#     def get_distribution(self, obs):
#         # Returns a distribution object (e.g., Categorical)
#         logits = self.net(obs)
#         return Categorical(logits=logits)

#     def forward(self, obs, action=None):
#         dist = self.get_distribution(obs)
#         if action is None:
#             action = dist.sample()
#         log_prob = dist.log_prob(action)
#         return action, log_prob, dist

# class AdaptivePPOManager:
#     def __init__(self, policy_net, lr=3e-4):
#         self.policy = policy_net
#         self.policy_old = PolicyNetwork(policy_net.obs_size, policy_net.act_size)
#         self.policy_old.load_state_dict(self.policy.state_dict())

#         # 1. HERE IS YOUR ADAM(W) ACCELERATION!
#         # The m_t and v_t logic is built *inside* this optimizer.
#         # You could use optim.AdamW here as well.
#         self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

#         # 2. PPO and Adaptive Parameters
#         self.epsilon = 0.2  # Initial clip range
#         self.kl_target = 0.015
#         self.epsilon_adapt_rate = 1.2
#         self.K_epochs = 10

#     def update(self, batch_states, batch_actions, batch_advantages):
#         # Get old log probs and distributions *before* updates
#         with torch.no_grad():
#             _, old_log_probs, old_dists = self.policy_old(batch_states, batch_actions)

#         # --- PPO Update Loop ---
#         for _ in range(self.K_epochs):
#             # Get new log probs and distributions
#             _, new_log_probs, new_dists = self.policy(batch_states, batch_actions)

#             # Calculate ratio
#             r_theta = torch.exp(new_log_probs - old_log_probs)

#             # Calculate PPO clipped loss
#             surr1 = r_theta * batch_advantages
#             surr2 = torch.clamp(r_theta, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
            
#             # We minimize the negative of the objective
#             loss = -torch.min(surr1, surr2).mean()

#             # 3. ADAM(W) DOES ITS WORK HERE
#             # This step uses m_t and v_t to "accelerate"
#             # down the gradient of the loss landscape.
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()

#         # --- Adaptation Step (After all K epochs) ---
        
#         # 4. Measure how much the policy *actually* changed
#         with torch.no_grad():
#             # Get the final distributions after all updates
#             final_dists = self.policy.get_distribution(batch_states)
            
#             # Calculate the KL divergence
#             # D_KL(pi_old || pi_new)
#             kl_actual = kl_divergence(old_dists, final_dists).mean().item()

#         # 5. Adapt epsilon based on KL
#         if kl_actual > 1.5 * self.kl_target:
#             # Too unstable, decelerate (shrink clip range)
#             self.epsilon = max(0.01, self.epsilon / self.epsilon_adapt_rate)
#             # print(f"KL too high ({kl_actual:.4f}). Decreasing epsilon to {self.epsilon:.3f}")
#         elif kl_actual < 0.5 * self.kl_target:
#             # Too stable, accelerate (widen clip range)
#             self.epsilon = min(0.4, self.epsilon * self.epsilon_adapt_rate)
#             # print(f"KL too low ({kl_actual:.4f}). Increasing epsilon to {self.epsilon:.3f}")

#         # 6. Sync old policy for next iteration
#         self.policy_old.load_state_dict(self.policy.state_dict())

#         return loss.item(), kl_actual, self.epsilon