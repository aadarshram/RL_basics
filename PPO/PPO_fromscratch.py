'''
Implements PPO-clip from scratch with setting mentioned in config file
'''

# Import libraries

import gym
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PPO:

    def __init__(self, env):
        self.env = env
        self.state_space = env.state_space.shape[0]
        self.action_space = env.action_space.shape[0]

        self.actor = Network(self.state_space, self.action_space)
        self.critic = Network(self.state_space, 1) # Ouput 1 value function for given state
    
    def update(self, num_episodes):

        for episode in range(num_episodes):

    def rollout(self):
        # Get batch data

        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_rewards_togo = []
        batch_lengths = []

        t = 0
        while t < self.timesteps_per_batch:
            episode_rewards = []
            state = self.env.reset()
            
            for time  in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(state)
                action, log_prob = self.get_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)

                episode_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                if (terminated or truncated):
                    break
            batch_lengths.append(time + 1)
            batch_rewards.append(episode_rewards)

        batch_obs = torch.tensor(batch_obs, dtype = torch.float)
        batch_actions = torch.tensor(batch_actions, dtype = torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype = torch.float)
        batch_rewards_togo = self.compute_rewards_togo(batch_rewards)

        return batch_obs, batch_actions, batch_log_probs, batch_rewards_togo, batch_lengths
    
    def get_action(self, state):

        mean_action = self.actor(state)
        distribution = MultivariateNormal(mean_action, self.covariance_matrix)

        action = distribution.sample()
        log_probs = distribution.log_probs(action)

        return action.detach().numpy(), log_probs.detach()
    
    def compute_rewards_togo(self, batch_rewards):
        batch_rewards_togo = []

        for episode_rewards in batch_rewards:
            discounted_reward = 0
            for reward in episode_rewards[::-1]:
                discounted_reward = reward + self.gamma * discounted_reward
                batch_rewards_togo.insert(0, discounted_reward)
        
class Network(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Network, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)
    
    def forward(self, x):

        x = torch.tensor(x, dtype = torch.float)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x