''' 
Implements REINFORCE algorithm from scratch for setting defined in config.json
'''

# Import libraries

import random
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym

# Get config
with open('config.json', 'r') as file:
    config = json.load(file)

    env_name = config['env']
    alpha = config['hyperparameters']['alpha']
    gamma = config['hyperparameters']['gamma']
    num_episodes = config['hyperparameters']['num_episodes']
    num_test_episodes = config['hyperparameters']['num_test_episodes']
    save_path = config['test_video']

class policy_network(nn.Module):
    
    def __init__(self, state_space, action_space):
        super().__init__()

        self.shared_net = nn.Sequential(
            nn.Linear(state_space, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.policy_mean_net = nn.Sequential(nn.Linear(128, action_space)) # To find mean of the action distribution
        self.policy_stddev_net = nn.Sequential(nn.Linear(128, action_space)) # To find stddev of the action distribution

    def forward(self, x):

        shared_features = self.shared_net(x.float())
        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.nn.Softplus()(self.policy_stddev_net(shared_features)) # Softplus is used to ensure positive outputs and better training stability.
        # Softplus(x) = 1/b * log(1+exp(b * x))

        return action_means, action_stddevs


class REINFORCE:

    def __init__(self, state_space, action_space):

        
        self.probs = [] # Stores probability values of sampled action
        self.rewards = [] # Stores corresponding rewards

        self.net = policy_network(state_space, action_space)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr = alpha)

    def sample_action(self, state):

        state = torch.tensor(np.array(state))
        action_means, action_stddevs = self.net(state)

        # Create distribution
        dist = Normal(action_means[0] + 1e-6, action_stddevs[0] + 1e-6) # Noise for mean and prevent 0 std dev (degenerate case)

        action = dist.sample(sample_shape = (1,))

        prob = dist.log_prob(action)

        self.probs.append(prob)

        action = action.numpy()
        return action
    
    def update(self):

        running_G = 0
        Gs =[]

        for R in self.rewards[::-1]:
            running_G = R + gamma * running_G
            Gs.insert(0, running_G)

        deltas = torch.tensor(Gs)

        loss = 0

        for log_prob, delta in zip(self.probs, deltas):
            loss -= log_prob * delta

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []

# Create environment

env = gym.make(env_name)
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

# We check for multiple seeds because it can vary based on seeds
rewards_over_seeds = []

for seed in [1, 2, 3, 5, 8]: # Fibonacci seeds

    print(f'Seed {seed}')

    # Set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Initialize agent
    agent = REINFORCE(state_space, action_space)
    
    reward_over_episodes = []

    for episode in range(num_episodes):

        initial_state, info = wrapped_env.reset(seed = seed)
        state = initial_state
        done = False

        while not done:
            action = agent.sample_action(state)
            next_state, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            state = next_state
            done = terminated or truncated
    
        reward_over_episodes.append(wrapped_env.return_queue[-1])

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}; Reward {wrapped_env.return_queue[-1][0]}")
        # Monte Carlo update
        agent.update()

    rewards_over_seeds.append(reward_over_episodes)

# Plot training results

plt.figure()

for seed_index, seed_rewards in enumerate(rewards_over_seeds):
    episodes = list(range(len(seed_rewards)))
    seed_rewards = list(map(lambda x: x[0], seed_rewards))
    plt.plot(episodes, seed_rewards, label = f'Seed {seed_index + 1}')

plt.title('Reward vs Episodes for multiple seeds')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()
plt.grid(True)
plt.show()

# Testing

test_env = gym.make(env_name, render_mode = 'human')
for test_episode in range(num_test_episodes):
    state, info = test_env.reset(seed  = random.randint(0, 100))
    done = False
    total_test_reward = 0
    frames = []

    while not done:
        action = agent.sample_action(state)
        next_state, reward, terminated, truncated, info = test_env.step(action)
        total_test_reward += reward

        frame = test_env.render()
        frames.append(frame)

        done = terminated or truncated
        state = next_state
    print(f'Achieved reward of {total_test_reward}')

    # Display performance

    fig = plt.figure()
    plt.axis('off')
    images = [[plt.imshow(frame, animated = True)] for frame in frames]

    Animation = animation.ArtistAnimation(fig, images, interval = 50, blit = True)

    Animation.save(f'{save_path}.mp4')
    plt.show()

test_env.close()