'''
A Double Deep Q-learning implementation to train an RL agent to solve the cartpole problem
'''

# Import libraries

import gym
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import math
import cv2
from matplotlib import animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Environment

env = gym.make('CartPole-v1', render_mode = 'rgb_array')

# Device

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')

print('Using', device)

# Hyperparameters

batch_size = 32
gamma = 0.99
epsilon_start = 1
epsilon_end = 0.1
epsilon_decay = 1e-2
C = 50 # Update rate of target Q-network
alpha = 1e-4


# Define replay buffer

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], capacity)
    
    def push(self, *args):
        '''
        Save a transition
        '''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        '''
        Sample a batch of transitions
        '''
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Define the Q-network

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Dropout(0.2)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Dropout(0.2)
        self.layer6 = nn.Linear(64, n_actions)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        y = self.layer6(x)

        return y # return Q values corresponding to each action
    

# Define epslion-greedy

def select_action(state, step, policy_net):

    epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1 * step * epsilon_decay)

    x = random.random()

    if x < epsilon:

        return torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)
    
    else:

        with torch.no_grad():

            return policy_net(state).max(1).indices.view(1,1)
        

def optimize_model(memory, policy_net, target_net, optimizer):

    if len(memory) < batch_size:

        return
    
    transition = memory.sample(batch_size)

    batch = Transition(*zip(*transition)) # Convert batch of transitions to transitions of batches

    # Compute mask for non-final states
    non_final_mask = torch.tensor(tuple(map(lambda state: state is not None, batch.next_state)), device = device, dtype = torch.bool)

    non_final_next_states = torch.cat([state for state in batch.next_state if state is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device = device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Find estimated Q-values
    estimated_state_action_values = reward_batch + next_state_values * gamma

    # Huber loss
    # criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, estimated_state_action_values.unsqueeze(1))

    # Optimize
    optimizer.zero_grad()
    loss.backward()

    # Deal with exploding gradients 
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

    optimizer.step()

def training(num_episodes, policy_net, memory, target_net, optimizer):

    total_reward = 0

    for episode in range(num_episodes):

        state, _ = env.reset()
        state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)

        terminated = False
        truncated = False

        step = 0

        while not (terminated or truncated):

            action = select_action(state, step, policy_net)
            observation, reward, terminated, truncated, _ = env.step(action.item())

            step += 1
            total_reward += reward
            reward = torch.tensor([reward], device = device)
            

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype = torch.float32, device = device).unsqueeze(0)
            
            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer)
            

            # Soft update target network

            if step % C == 0:
          
                target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % (num_episodes / 10) == 0:
            print(f'Average total reward obtained every {(num_episodes / 10)} episodes is {total_reward / (num_episodes / 10)} ')
            total_reward = 0

            
    print('Training complete')

def testing(policy_net, num_episodes, save_path = 'animation'):

    # Test agent
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
        terminated = False
        truncated = False

        frames = []
        total_reward = 0
        steps = 0
        while not (terminated or truncated):

            frame = env.render()
            frames.append(frame)

            with torch.no_grad():
                action = policy_net(state).max(1).indices.item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            state = torch.tensor(next_state, dtype = torch.float32, device = device).unsqueeze(0)
            
            steps += 1

        print('Testing complete')
        print(f'Achieved a total reward of {total_reward} in {steps} steps')

        # Display performance

        fig = plt.figure()
        plt.axis('off')
        images = [[plt.imshow(frame, animated = True)] for frame in frames]

        Animation = animation.ArtistAnimation(fig, images, interval = 50, blit = True)

        Animation.save(f'{save_path}.mp4')
        plt.show()


def main():

    
    # Environment info

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    # Initialize the Q-network and target Q-network

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    
    target_net.load_state_dict(policy_net.state_dict()) # Copy weights

    optimizer = optim.AdamW(policy_net.parameters(), lr = alpha, amsgrad = True)
    # optimizer = optim.SGD(policy_net.parameters(), lr = alpha)

    capacity = int(1e5)
    memory = ReplayMemory(capacity)

    num_episodes = 100000

    # Training
    training(num_episodes, policy_net, memory, target_net, optimizer)

    # Testing
    testing(policy_net, num_episodes = 1)

    

if __name__ == '__main__':
    main()