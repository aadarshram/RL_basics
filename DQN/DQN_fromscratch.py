'''
A Double Deep Q-learning implementation to train an RL agent to solve the cartpole problem
'''

# Import libraries

import gym
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import math
from matplotlib import animation
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.memory = deque([], capacity)
        self.Transition = Transition
    def push(self, *args):
        '''
        Save a transition
        '''
        self.memory.append(self.Transition(*args))

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
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 32)
        self.layer3 = nn.Dropout(0.2)
        self.layer4 = nn.Linear(32, n_actions)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        y = self.layer4(x)
        
        return y # return Q values corresponding to each action
    

# Define epslion-greedy

def select_action(env, state, episode, policy_net, device, epsilon_start, epsilon_end, epsilon_decay):

    epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1 * episode * epsilon_decay)

    x = random.random()

    if x < epsilon:

        return torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)
    
    else:

        with torch.no_grad():

            return policy_net(state).max(1).indices.view(1,1)
        

def optimize_model(memory, policy_net, target_net, optimizer, Transition, device, gamma, batch_size):

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

    # Error clipping tends to better performance
    err = 1
    torch.clamp(loss, - err, err)

    # Optimize
    optimizer.zero_grad()
    loss.backward()

    # Deal with exploding gradients 
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

    optimizer.step()

# Training

def training(env, num_episodes, policy_net, memory, target_net, optimizer, device, Transition, epsilon_start, epsilon_end, epsilon_decay, training_starts, gamma, batch_size, C):

    print('Training...')
    total_reward = 0
    episodes = 0

    for episode in tqdm(range(num_episodes)):

        episodes += 1

        if episode < training_starts:
            pass

        state, _ = env.reset()
        state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
        terminated = False
        truncated = False

        while not (terminated or truncated):

            action = select_action(env, state, episode, policy_net, device, epsilon_start, epsilon_end, epsilon_decay)
            observation, reward, terminated, truncated, _ = env.step(action.item())

            total_reward += reward
            reward = torch.tensor([reward], device = device)
            

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype = torch.float32, device = device).unsqueeze(0)
            
            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer, Transition, device, gamma, batch_size)
            

            # Soft update target network

            if episodes % C == 0:
          
                target_net.load_state_dict(policy_net.state_dict())

                
    print('Training complete')

# Testing

def testing(env, policy_net, num_episodes, save_path, device):

    print('Testing...')

    # Test agent
    for episode in tqdm(range(num_episodes)):
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


def DQN_agent(env_name, alpha, gamma, batch_size, C, buffer_size, epsilon_start, epsilon_end, epsilon_decay, num_episodes, num_test_episodes, test_video_path, training_starts):

    env = gym.make(env_name, render_mode = 'rgb_array')

    # Device

    device = torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()
                        else 'cpu')

    print('Using', device)

    # Define replay buffer

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    
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

    capacity = int(buffer_size)
    memory = ReplayMemory(capacity, Transition)

    # Training
    training(env, num_episodes, policy_net, memory, target_net, optimizer, device, Transition, epsilon_start, epsilon_end, epsilon_decay, training_starts, gamma, batch_size, C)

    # Testing
    testing(env, policy_net, num_test_episodes, test_video_path, device)

    return None

if __name__ == '__main__':

    # Get config

    with open('example_config.json', 'r') as file:
        config = json.load(file)

    env_name = config['env']
    alpha = config['hyperparameters']['alpha']
    gamma = config['hyperparameters']['gamma']
    batch_size = config['hyperparameters']['batch_size']
    C = config['hyperparameters']['trgt_update_every']
    buffer_size = config['hyperparameters']['buffer_size']
    epsilon_start = config['hyperparameters']['epsilon_start']
    epsilon_end = config['hyperparameters']['epsilon_end']
    epsilon_decay = config['hyperparameters']['epsilon_decay'] 
    num_episodes = config['hyperparameters']['num_episodes']
    num_test_episodes = config['hyperparameters']['num_test_episodes']
    training_starts = config['hyperparameters']['training_starts']
    test_video_path = config['test_video']

    DQN_agent(env_name, alpha, gamma, batch_size, C, buffer_size, epsilon_start, epsilon_end, epsilon_decay, num_episodes, num_test_episodes, test_video_path, training_starts)

