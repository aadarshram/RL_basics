'''
Implement a Deep Q-learning architecture to solve the cartpole problem where the goal is to balance an inverted pendulum from falling
'''

# Import libraries

import gym

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import deque
import random

# parameters

use_cuda = True
episode_limit = 100
target_update_delay = 2 # update target network every 2 episodes
test_delay = 10
lr = 1e-4 # learning rate
# for exponential decaying of epsilon in epsilon greedy method
max_epsilon = 0.99
min_epsilon = 0.1
epsilon_decay = 0.9 / 2.5e3 # need update
gamma = 0.99 # discount factor
memory_len = 10000

# Create environment

env = gym.make('CartPole-v1')

# Device

device = torch.device("cuda" if use_cuda and torch.cuda.is_available() 
                      else "cpu")

n_features = len(env.observation_space.high)
n_actions = env.action_space.n

memory = deque(maxlen = memory_len) # deque of tuples : (state, action, reward, next state)

# define the model

class Model(nn.Module):
  def __init__(self,n_features, n_actions):
    super(Model, self).__init__()
    self.fc1 = nn.Linear(n_features, 32)
    self.fc2 = nn.Linear(32, 32)
    self.fc3 = nn.Linear(32, n_actions)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

criterion = nn.MSELoss() # loss function
policy_net = Model(n_features, n_actions) # Q network
target_net = Model(n_features, n_actions) # Target Q network
target_net.load_state_dict(policy_net.state_dict()) # copy the parameters of Q network to Target network
target_net.eval() # set it in evaluate mode

# function to generate state tensor

def get_state_tensor(sample, states_idx):
  sample_len = len(sample)
  states_tensor = torch.empty((sample_len, n_features), dtype = torch.float32, requires_grad = False) # create empty state tensor

  for i in range(sample_len): # find states vector from sample
    for j in range(n_features):
      states_tensor[i, j] = sample[i][states_idx][j]
  return states_tensor

# function to get the action using epsilon-greedy method

def get_action(state, epsilon = min_epsilon):
  if random.random() < epsilon: # explore
    action = random.randrange(0, n_actions)
  else: # exploit
    state = torch.tensor(state, dtype = torch.float32)
    action = policy_net(state).argmax.item() # action with max Q return
  return action

# fit function

def fit(model, input, label):
  #input = input.to(device)
  #label = label.to(device)
  train_df = TensorDataset(input, label) # create a dataframe for training
  train_dl = DataLoader(train_df, batch_size = 5)
  optimizer = torch.optim.Adam(params = model.parameters(), lr = lr) # define optimizer function
  model.train() # train mode
  total_loss = 0
  for x,y in train_dl:
    output = model(x)
    loss = criterion(output, y)
    total_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  model.eval() # set back to eval mode
  return total_loss / len(input) # return average loss

# define optimization step

def optimize_model(train_batch_size = 100):
  train_batch_size = min(train_batch_size, len(memory)) # upper limit set by memory
  train_sample = random.sample(memory, train_batch_size) # sample random transition
  state = get_state_tensor(train_sample, 0)
  next_state = get_state_tensor(train_sample, 3)
  q_estimates = policy_net(state) # estimate Q
  next_state_q_estimates = target_net(next_state) # estimate Q'+
  fit(policy_net, state, (train_sample[2] + gamma * next_state_q_estimates)) # loss is MSE in Q' and Q
  return None

def train_one_episode():
  global epsilon
  current_state = env.reset() # inital state
  done = False
  net_reward = 0
  while not done:
    action = get_action(current_state, epsilon) # get action
    next_state, reward, done, _ = env.step(action) # do the action and get feedback
    memory.append(current_state, action, next_state, reward) # replay memory
    optimize_model(100)
    current_state = next_state
    total_reward += reward
    epsilon -= epsilon_decay
  return total_reward

def test():
  state = env.reset()
  done = False
  total_reward = 0
  while not done:
    action = get_action(state) 
    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward
  return total_reward

def main():
  best_test_reward = 0
  for i in range(episode_limit):
    total_reward = train_one_episode()

    print(f'Episode {i+1}: score {total_reward}')
print("a")