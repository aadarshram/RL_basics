import sys
sys.path.insert(0, '../')
from Qlearning import Qlearning_fromscratch
from DQN import DQN_fromscratch

# Hyperparameters
env_name = 'FrozenLake-v1'
alpha = 0.03
gamma = 0.99
epsilon_start = 0.8
epsilon_end = 0.2
epsilon_decay = 0.005
num_episodes = 1000
num_test_episodes = 100
test_video_path = 'animation'

# Call agent
Qlearning_fromscratch.Qlearning_Agent(env_name, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay, num_episodes, num_test_episodes, test_video_path)
# batch_size = 32
# C = 20
# buffer_size = 10000
# training_starts = 10
# DQN_fromscratch.DQN_agent(env_name, alpha, gamma, batch_size, C, buffer_size, epsilon_start, epsilon_end, epsilon_decay, num_episodes, num_test_episodes, test_video_path, training_starts)