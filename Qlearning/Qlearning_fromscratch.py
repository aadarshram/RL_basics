'''
Implement a basic Q-learning algorithm from scratch for any general gym environment and hyperparameter choice given in config.json
'''

# Import libraries
import numpy as np
import gymnasium as gym
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Get config
with open('config.json', 'r') as file:
    config = json.load(file)

env_name = config['env']
alpha = config['hyperparameters']['alpha']
gamma = config['hyperparameters']['gamma']
epsilon_start = config['hyperparameters']['epsilon_start'] 
epsilon_end = config['hyperparameters']['epsilon_end'] 
epsilon_decay = config['hyperparameters']['epsilon_decay'] 
num_episodes = config['hyperparameters']['num_episodes']
num_test_episodes = config['hyperparameters']['num_test_episodes']
test_video = config['test_video']
# Environment 

env = gym.make(env_name, render_mode = 'rgb_array')

state_space = env.observation_space.n
action_space = env.action_space.n

# Initialize Q-table

Qtable = np.zeros((state_space, action_space))

# Training

epsilon = epsilon_start
for episode in range(num_episodes):
    state = env.reset()[0]
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # Epsilon greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Qtable[state, :])
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        td_error = reward + gamma * np.max(Qtable[next_state, :]) - Qtable[state, action]

        Qtable[state][action] += alpha * td_error

        state = next_state

    epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(- epsilon_decay * episode)
    print(f'Episode {episode + 1}: Total Reward = {total_reward}')

# env.close()

# Testing

# eval_env = gym.make(env_name, render_mode = 'rgb_array')
eval_env = env
frames = []
total_rewards = []

for episode in range(num_test_episodes):

    state = eval_env.reset()[0]
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = np.argmax(Qtable[state, :])
        next_state, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        state = next_state

        frame = env.render()
        frames.append(frame)
    total_rewards.append(total_reward)

print(f'Test reward over {num_test_episodes} episodes: {np.mean(total_rewards)} +/- {np.std(total_rewards)}')
eval_env.close()

# Create and save animation

fig = plt.figure()
plt.axis('off')

images = [[plt.imshow(frame, animated = True)] for frame in frames]

Animation = animation.ArtistAnimation(fig, images, interval = 50,blit = True, repeat_delay = 1000)

Animation.save(f'{test_video}.mp4')
plt.show()


