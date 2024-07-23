'''
Implement a basic Q-learning algorithm from scratch for any general gym environment and hyperparameter choice given in config.json
'''

# Import libraries
import numpy as np
import gymnasium as gym
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

def _train_agent(env, Qtable, epsilon_start, num_episodes, gamma, alpha, epsilon_end, epsilon_decay):

    # Train for multiple seeds
    episode_rewards_overseeds = []
    episode_avg_rewards_overseeds = []
    TD_errors_overseeds = []
    for seed in [1, 2, 3, 5, 8]:
        # Set seed
        np.random.seed(seed)
        
        epsilon = epsilon_start
        episode_rewards = []
        episode_avg_rewards = []
        TD_errors = []

        for episode in range(num_episodes):
            state = env.reset(seed = seed)[0]
            total_reward = 0
            terminated = False
            truncated = False

            steps = 0
            total_td_error = 0
            while not (terminated or truncated):
                steps += 1
                # Epsilon greedy
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Qtable[state, :])
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                td_error = reward + gamma * np.max(Qtable[next_state, :]) - Qtable[state, action]
                total_td_error += td_error
                Qtable[state][action] += alpha * td_error

                state = next_state
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(- epsilon_decay * episode)
            if episode % (num_episodes / 100):
                print(f'Episode {episode + 1}: Total Reward = {total_reward}; Steps = {steps}')
            episode_rewards.append(total_reward)
            episode_avg_rewards.append(total_reward / steps)
            TD_errors.append(total_td_error / steps)
        episode_rewards_overseeds.append(episode_rewards)
        episode_avg_rewards_overseeds.append(episode_avg_rewards)
        TD_errors_overseeds.append(TD_errors)

    env.close()

    return episode_rewards_overseeds, episode_avg_rewards_overseeds, TD_errors_overseeds

def _test_agent(env_name, Qtable, num_test_episodes, test_video_path):


    eval_env = gym.make(env_name, render_mode = 'rgb_array')
    total_rewards = []

    for _ in range(num_test_episodes):

        state = eval_env.reset()[0]
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = np.argmax(Qtable[state, :])
            next_state, reward, terminated, truncated, _ = eval_env.step(action)

            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
    
    print(f'Test reward over {num_test_episodes} episodes: {np.mean(total_rewards)} +/- {np.std(total_rewards)}')


    # Create and save animation for 1 test episode

    state = eval_env.reset(seed = random.randint(0, 100))[0]
    terminated = False
    truncated = False
    frames = []
    while not (terminated or truncated):
        action = np.argmax(Qtable[state, :])
        next_state, reward, terminated, truncated, _ = eval_env.step(action)

        state = next_state

        frame = eval_env.render()
        frames.append(frame)

    eval_env.close()

    fig = plt.figure()
    plt.axis('off')

    images = [[plt.imshow(frame, animated = True)] for frame in frames]

    Animation = animation.ArtistAnimation(fig, images, interval = 50,blit = True, repeat_delay = 1000)

    Animation.save(f'{test_video_path}.mp4')
    plt.show()

    return None
    
def _plot_training_results(episode_rewards_overseeds, episode_avg_rewards_overseeds, TD_errors_overseeds):

    fig, axs = plt.subplots(3, figsize = (10,10))

    for episode_rewards, episode_avg_rewards, TD_errors in zip(episode_rewards_overseeds, episode_avg_rewards_overseeds, TD_errors_overseeds):

        axs[0].plot(episode_rewards, alpha = 0.5)
        axs[1].plot(episode_avg_rewards, alpha = 0.5)
        axs[2].plot(TD_errors, alpha = 0.5)
    
    axs[0].set_title('Cumulative rewards vs Episodes')
    axs[1].set_title('Avg reward vs Episodes')
    axs[2].set_title('Avg TD error vs Episodes')

    plt.tight_layout()
    plt.savefig('Training_results.png')

    return None

def Qlearning_Agent(env_name, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay, num_episodes, num_test_episodes, test_video_path):
   

    # Environment 

    env = gym.make(env_name, render_mode = 'rgb_array')
    # env.seed(0)
    state_space = env.observation_space.n
    action_space = env.action_space.n

    # Set seed for reproducability

    
    # Initialize Q-table

    Qtable = np.zeros((state_space, action_space))

    # Training
    episode_rewards_overseeds, episode_avg_rewards_overseeds, TD_errors_overseeds = _train_agent(env, Qtable, epsilon_start, num_episodes, gamma, alpha, epsilon_end, epsilon_decay)

    # Testing
    _test_agent(env_name, Qtable, num_test_episodes, test_video_path)
    
    # Plot training results
    _plot_training_results(episode_rewards_overseeds, episode_avg_rewards_overseeds, TD_errors_overseeds)
    '''
    exploration vs exploitation distribution
    epsilon curve
    alpha curve
    different exploration strategy?
    adeeeeeeeeee
    sarsa try
    monte carlo methpds
    HER
    '''
    return None

if __name__ == '__main__':

    # Get example config
    with open('example_config.json', 'r') as file:
        config = json.load(file)

    env_name = config['env']
    alpha = config['hyperparameters']['alpha']
    gamma = config['hyperparameters']['gamma']
    epsilon_start = config['hyperparameters']['epsilon_start'] 
    epsilon_end = config['hyperparameters']['epsilon_end'] 
    epsilon_decay = config['hyperparameters']['epsilon_decay'] 
    num_episodes = config['hyperparameters']['num_episodes']
    num_test_episodes = config['hyperparameters']['num_test_episodes']
    test_video_path = config['test_video']

    Qlearning_Agent(env_name, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay, num_episodes, num_test_episodes, test_video_path)


