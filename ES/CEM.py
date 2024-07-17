'''
Implements Cross Entropy Optimization Method for a simple policy network to solve an RL problem
'''

# Import libraries
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib import animation

# Create environment
env = gym.make('CartPole-v1', render_mode = 'rgb_array')

def select_elite(candidates, rewards, elite_threshold):
    num_elites = int(elite_threshold * len(candidates))
    elite_indices = np.argsort(rewards)[- num_elites:]
    elite_samples = [candidates[i] for i in elite_indices]

    return np.array(elite_samples)


def CEM(env, num_iterations, num_samples, elite_threshold):
    
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    W_means = np.zeros((state_space, action_space))
    W_std_devs = np.ones((state_space, action_space))

    for iter in range(num_iterations):

        # Check samples

        candidates = np.random.normal(W_means, W_std_devs, size = (num_samples, state_space, action_space))
        rewards = np.zeros(num_samples)

        for i in range(num_samples):

            policy_weights = candidates[i]

            state, _ = env.reset()
            total_reward = 0
            
            while True:
                action_probs = np.dot(state, policy_weights)
                action = np.argmax(action_probs)

                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                if (terminated or truncated):
                    break

            rewards[i] = total_reward
        
        # Choose elite

        elite_candidates = select_elite(candidates, rewards, elite_threshold)
        W_means = np.mean(elite_candidates, axis = 0)
        W_std_devs = np.std(elite_candidates, axis = 0)

        print(f'Mean reward in iteration {iter + 1} is {np.mean(rewards)}')

    return W_means

# Training

# Hyperpamareters
num_iterations = 10
num_samples = 50
elite_threshold = 0.5

# Training
best_policy = CEM(env, num_iterations, num_samples, elite_threshold)

# Testing
print('Testing')
state, _ = env.reset()

total_reward = 0
frames = []

while True:
    action_probs = np.transpose(np.dot(np.transpose(state), best_policy))
    action = np.argmax(action_probs)

    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    frame = env.render()

    frames.append(frame)

    if (terminated or truncated):
        break

# Display performance

fig = plt.figure()
plt.axis('off')
images = [[plt.imshow(frame, animated = True)] for frame in frames]

Animation = animation.ArtistAnimation(fig, images, interval = 50, blit = True)

Animation.save('animation.mp4')

print('Total reward in test is:', total_reward)
print('Test animations saved @ animation.mp4')
plt.show()
env.close()
