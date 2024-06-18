'''
Implement Q-learning for a robot to navigate a simple environment avoiding obstacles
'''

# Import libraries
import numpy as np
import time

# Environment
grid_size = 5
num_actions = 4
initial_state = (0,0)
goal_state = (grid_size - 1, grid_size - 1)
obstacles = [(1,1), (2,2), (3,3)]

# Initialize Q-table
q_table = np.zeros((grid_size, grid_size, num_actions))


# Define reward
def get_reward(state):
    if state == goal_state:
        return 10
    elif state in obstacles:
        return -10
    else:
        return -1
    
# Define transition function
def get_next_state(state, action):
    x, y = state
    if action == 0: # Up
        next_state = (x, min(y + 1, grid_size - 1))
    elif action == 1: # Down
        next_state = (x, max(y - 1, 0))
    elif action == 2: # Left
        next_state = (max(0, x - 1), y)
    else: # Right
        next_state = (min(x +1, grid_size - 1), y)

    return next_state

# Training

def training(num_episodes, alpha, gamma, epsilon, max_steps):

    for episode in range(num_episodes):
        state = initial_state
        done = False

        steps = 0
        while not done:

            steps = steps + 1

            # epsilon-greedy
            if np.random.uniform(0,1) < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(q_table[state])
            
            # Update
            next_state = get_next_state(state, action)
            reward = get_reward(next_state)

            # Update Q-table
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            # Termination
            if state == goal_state or state in obstacles:
                done = True
            elif steps == max_steps:
                print('Max steps reached.')
                done = True

    print('Training done!')

# Testing
def testing(max_steps):
    print('Testing')
    start_time = time.time()
    state = initial_state
    done = False

    steps = 0

    while not done:

        steps += 1

        action = np.argmax(q_table[state])
        next_state = get_next_state(state, action)
        state = next_state

        if state == goal_state:
            end_time = time.time()
            print(f'Reached goal in {end_time - start_time} seconds')
            done = True
        elif state in obstacles:
            print('Oops, I hit an obstacle!')
            done = True
        elif steps == max_steps:
            print('Max steps reached.')
            done = True
    
    print('Testing done!')

def main():

    num_episodes = 1000

    # Hyperparameters
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.2

    max_steps = 1000 # Maximum steps allowed in each episode (helps in converging)

    # Experimenting with different hyperparameters
    training(num_episodes, alpha, gamma, epsilon, max_steps)
    testing(max_steps)

if __name__ == '__main__':
    main()