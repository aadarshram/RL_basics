'''
This is value iteration algorithm to find optimal policy. It is based on dymanic programming and requires finite states and actions and also known environment dynamics (model based)
'''

# import libraries
import numpy as np

# Algorithm
def ValueIteration(N_STATES, N_ACTIONS, err, gamma, T, R):

    V_s = np.ones(N_STATES)

    delta = 0

    while True:
        for s in range(N_STATES):
            v_best = - np.inf
            for a in range(N_ACTIONS):
                v = 0
                for s_next in range(N_STATES):
                    v += T[s][a][s_next] * (R[s] + gamma * V_s[s_next])  
                if v > v_best:
                    v_best = v
            delta = max(delta, v - V_s[s])
            V_s[s] = v

        if delta < err:
            break

    pi_s = np.random.randint(0, N_ACTIONS, size = N_STATES)

    for s in range(N_STATES):
        v_best = - np.inf

        for a in range(N_ACTIONS):
            v = 0
            for s_next in range(N_STATES):
                v += T[s][a][s_next] * (R[s] + gamma * V_s[s_next])  
            if v > v_best:
                a_best = a
                v_best = v

        pi_s[s] = a_best


    return pi_s

# Main ----------------

# Define environment and settings

N_STATES = 3
N_ACTIONS = 2
T = np.random.rand(N_STATES, N_ACTIONS, N_STATES)
T = T / T.sum(axis = 2, keepdims = True)
R = np.random.randint(0, 2, size = N_STATES)

err = 10
gamma = 0.99

pi_s = ValueIteration(N_STATES, N_ACTIONS, err, gamma, T, R)

print(pi_s)

    
