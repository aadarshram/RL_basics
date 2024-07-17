'''
Generalized Policy Iteration (GPI):
This is one of the most basic methods that directly uses bellman equation to find the optimal policy. This approach is based on dynamic programming and works for finite states and actions and when transition probabilities are known (model based)
'''

# Import libraries
import numpy as np

def PolicyIteration(gamma, err, N_STATES, N_ACTIONS, T, R):

    # Initialize V(s) and pi(s)
    V_s = np.ones(N_STATES) 
    pi_s = np.random.randint(0, N_ACTIONS, size = N_STATES)

    delta = 0

    # Policy evaluation
    def Policy_eval():
        while True:  
            
            for s in range(N_STATES):
                v = 0
                for s_next in range(N_STATES):
                    v += T[s][pi_s[s]][s_next] * (R[s] + gamma * V_s[s_next])
                delta = max(delta, abs(v - V_s[s]))

                V_s[s] = v
                
            if delta < err:
                break

    policy_stable = True

    # Policy improvement
    def Policy_improv():
        global policy_stable
        for s in range(N_STATES):
            a_old = pi_s[s]
            v_best = - np.inf
            a_best = a_old

            for a in range(N_ACTIONS):
                v = 0
                for s_next in range(N_STATES):
                    v += T[s][a][s_next] * (R[s] + gamma * V_s[s_next])  
                if v > v_best:
                    a_best = a
                    v_best = v

            pi_s[s] = a_best

            if pi_s[s] != a_old:
                policy_stable = False
        
    if policy_stable:
        return V_s, pi_s
    else:
        Policy_eval()
        Policy_improv()

# Main ----------------

# Define environment and settings

N_STATES = 3
N_ACTIONS = 2
T = np.random.rand(N_STATES, N_ACTIONS, N_STATES)
T = T / T.sum(axis = 2, keepdims = True)
R = np.random.randint(0, 2, size = N_STATES)

err = 10
gamma = 0.99

V_s, pi_s = PolicyIteration(gamma, err, N_STATES, N_ACTIONS, T, R)

print(V_s, pi_s)
