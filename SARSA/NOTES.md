## SARSA (State-Action-Reward-State-Action)

SARSA, much similar to Q-learning, is an on-policy variant of the TD-learning update unlike the latter which updates off-policy. Can imagine as an extension of the exact TD-learning update from state value functions to state-action value functions (Q-values).

Recall, Q-learning update:

Q(St, At) += alpha * (Rt+1 + gamma * max_a Q(St+1, a) - Q(St, At))

SARSA update:

Q(St, At) += alpha * (Rt+1 + gamma * Q(St+1, At+1) - Q(St, At))

**Algorithm**
```
Input: policy pi, num_episodes, alpha, GLIE{epsilon_i}
algorithm
Output: Q 
Initialize Q arbitrarily (eg: Q = 0 $\forall$ s, a)
for i=1 to num_episodes do
    epsilon = epsilon_i
    Observe initial state S0
    t = 0
    Repeat until St = terminal state
        Choose At using pi
        Step At and observe Rt+1, St+1
        SARSA update:
        Q(St, At) += alpha * (Rt+1 + gamma * Q(St+1, At+1) - Q(St, At))
        t = t + 1
end for
return Q
```

**NOTE**:
- Q-learning converges faster to the optimal policy than SARSA since updates are made greedily whilst SARSA may only converge to a suboptimal policy.
- SARSA yields a more conservative policy, though, suboptimal. Useful for costly online learning and stochastic environments.
- Example: Cliff Walking (OpenAI-gym). The goal is to walk from start to end on a cliff wherein the agent receives a reward of -1 for every step and -100 if fallen of the cliff. The optimal policy is to walk along the edge of the cliff which Q-learning rightly converges to. SARSA may converge to a path bit inward and slightly longer but farther from the edge. In case of stochasticity or exploration, SARSA policy may be preferred over Q-learning.

**TODO**:
- Implement SARSA algorithm. 
- Test against Q-learning in Cliff Walking.
