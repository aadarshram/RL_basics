# Q-Learning

- Tabular. Stores Q-values in a table spanning all state-action pairs. (Impractical for large state-action spaces or continuous spaces).
- Model-free. Learns from sample path.
- Value-based. Value functions of state-action pairs, Q(s, a).
- Off-policy. The policy unrolled in the environment is different from one used in the update. Particularly, to tackle-exploration exploitation trade-off, the unrolled policy can devaite from greedy to explore new states but the update assumes a greedy policy to maximize the expected cumulative return.

Once, Q-values are learnt, then the optimal policy for every state s, is the action a s.t Q(s, a) is maximum.

The optimal Q-value function is determined by the Q-learning algorithm which given a policy (here, greedy), does TD update until convergence.

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
        Q-learning update:
        Q(St, At) += alpha * (Rt+1 + gamma * max_a Q(St+1, a) - Q(St, At))
        t = t + 1
end for
return Q
```

**NOTE**:
- Q-Learning may also be called **SARSA Max**. **SARSA** (State-Action-Reward-State-Action) refers to an on-policy variant which uses the same unrolled policy for evaluation as well. *(See SARSA/)*
- **GLIE**{epsilon_i}: A schedule of epsilon values over the episodes. The algorithm is s.t it is greedy in the limit with infinite exploration. Conditions epsilon to decay over time (greedy in the limit) but $\sum$epsilon = $\infty$ (infinite exploration). This is important for the algorithm to converge.
- The Q-values approximately converge to Q_$\pi$ if num_episodes is large enough.
- The simplest epsilon schedule called **epsilon greedy** is to use epsilon as a threshold against a random probability to choose any action (explore) or one with highest Q-value (greedy; exploit). Advanced methods exist that encourage efficient exploration and exploitation balance (TODO: Maybe a note on this).
- Works for environments with low-dimensional finite state and action spaces. FOr higher-dimensional problems, Q-value table may be approximated via a function. *(See, DQN/)*.



**TODO**:
- Implementation of the algorithm Frozen lake and taxi - https://huggingface.co/learn/deep-rl-course/en/unit2/hands-on  
- Additional reading - ch 5,6,7 in Sutton.
- L2 by Pieter Abbeel - youtube.com/watch?v=Psrhxy88zww&feature=youtu.be

