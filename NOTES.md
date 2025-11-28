# Basics of Reinforcement Learning

## Contents
- [Basics of Reinforcement Learning](#basics-of-reinforcement-learning)
  - [Contents](#contents)
  - [What is Reinforcement Learning anyway?](#what-is-reinforcement-learning-anyway)
  - [Markov Decision Processes (MDPs)](#markov-decision-processes-mdps)
    - [Finite-Horizon MDPs](#finite-horizon-mdps)
      - [DP Algorithm](#dp-algorithm)
    - [Infinite Horizon MDPs](#infinite-horizon-mdps)
      - [The Bellman Optimality Equation](#the-bellman-optimality-equation)
    - [Value Iteration](#value-iteration)
    - [Policy Iteration](#policy-iteration)
  - [The Learning Setting](#the-learning-setting)
    - [Temporal Difference Learning (TD)](#temporal-difference-learning-td)
    - [Monte Carlo Policy Evaluation (MCPE)](#monte-carlo-policy-evaluation-mcpe)
  - [Q-learning](#q-learning)
    - [Q- Bellman Equation](#q--bellman-equation)
  - [Deep Q-learning](#deep-q-learning)
  - [Policy-based Methods](#policy-based-methods)
    - [Policy gradient method](#policy-gradient-method)
    - [Policy-gradient theorem](#policy-gradient-theorem)
    - [Monte Carlo REINFORCE](#monte-carlo-reinforce)
  - [Actor-Critic Methods](#actor-critic-methods)
    - [Advantage Actor-Critic (A2C)](#advantage-actor-critic-a2c)
  - [Proximal-Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
    - [Clipped Surrogate Objective](#clipped-surrogate-objective)
  - [TODO FURTHER](#todo-further)

## What is Reinforcement Learning anyway?

One of the natural ways animals learn is through interaction with the environment via trial-and-error. Each interaction provides some feedback which guides the behaviour of the animal in that environment. This is what is known as **Reinforcement Learning**. The feedback is often given as a positive reward or negative reward given that the goal at hand follows the **reward hypothosis**, i.e, can be described as a maximization of an expected cumulative reward over its interactions. This then reinforces certain behaviours/interactions of the animals in/with the environment and learning such behaviour 
forms the essence of reinforcement learning. Formally,

There exists an **agent** that learns a **policy** as a mapping from **state** to **action** by interacting with an **environment** and for each action taken at a certain a state, the agent goes to a **next state** with some **transition probability** and receives a certain **reward** based on the environment. The policy is learnt such that the agent maximizes the expected cumulative reward over its interactions with the environment.

A policy is modelled as a **markov decision process (MDP)** wherein it exploits the **markov property**, ie, the choice of action to take at any step depends only on its current state and not its history. This is to simplify the theoretical problem at hand.

**NOTE**:

- A policy is not strictly speaking a mapping from the state to action but rather **observation** to action. The difference lies in the fact that an agent may not be able to observe the complete state of the environment and so has to learn a policy based on the partial observation he encounters.
- Oftentimes, the environment is stochastic and so it is reasonable to **discount** future rewards when calculating the expected cumulative return. Fun fact: The discount factor is still used in deterministic environments, reason attributed to a need for convergence of the expected cumulative return in infinite horizon problems. Perhaps, the former explanation came about as a conveniant corollary or an additional benefit.
- The interactions could be **episodic** or **continuous**. An episode refers to a series of interactions that lead to a **terminal state** after which an agent stops.
- An interesting challenge in the RL setting is that of the **exploration-exploitation trade-off**. At any instant while an agent interacts with an environment, he can either exploit his learnt policy by choosing actions that maximizes his expected reward or take a random action to explore unknown states. While the former helps in finding the optimal policy (w.r.t states explored thus far) the latter allows to find better solutions, if exists.
- While it is said that an agent learns a policy in an RL setting, it is not entirely true. Although the goal is to find an optimal policy, one can either directly learn the best action for each state or learn **value function**s which describes the value of a state as an expected cumulative reward if continued from the said state. One then chooses actions such that he always lands on the state with the comparatively best value.
- In uncertain environments, a **deterministic policy** may not be the best. One then models a **stochastic policy** which learns a mapping from state to distribution of actions.
- Although an RL problem assumes a markov setup it is only w.r.t its definition of state. A state can be set of information that can completely describe all that's needed for a future step prediction. Eg: You may choose to model a triad history of observations as one state.


**TODO**:
- Formal math
- Some advanced resources for insights.TO READ
  - https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
  - youtube.com/watch?v=2GwBez0D20A&feature=youtu.be 
  - Sutton and Barto - ch 1,2,3


## Markov Decision Processes (MDPs)

The learning paradigm emerges due to the lack of an environment model, particularly, state-action transition probabilities. However, much of its work is built on top of the theory when such probabilities are known in MDP settings.

### Finite-Horizon MDPs

Has a fixed finite number of steps before termination. In this case, the problem becomes easy to solve using a programmer's favourite **DP algorithm**(dynamic programming).

#### DP Algorithm
The DP algorithm follows from the **Optimality Principle** that "every tail trajectory of an optimal trajectory is also optimal". Hence, in order to find the optimal policy one recursively finds every optimal k-step tail policy starting from the terminal state (trivial) till the start. This backward looking algorithm is much efficient than a forward looking variant whose search space explodes exponentially.

**The Algorithm**:  
```
Initialize V_N(X_N) = r(X_N) (trivial)  
For all k = N-1,...,0:  
    V_k(X_k) = max_a E(r(X_K) + V_k+1(X_k+1))  
```
- argmax(.) at every step yields the optimal policy $\pi$.
- Solving backward for the optimal policy enables simplifying the cumulative reward from a current state into a summation of the current intermediate reward and the optimal value function of the next state following the optimality principle. Basically, to find every optimal tail trajectory, maximize the current step reward and then follow its own optimal tail. It can be shown that the final trajectory thus found is indeed optimal.


### Infinite Horizon MDPs

The DP algorithm does not work in case of variable horizon (as in Stochastic Shortest Paths (SSPs)) or infinite horizon (as in discounted MDPs). 

#### The Bellman Optimality Equation
Consider the DP algorithm update and let the horizon k go to infinity. Then, the optimal value function would follow:  
V*(s) = max_a E(r(s) + $\gamma$ V*(s+1))

where $\gamma$ is the discount factor. 

Notice that we can define a Belmann Operator, T such that V*(s) = TV*(s) turning the optimal value function into a fixed point for the said operator. Hence, from the theory of fixed point iterations, one comes up with iterative algorithms which then converges to the optimal value function.

### Value Iteration
Model the Bellman equation into an iterative update:  
V_k+1(s) = max_a E(r(s) + $\gamma$ V_k(s+1)) $\forall$ s

Starting from any arbitrary value function, applying the operator infintely often until convergence yields the optimal value function.

**Algorithm**: 
``` 
Initialize V_0, threshold  
V_k = V_0  
steps = 0  
While steps < threshold:  
    V_k+1(s) = max_a E(r(s) + $\gamma$ V_k(s+1))   $\forall$ s  
    V_k(s) = V_k+1(s)  
```

In practical implementation, one chooses a suitable threshold corresponding to specified error bounds.

Alternative variants:
- **Asynchronous value iteration** updates only 1 state each step which is useful when working with sample paths (preferred).
- **Gauss-Siedel Value iteration** updates all states but sequentially using most recent estimates for every following update.
  
### Policy Iteration
One can also desire to update policies directly rather than through value functions. Similar to the bellman optimality update operator T, one can define a policy specific operator T_$\pi$ that follows a policy $\pi$ instead and yield the policy-specific value function, V_$\pi$.

In policy iteration, one iteratively evaluates a policy finding its value function and then does one step improvement to a better policy. The policy evaluation follows similar to value iteration. The improved policy is determined from a one-step bellman optimality update on the evaluated value function.

**Algorithm**:
```  
Initialize $\pi$\_0, threshold  
V_$\pi$_k = ValueIteration($\pi$_0)  
While steps < threshold  
    $\pi$\_k+1 = argmax E[r(s) + $\gamma$ V_$\pi$_k(s+1)]  
```

In practical implementation, one chooses a suitable threshold corresponding to specified error bounds.

Alternative variants:
- **Asynchronous Policy Iteration** - Can vary the number of states updated at each step and the number of value iteration updates for policy evaluation.

**NOTE**:
- A reinforcement learning problem can be considered of two types- **evaluation** and **control**. Evaluation algorithms solve for value functions given a policy while control algorithms solve for the policy itself. Value iteration is often used for policy evaluation where the discussed algorithm can be modified to follow policy-specific update than one greedily maximizing the returns (optimality update).
- SSPs can be modelled as infinite horizon by assuming recurrent transitions from the terminal state.

**TODO**:
- Implement algorithms- DP, value iteration, policy iteration (variants)


## The Learning Setting

In a real world, we do not have an accurate model of the environment, ie, the transition probabilities which calls for the learning paradigm. One way is to learn the model dynamics itself and repeat the previously discussed or to learn from a sample path (prefered)

Based on the theory of stochastic iterative algorithms, the learning methods tackle this by substituting the expectations with a sample. Then, under certain mathemaical assumptions these algorithms can converge to learn the optimal policy or value.

**NOTE**: 
- The standard template for a fixed point iterative algorithm to solve, say, Hr* = r* is as follows:  
  r_k+1 = r_k + $\alpha$ (h(r) - r_k)

  where h(r) = Hr - r

  Evaluating h(r) = 0 yields the optimal solution r*.

- In case of stochastic iteration H is unknown and hence replaced with an estimate $\hat{H}$ (correspondingly $\hat{h}$)

### Temporal Difference Learning (TD)

The TD(0) version of Temporal difference learning follows the stochastic iterative algorithm template for the policy evaluation method.  
Recall:  
 V_$\pi$\_k+1(s) = E(r(s) + $\gamma$ V_$\pi$\_k(s+1))  

 The expectation is expanded as:  
 r(s) + $\gamma \sum_j$Ps(j)($\pi$(s)) V_$\pi$\_k(j)  

 The TD(0) update follows:  
 V_$\pi$\_k+1(s) = V_$\pi$\_k(s) + $\alpha$ (r(s) + $\gamma \hat{V}$\_$\pi$\_k(s') - V_$\pi$\_k(s))  

 where the value function of the next state is simply substituted with a current estimate of value function of the sampled next state.

 **NOTE**:
 - This method of estimating future cumulative returns with a value function is called **bootstrapping**.
 - Bootstrapping with the current value estimate introduces bias in the algorithm.
 - The expression (r(s) + $\gamma \hat{V}$\_$\pi$\_k(s') - V_$\pi$\_k(s)) is reffered to as the **TD error**.
  
### Monte Carlo Policy Evaluation (MCPE)
Similar to the mean estimation problem, using an average of samples to estimate the true mean, MCPE collects the cumulative return from a sample path and iteratively updates the value function.  

 V_$\pi$\_k+1(s) = V_$\pi$\_k(s) + $\alpha$ (R_k - V_$\pi$\_k(s)) 

 where R_k = $\sum r(s)$ (discounted) until termination.

 **NOTE**:
 - This method is unbiased but has high variance under finite samples.
 - Learning algorithms can either me **offline** or **online**. Offline algorithms wait until termination to update while online variants update at every sample step.
- As a best of both worlds between TD(0) and MCPE introduced is TD($\lambda$) which is uses a $\lambda$-weighted combination of all n-step returns:

    V_$\pi$\_k+1(s) = V_$\pi$\_k(s) + $\alpha$ (1-$\lambda$) $\sum$ $\lambda^n$R_k_n

    where R_k_n = $\sum_n r(s)$ + $\gamma^n$ V_$\pi$\_k($s_n$)

**NOTE**
- Alternative variants of TD and MCPE include updating on **every visit** of a state or only on **first visit** for a given episode. While the former may yield lower variance unlike latter it can be biased due to the dependent returns from the same episode.
- The TD algorithm is based on the Bellman equation and hence heavily relies on the markov property to work. Thus, for non-Markovian or partially markov environments (which is more often the case), Monte Carlo methods or TD($\lambda$) might prove better.

## Q-learning
The above discussion covers learning in the evaluation problem but faces a challenge in control, ie, finding optimal policies. Turns out the stochastic iterative algorithm demands of an operator as an expectation which is not so in case of the Bellman optimality operator (unlike the policy-specific version). To this end, **Q-values** which are value functions of state-action pairs instead are introduced to tackle this.

V(s) = max_a Q(s, a)

### Q- Bellman Equation

Q*(s, a) = E(r(s,a) + $\gamma$ max_b Q*(s+1, b))

Q-learning is then, simply, TD method involving Q-values.

*Algorithm, implementation and further notes can be found in QLearning/*

## Deep Q-learning

Q-learning algorithm being tabular-based method, fails in environments with large state-spaces. One would like to then convert the table into a generalized function approximation for the Q-values. A popular way to learn any function approximation is via deep neural networks. Deep Q-learning (DQN) is a Q-learning implementation via neural network function approximation. Turns out, function approximation brings in new challenges.

*Algorithm, implementation and further notes can be found in DQN/*

## Policy-based Methods

DQNs while impressive, predicts Q-value per action given state and hence cannot model high-dimensional or continuous action spaces. One then attempts to skip the value function prediction and directly predict a distribution over actions given state instead giving rise to **policy-based methods**. Learning action distribution also helps modelling **stochastic policies** wherein a given state may correspond to multiple desired actions be taken.

Parametrize the policy $\pi$(a|s) by $\theta$ and find optimal $\theta$ maximizing the expected cumulative return following $\pi$, J($\theta$).

### Policy gradient method
A subset of policy-based methods, policy gradients maximizes the objective via gradient ascent. (TODO: Brief about other policy based methods: hill climbing, simulated annealing, evolutionary algorithms, etc.)

max_$\theta$ J($\theta$) = E[R($\tau$)] # $\tau$ sampled from $\pi$($\theta$)

Update:  
$\theta$ = $\theta$ + $\alpha$ * $\nabla$ J($\theta$)

### Policy-gradient theorem

To follow the stochastic iteration template (substitute unknown expectation with sample), we desire to find an expression for the expectation of the gradient of the objective described before. This calls for certain mathematical simplications (with fine assumptions) which underlie the policy-gradient theorem.

$\nabla$ J($\theta$) = E[ $\sum$_t $\nabla$\_$\theta$ log $\pi$\_$\theta$ (at | st) R($\tau$)]

### Monte Carlo REINFORCE

The expectation of the gradient figured previously can then be substituted via monte carlo sample mean over a set of trajectories which yields the **Monte Carlo REINFORCE** algorithm.
  
TODO: Algorithm and deep dive (in PolicyGradients/)

**NOTE**
- Policy-based methods have better convergence properties (TODO: brief) but can converge to a local maxima than global. The latter challenge was introduced by the non-convexity of the parametrized objective (value-based methods yield convex objective and hence global maxima).
- REINFORCE algorithm has high variance (due to monte carlo returns).

**TODO**
- Additional readings and implementation.
- https://www.youtube.com/watch?v=y3oqOjHilio
- hands on cartpole and pixelcopter - https://huggingface.co/learn/deep-rl-course/en/unit4/hands-on 
- ch 13 suttom barto
- https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
- https://johnwlambert.github.io/policy-gradients/
- https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py
- https://github.com/MrSyee/pg-is-all-you-need
- https://jonathan-hui.medium.com/rl-policy-gradients-explained-9b13b688b146 


***NOTE:Beyond this section, I only have a running understanding of the concepts and is prone to volatile updates!***

## Actor-Critic Methods

The policy-gradient method, REINFORCE faces high variance and hence sample inefficiency (variance asympotitically converges to 0 with infinite samples) and slow learning owing to its Monte Carlo estimate of the returns. Actor-Critic methods, instead use a Critic to estimate the value function of the state-action to bootstrap the Monte Carlo return in the Actor update (control).

Actor: St -> At
Critic: {St, At} -> Q(St, At)
Update:
  Actor: d$\theta$ = $\alpha$ * $\nabla$ log$\pi$(s,a) Q(s, a)

The Critic, as usual, is update via on-policy TD-learning.

(Env: St+1, Rt+1)
Actor: St+1 -> At+1
Critic: {St+1, At+1} -> Q(St+1, At+1)
Update:
  Critic: dw = $\beta$ * (Rt+1 + $\gamma$ * Q(St+1, At+1) - Q(St, At)) $\nabla$ Q(St, At)

**TODO**:
- More reading on bias variance tradeoffs
  - https://blog.mlreview.com/making-sense-of-the-bias-variance-trade-off-in-deep-reinforcement-learning-79cf1e83d565 
  - https://www.endtoend.ai/blog/bias-variance-tradeoff-in-reinforcement-learning/
  - https://balajiai.github.io/high_variance_in_policy_gradients

### Advantage Actor-Critic (A2C)

Turns out, a better signal to the actor would be the relative value of certain state-action pair than absolute yielding the Advantage function:
A(St, At) = Q(St, At) - V(St)

Since, Q(St, At) = R(St, At) + $\gamma$ * V(St+1)

Simplifying,
A(St, At) = TD Error of V(St) !

Using the advantage function for the Actor update and a state-value prediction Critic network yields a variant- Advantage Actor-Critic (A2C)

TODO: 
- Implementation of robot arm using A2C in panda gym - https://huggingface.co/learn/deep-rl-course/en/unit6/hands-on 
- Further reading:
  - https://spinningup.openai.com/en/latest/spinningup/rl_intro.html?highlight=advantage%20functio#advantage-functions 
  - https://www.youtube.com/watch?v=AKbX1Zvo7r8
  - https://arxiv.org/abs/1602.01783v2 
  
## Proximal-Policy Optimization (PPO)
A single sample transition may not be a good representative of the dynamics of a certain state and hence one would like to ensure conservative policy updates. To this end, we compute the ratio of the current to former policy for a given state-action pair clipped to the range $\in$ (1e, 1+e) for some threshold e (Usually, =0.2) and use that as a drop-in replacement to the log probability in the actor update:

### Clipped Surrogate Objective
Recall Actor update:
$\nabla$ J($\theta$) = E[ $\sum$_t $\nabla$\_$\theta$ log $\pi$\_$\theta$ (at | st) At]

Drop-in replacement of the ratio (clipped),  
r_t ($\theta$) = $\pi$ (at|st) / $\pi$\_former (at|st)

Yields,  
J($\theta$) = E[ $\sum$_t min(r_t($\theta$)At, clip(r_t($\theta$), 1-e, 1+e)At)]

The clipped surrogate objective is an improvement over a previous work, **Trust Region POlicy Optimization (TRPO)** which also uses the ratio but alongside other complicated measures than a clipped objective to ensure monotonic policy improvement (TODO: brief more).

The actual objective of PPO is a combination of the clipped surrogate objective and a sqaured value errror and entropy bonus loss. (TODO: brief more)

**TODO**
- Additional readings:
  - read Towards Delivering a Coherent Self-Contained Explanation of Proximal Policy Optimization” by Daniel Bick, especially part 3.4. 
  - hands on https://huggingface.co/learn/deep-rl-course/en/unit8/hands-on-cleanrl 
  - https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl
  - https://www.youtube.com/watch?v=KjWF8VIMGiY
  - https://openai.com/index/openai-baselines-ppo/
  - https://spinningup.openai.com/en/latest/algorithms/ppo.html
  - https://arxiv.org/abs/1707.06347
  - https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
  - https://www.youtube.com/watch?v=MEt6rrxH8W4
  - https://www.youtube.com/watch?v=C3p2wI4RAi8
  - Sample factory implementaiton for vizdoom- https://huggingface.co/learn/deep-rl-course/en/unit8/hands-on-sf 

## TODO FURTHER
- Advanced topics in RL: https://huggingface.co/learn/deep-rl-course/en/unitbonus3/introduction 