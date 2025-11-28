# Deep Q-learning (DQN):

Given the lack of scalability in tabular methods for environments with large state spaces, DQN implements a neural network based function approximation/ parameterization for the Q-values. Specifically, it learns an NN as a mapping from input states(observations) to output Q-values for each admissable action for the corresponding state.

The update equation computes a loss function as the TD error and updates weights of the NN accordingly via gradient descent.

## Challenges of function approximation

### Catastrophic forgetting
A generalized function approximation via neural nets may tend to catastrophically forget learning prior experiences as it learns new ones. A general solution is to collect experiences in a buffer (**experience replay buffer**) and repeat learning across the same experiences multiple times regularly.

### Experience Correlation
Learning from sequential data poses a challenge of high correlation between adjacent samples leading to biased gradient steps. Gradient descent assumes i.i.d samples for proper updates. One leverages the experience replay buffer to randomly sample transitions to learn from uncorelated data.

### The Moving Target Problem
All Q-values are outputs of a function approximation which share the same weights. Hence, every update influences every Q-value which can lead the TD error to never reduce. One then copies a conservative version of the NN and uses the latter to compute the TD target. Updating the copy (called the Target Network) less frequently helps stabilize the TD target.

### Overestimation Bias
While the TD update greedily exploits the current Q-values to find optimal action step, the highest value functions need not correspond to the optimal action (noisy estimate). While this exists in prior tabular methods as well, the problem is amplified under function approximation and the noisy NN updates. One leverages the conservative copy to estimate the TD targets reducing the overestimation. 

**Algorithm**:
Initialize replay buffer D to capacity N
Initialize Q parameterized by theta (random)
Initialize Q_hat parameeterized by theta_hat (= theta) # Target Network
For episodes 1 to M do
    For t=1 until Termination
        with probability epsilon select random action At
        else select At = argmax_a Q(St, a, theta)
        Step At and receieve Rt+1, St+1
        Store transition in replay buffer (St, At, Rt+1, St+1) in D
        Sample minibatch of transitions from D
            y_i = Ri+1 + gamma * max_a Q_hat(Si+1, a, theta_hat)
            y_hat_i = Q(Si+1, Ai, theta)
            L = MSE(y_i, y_hat_i)
            Gradient descent update (theta)
        Every C steps
            Q_hat = Q # Update the copy
    end for
end for



**NOTE**:
- Can handle large state spaces but requires finite action space.
- The implementation of an extra conservative copy is often a variant called **Double DQN**.
- The discussed issues fall into a larger concept called the **Deadly Triad**: Function approximation + Bootstrapping + Off-Policy Learning -> Unstable Learning.
- Does Monte Carlo return instead of TD estimate resolve these issues? Yes (no bootstrapping) but introduces slow learning, higher variance and is inherently on-policy. Despite its deadliness we desire the triad.


**TODO**:
- implement DQN for atari environments maybe (Pong, Seaquest, QBert, Ms Pac Man).
-  Deep RL Abbeel- https://www.youtube.com/watch?v=Psrhxy88zww
- DQN atari paper - https://arxiv.org/abs/1312.5602
-  intro to Double Q learning - https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html 
-  prioritized experience replay- https://arxiv.org/abs/1511.05952
-  duelling architectures - https://arxiv.org/abs/1511.06581 
- more on deadly traid - around ch11 in Sutton.