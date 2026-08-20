# The Double Deep Q-Network

This agent is trained under a Double Deep Q-Network which consists of an input layer which accepts a 7 dimension observation vector and is set against a random rule-based opponent. The DDQN uses the the custom gymnasium environment (Poker) to let the agent line in and play against, and the data from training this network is then passed through the inference class (Play) to use in real time.


## Why DDQN > DQN

The vanilla DQN's uses a single network the both picks and evaluates the action that the agent selects. During training, Q-values are estimated, with these estimates being noisy throughout the episodes, so when the $\epsilon$-greedy policy selects a greedy action, the value is an overestimated because of that noise since the max operation biases higher values. The overestimation then compounds across the steps.

The DDQN fixes this problem by separating the action selection from the action evaluation. It creates two networks, policy_net which picks the action with the $\epsilon$-greedy policy and the target_net network which evaluates it to the target value. In particular, during training the actions that reap high benefits like all-in and higher bet sizes, will be grossly overestimated because the DQN factors rare high value actions, regardless of the risk associated with it. With the DDQN implementation, it keeps the policy from prematurley converging to this and avoids the constant risky actions.


## The Mechanics for Training

### Hyperparameters


#### Epsilon 

The exploration rate. It is the probability of the agent choosing a random action over the optimal action. The purpose of this parameter is to let the agent explore for new potentially (more) optimal routes while taking advantage of the current optimal route it knows. We set this to be (initially) 0.9.

#### Alpha

The learning rate. How much the agent considers new information relative to the existing information. The purpose of this parameter is to choose how quickly the agent adapts to new information. The more information is processes the slower the program will be , the lower the value the more conservative it will be. This is also just 1 - \epsilon . We set this to be (initially) 0.1.

#### Gamma

The discount rate. How much the agent values total rewards compared to its immediate rewards. The purpose of this is to make the agent consider long-term consequences of its actions. We set this to be 0.97

#### Episodes
The number of training episodes. The purpose of this is to set a number of times the agent iterates through the environment, from start to terminal. We set this to be 87,500.

#### Tau

The soft updates. How much of the online network's weights are blended in the target's network at each training step. The value is usually really low so that the target network shifts into the online network smoothly. This avoids oscillations and divergence in training. We set this to be 0.005

### Replay Buffer
This network utilizes a replay buffer which acts as a bank that the agent can deposit and withdraw from, where The transactions are experiences from initial state to terminal state.  For each life (episode) the agent lives it stores the action it took, the state of the game, what the successive state it’s in as a result of that action, and the accompanying reward it gets for reaching that successive state. Each episode the program collects that data and stores it into this bank for better sample data. The bank is actually just a deque where once the max capacity is met, if an experience is added the oldest entry into the deque is deleted. The max length for my replay buffer was 275,000 experiences; the DDQN samples this bank during training with a uniform distribution.

### Epsilon-Greedy
This network also uses the $\epsilon$-greedy policy. This policy utilizes the epsilon hyperparameter, which represents the ratio of learning that the agent is allowed to do relative to the amount of exploiting it must do. So with our initial epsilon of 0.9 the first thousand or so episodes, the agent will learn 90% of the time. This doesn't last forever though, since we want our agent to exploit as well, we establish a decay rate for epsilon where the parameter starts at some value but over time decreases at an exponential rate (we have an exponential decay which is set at 113750). The way we choose if the agent exploits or learns though, is through a uniform random sample of a rational [0,1] - if that sample is less than (or equal to) epsilon then the agent will learn by taking a random action, but if it is greater the agent will take the action that reaps the greatest q-value.

### AdamW Method

### Huber Loss Function

## Result
