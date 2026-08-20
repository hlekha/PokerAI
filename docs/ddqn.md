# The Double Deep Q-Network



This Double Deep Q-Network consists of an input layer which accepts a 7 dimension observation vector. This vector is passed through two hidden layers consisting of 256 nodes each, where each node applies the ReLU function. The duplicates this network and 



## Why DDQN > DQN





## The Mechanics



This network utilizes a replay buffer which acts as a bank that the agent can deposit and withdraw from. The transactions contain experiences.  For each life (episode) the agent lives it stores the action it took, the state of the game, what the successive state it’s in as a result of that action, and the accompanying reward it gets for reaching that succesive state. Each episode the program collects that data and stores it into this bank for better sample data. The bank is actually just a deque where once the max capacity is met, if an experience is added 

## Result
