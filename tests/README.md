# Tests

For a while the performance of the agent was poor, and as a result, the custom environment and Double Deep Q-Network had to go through a couple iterations of troubleshooting, implementing a fix, testing it, and recording its effect using an additional plot of how the loss behaves over the steps its function takes.

## The Problem

The problem for the reward collapse dillema was that the reward would eventually collapse to 0. Despite its initial success with reward reaching over 40%, and reward rarely being negative, the reward over episodes scatter plot appeared as if the epsilon-greedy policy was working with opposite logic; in the beginning we would have high-reward episodes with some reaching over 40% (of hand), but then as we'd approach the final episode we'd see a gradual stabilization of the rolling average - converging to zero - with the episodes densely populating the lower quartile of rewards. This decline in progress was iteratively troubleshooted to produce the following tests.
<p align=center>
  <img width="567" height="471" alt="dqn_preddqn logicalbugfixing" src="https://github.com/user-attachments/assets/5a78eda4-1f9f-45b9-a2f8-71bbdb779c7b" />
</p>

### Reward Collapse Test 1
To begin with the troubleshooting, I needed to point out exactly what was causing the problem, therefore I tracked the performance of the loss function with plot_loss to observe how the loss behaves over the steps that its function takes. I then implemented it so that at the end of training the loss and reward could be plotted, and their performance tracked over the course of the training period. For the sake of the tests, since the training period for the main program was set at 87500 episodes which took about eight hours, I decreased the numver of episodes, and changed the decay rate at a respective ratio to the agent's official training. Once the plots were finished and printed, I observed that the rewards were scaled by a factor of a staggering 10^13 - which is definitely impossible considering the maximum that the stacks are allowed to start out at is 500. With this newfound data, I came to the conclusion that something in the code was inflating the reward (and therefore the loss) by a large factor.  
<p align=center>
  <img width="583" height="475" alt="first_iterationof_DDQN" src="https://github.com/user-attachments/assets/e06c815e-5653-47d5-ae17-46b976045bb5" />
</p>

### Reward Collapse Test 2
The next step was to analyze the custom environment and the neural network part of the code to see if I could spot anything tampering with the reward. Soon after, I found a problem residing in the environment - the apply_action function and the opponent model. The problem was that betting was allowed to exceed the stack amount of the respective player. In other words, there was no flag to check if the bet amount was more than the stack. 
After fixing this the following graphs were printed:
<p align=center>
  <img width="583" height="475" alt="second_iterationof_DDQN" src="https://github.com/user-attachments/assets/b333a8c7-1c1e-45a1-9ba3-d627fc982ea5" />
</p>
Unfortunately, when comparing these with the base vanilla graphs seen in test 1, the difference is negligible. This means that the effect of the problem found in this test wasn't significant in training - likely because a bet greater than the player's stack is rare. To conclude, although the bug was not impactful, it helps narrow down the real culprit of the reward collapse dillema.

### Reward Collapse Test 3
