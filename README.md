# Poker AI

<p align=center>
  <img width="640" height="380" alt="Las Vegas Wtf GIF by Looney Tunes" src="https://github.com/user-attachments/assets/6bb93a09-cda1-479e-a70b-f5730a2118af" />
</p>

## Overview
Aimed at perfecting the performance of the game Poker, this agent was built to optimize profit and win-rate, while automating the decision-making process given information about the game state. Specifically, this agent is tailored for the heads-up variation of Poker, which acts just like a Texas and Hold'em game, but is strictly two players - where the small blind plays first only at the preflop stage. 
This agent utilizes the intersection of reinforcement and deep learning - the Double Deep Q-Network (DDQN), which utilizes Q-learning, and two deep neural networks. For more insight about this hop over to the [Key Features (##key-features)] section.



## Architecture 
<p align=center>
  <img width="1471" height="644" alt="architecture" src="https://github.com/user-attachments/assets/f1bbc3d6-942d-47d5-b75d-14ec578ef50d" />
</p>

## Prerequisites
This code depends on the following libraries to be installed: gymnasium, for the environment building; matplotlib to see a graphical representation of the performance; torch for constructing the neural network; treys for its hand strength calculator, and simulating a deck of cards; as well as numpy for its respective functions.
```shell
pip install gymnasium
pip install numpy
pip install matplotlib
pip install torch
pip install treys
```

## Key Features
The DDQN is an extremely impressive, and intuitive method of building an agent, it is much like training a pet. This neural network essentially solves the Bellman Optimality Equation by instead utilizing methods such as gradient descent, experience replay, an epsilon-greedy policy, and a second neural network that separates the agent from evaluating an action, and actually selecting it. Paired with a custom environment, as well as a Monte-Carlo hand-equity calculator that calculates the equity of your hand by utilizing Monte Carlo simulations.

## Usage
Running the program will automatically prompt you to enter the necessary details like the stack amount, the cards in your hand, and the blind amount. From there the hand will enter a loop of receiving information about the board cards, and opponent actions, and outputting the respective optimal action. By the end up of it you'll be filthy rich, and will win so much, your opponents will suspect you of cheating. 
<p align=center>
  <img width="400" height="362" alt="Screenshot 2026-07-26 095709" src="https://github.com/user-attachments/assets/2fa69368-d69b-498a-8277-c6aaedcdeb05" />
</p>
Some things you can adjust in the code to change how you want the training of the agent to alter are the hyperparameters. Currently they are set at:
```python
BATCH = 256
GAMMA = 0.97
EPS_0 = 0.9
EPS_FINAL = 0.01
EPS_DECAY = 300
TAU = 0.005
ALPHA = 3e-4
```
talk about what cahnging each hyperparameter does
talk about chjanging episode amount and the calcualtion for eps decay
talk about how  aplot demonstrating the loss and reward function are updated live
<p align=center>
  <img width="407" height="377" alt="Screenshot 2026-07-26 095831" src="https://github.com/user-attachments/assets/7ab985e2-d0c4-413e-9ef6-e5280d7396c9" />
</p>

## Future Improvements
Currently the agent exhibits a high bias towards choosing the all-in action. Since each episode simulates a single hand, during training the agent is only alive for one hand per episode; the agent's objective is to maximize expected reward for a given episode, the agent believes that it only has one hand to play to maximize its stack. As a result, the agent decides that betting its entire stack is the best way to do optimize. It neglects the variance and accepts more risk, adopting an all-or-nothing mentality.

This reckless behavior signals that there is a mismatch between the optimization objective and how poker is successfully played. While observing how poker is played professionally, I noticed players rarely go all-in and this is because going all-in introduces high variance. To minimize this professional players make decisions with long term success in mind - they optimize their risk-adjusted return with smaller bets to preserve their stack, and regulate risk management. As they experience more games, then they start deploying more aggressive strategies based on whatever mathematical model of the game they have in their mind. Conversely, due to the environment design, the agent has no concept of long term survival, thus it is unable to be conservative of its stack, mitigate its variance, and regulate the risk it accepts. As a result, it learns to deploy an "all-in all-the-time strategy," and while this aggressive strategy achieves a win rate of [enter metric] - and produces quick, massive profit when successful - it also leads to the agent overcommitting on hands of poor equity. 

However, there exists a flaw with even the professional players' strategy. The patient "loading stage" that they commit to in early game, only lasts until they download the data on their opponent's playstyle, then they issue their attack; the downside of this is due to the blinds you must sacrifice each game, depending on the length of this loading stage, it can be quite costly. Luckily, the advantage of reinforcement learning in this project is that the training happens in a simulated environment thousands of times before the inference commences - effectively skipping the costly loading stage and skipping straight to the exploitation while your opponent is stuck learning.

I plan to improve this design by reframing the reward function so that the agent can adapt to a stronger long-term profitability rather than maximizing the reward of a single hand. I would implement this by introducing a metric to measure variance of the reward per hand, and penalties for high-variance actions under low win rate conditions. I also could redesign the environment so that an episode reflects consecutive hands rather than just one. I hypothesize that these changes would enact the agent to reserve its all-in actions for ideal conditions.

Although these changes would likely inflict a lower average profit per single hand, the agent will react with stable return, longer gameplay (more fun), better risk tolerance, and improved decision-making.

## License
Distributed under the Apache-2.0 License. See LICENSE for more information.

## Contact 
My LinkedIn: https://linkedin.com/in/hayden-lekha

My Email: haydenlekha@gmail.com

Project Link: https://github.com/hlekha/PokerAI
