# Poker AI


[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%2F%20gymnasium-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align=center>
  <img width="640" height="380" alt="Las Vegas Wtf GIF by Looney Tunes" src="https://github.com/user-attachments/assets/6bb93a09-cda1-479e-a70b-f5730a2118af" />
</p>

_**A heads-up No-Limit Texas Hold'em poker agent trained under a Double Deep Q-Network using a custom gymnasium environment, a Monte Carlo equity calculator, and much more!**_


## Table of Contents

In this README I will talk briefly about:
- [**Overview**](#overview)
- [**Key Features**](#key-features)
- [**The Mechanics**](#the-mechanics) \
. . . . . . . . . . . . . . . [_Architecture_](#architecture) \
. . . . . . . . . . . . . . . [_Monte Carlo Equity Calculator_](#monte-carlo-equity-calculator) \
. . . . . . . . . . . . . . . [_Environment_](#state-space) \
. . . . . . . . . . . . . . . [_DDQN_](#DDQN)
- [**Training**](training) \
. . . . . . . . . . . . . . . [_Hyperparameters_](#hyperparameters) \
. . . . . . . . . . . . . . . [_Reward Function_](#reward-function) 
- [**Results & Performance**](#results--performance)
- [**Current Limitations**](#current-limitations)
- [**Getting Started**](#getting-started)
- [**Future Improvements**](#future-improvements)
- [**License**](#license)
- [**Contact**](#contact)

## Overview
A reinforcement learning poker agent built with PyTorch and Gymnasium that learns heads-up Texas Hold'em decision-making through a Double Deep Q-Network (DDQN). This agent was built to optimize profit and win-rate, while automating the decision-making process given information about the game state. 

This agent utilizes the intersection of reinforcement and deep learning - the Double Deep Q-Network (DDQN), which utilizes Q-learning, and two deep neural networks. This RREADME, as well as other documents in this repo will rely heavily on knowledge of poker terminology, for relevant and foundational dictionary of these terms, you can refer to this [dictionary](./docs/PokerKnowledge.md).

## Key Features

This project utilizes many technical tools, all of which I will go into more detail in other sections, but to briefly summarize:

- **Custom Gymnasium Environment** — Creates an environment for the agent to live in, while modelling action consequences, retrieving  and showdown logic.

- **Double Deep Q-Learning** — Utilizes two neural networks to optimally learn action values from its own simulated experiences.

- **Experience Replay & Target Network** — Uses replay memory and soft target network updates to improve training stability and time.

- **Monte Carlo Simulations** — Uses probability theory and simulations to estimate the win rate given its current hand and board.

- **Poker State Representation** — Incorporates hand equity, pot odds, effective stack, position, opponent aggression, street, and betting state.

- **Discrete Action Space** — Supports folding, checking, calling, multiple bet sizes, and all-in decisions.

- **Poker Engine** — Models poker rules (for the respective poker variation) with betting rounds, variable stack sizes, position, blinds, and dealing.

- **Reward Shaping** — Uses equity and pot odds for calculating intermediate rewards to guide learning.

- **Playable Trained Model** — Saved model weights can be loaded into an interactive interface for real poker decisions.


## The Mechanics  

### Architecture
<p align=center>
  <img width="1471" height="644" alt="architecture" src="https://github.com/user-attachments/assets/f1bbc3d6-942d-47d5-b75d-14ec578ef50d" />
</p>

### Monte Carlo Equity Calculator
Equity in poker regards to the probability that the player's hand will win against the opponent's, given the known board cards. This component of the code estimates this probability using Monte Carlo simulations. My agent needs the equity as this is the agent's strongest signal in figuring out how strong its position in the game is. Since Poker is a game of incomplete information, where the agent doesn't know the opponent's cards nor the future community cards, the estimation of equity can help impute missing information with repeated sampling of the unknown cards; this is done by utilizing Monte Carlo simulations. 

For a single iteration of the MC simulation, the opponent is simulated as well as the remaining pieces of the board, using the treys library, we are able to evaluate the two poker hands against the board. The strengths of the two hands are then compared and if the hero's hand is stronger, the count is incremented. For more information on this component of the code go [here](./docs/mc_calculator.md).

### Environment 

Every RL agent needs an environment to live within, this RL agent uses a custom built environment using (formerly) OpenAI's gymnasium. The environment features a poker engine that helps to enforce the rules of and model a heads-up Texas Hold'em game, as well as the gymnasium standard reset, step, and _get_obs functions. The environment's helper functions (the methods excluding the gymnasium standard functions) manages the game state, card dealing, blinds, stacks, pot size, betting rounds, contributions, positions, and street progression. 

The step function's purpose is to simulate a step if the agent's path to the terminal state - from one state to the next. At each step, the agent selects one of seven discrete actions from fold to check/call to betting different bet sizes, to an all-in. The environment applies the action's consequences to the current game state, and simulates the opponent's reaction; this all aids in shifting the initial state towards the successive state. 

The environment also calculates rewards. This calculation is based on the changes in the agent's chip stack from the initial state to the terminal, and also utilizes reward shaping to reinforce the use of good strategies and aversion of foolish ones. 

The purpose of this environment is to expose raw game information to the neural network. It does this by obtaining an observation vector which collects select information about the game state like hand equity. These observations get passed through the DDQN every episode, and is used for training and optimizes the network. I delve deeper about this crucial component [here](./docs/environment.md).

### DDQN
For a more in depth description of this component of the code go [here](./docs/ddqn.md)


## Training


### Hyperparameters



**Batch**: The amount of experiences that is sampled from the memory bank each training step to compute loss update. _Set to 256_ \
**$\gamma$**: The discount rate. How much the agent values total rewards compared to its immediate rewards. The purpose of this is to make the agent consider long-term consequences of its actions. _Set to 0.97_ \
**$\epsilon _0$**: The exploration rate. It is the probability of the agent choosing a random action over the optimal action. The purpose of this parameter is to let the agent explore for new potentially (more) optimal routes while taking advantage of the current optimal route it knows. _Set to 0.90_ \
**$\epsilon _{final}$**: What epsilon will decay to after training is complete. _Set to 0.01_ \
**$\epsilon _{decay}$**: The decay rate for exploration. It controls how fast epsilon decreases from $\epsilon_0$ to $\epsilon{final}$. _Set to 113750_ \
**$\tau$**: The soft updates. How much of the online network's weights are blended in the target's network at each training step. The value is usually really low so that the target network shifts into the online network smoothly. This avoids oscillations and divergence in training. _Set to 0.005_ \
**$\alpha$:** The learning rate. How much the agent considers new information relative to the existing information. The purpose of this parameter is to choose how quickly the agent adapts to new information. The more information is processes the slower the program will be , the lower the value the more conservative it will be. _Set to 3e-4_
These hyper parameters should be changed to how you want the respective trade offs it inflicts to to be balanced. 

### Reward Function

The reward function is built inside the environment but, since it's a crucial step in the training I will talk about it here. The reward function is made up of "terminal" rewards that the agent receives only when he reaches the terminal state - which in our case is the end of a poker game - and the intermediate rewards which I calculated based on an edge metric and a scale factor of 0.3. The scale factor still needs to be tested for the optimal number, but is used to make the edge metric not too significant where the agent overestimates the value of certain action, but not too inconsequential where the agent underestimates the value. The edge is calculated using the difference from the win rate and the pot odds (the ratio of call amount to the pot plus the call amount). The final reward per game is the cumulation of the intermediate rewards as well as the terminal rewards.

The purpose of the intermediate rewards is to reduce the foresight that the agent needs. Since the probability of winning from the start of the poker game to the end is so volatile, and its final payoff (the accompanying reward for winning) is so distant, the agent needs more signals so it understands more complex patterns of the game. This technique of reward shaping also helps to accelerate training time and sample efficiency. 

## Results & Performance
<p align=center>
  <img width="407" height="377" alt="final_reward_matched_resolution" src="https://github.com/user-attachments/assets/35416174-fad4-424c-96b6-e1e68ecafb74" />
</p>
Unfortunatley, the resolution got botched. After some research, and a query to Claude, this is more because of the resolution of my PC
<p align=center>
  <img width="407" height="377" alt="Screenshot 2026-07-26 095831" src="https://github.com/user-attachments/assets/7ab985e2-d0c4-413e-9ef6-e5280d7396c9" />
</p>


## Getting Started

### Stack
* **Language**: Python 
* **Reinforcement Learning Tools**: PyTorch and gymnasium 
* **Key Libraries**: NumpPy, treys, matplotlib

### Prerequisites
This code depends on the following libraries to be installed: **gymnasium**, for the environment building; **matplotlib** to see a graphical representation of the performance; **PyTorch** for constructing the neural network; **treys** for its hand strength calculator, and simulating a deck of cards; as well as **numpy** for its respective functions. You must also make sure that you have Python 3 and pip installed.


### Installation

1. Clone the repo
```bash
git clone https://github.com/hlekha/PokerAI.git
cd PokerAI
```
2. Create a virtual environment
_For Windows_:
```bash
python -m venv .venv
.venv\Scripts\activate
```
_For macOS/Linux_:
```Zsh
python3 -m venv .venv
source .venv/bin/activate
```
3. Install the dependencies
```bash
pip install -r requirements.txt
```

### How to Train
To begin training the DDQN agent, run:
```bash
python src/DDDQN_Training.py
```
Once training is complete the policy network weights are saved and can be loaded later for inference.

If you want to run the trained agent then run the following script:
```bash
python src/inference.py
```
This will allow you to prompt the user about information regarding the game, and will continue its prompts until the user ends it. Until then, after the information is sent to the agent, the agent prints out the optimal action for the given scenario.

## Current Limitations

Although the agent is capable of learning poker actions via reinforcement learning, there are some restrictions in place in the current implementation that influence the strategies in an unfavorable way.

### Single-Hand Optimization

Currently, due to the terminal state being the end of a single poker hand, each individual poker hand is optimized separately. This means that the agent's objective is the maximization of its expected reward for an individual hand rather than maximization of long-term profitability.

One side effect that was observed during training was the agent's tendency to employ very aggressive plays, including all-ins. Even though the agent receives positive expected reward in the current environment by doing so, it does not learn about the long-term effects of taking too many risks since chip preservation and future hands are not included in the agent's objective.


### Simplified Opponent Model

Since the current training opponent selects the actions it wants to take randomly, the agent is more geared towards exploiting a stochastic opponent than to playing against a realistic poker opponent.

Furthermore, the state feature, opponent aggression, lose their meaning since the opponent does not have any consistent behavior. Training against multiple opponent strategies or previous versions of the agent could add variety into the learning process.

### Uniform Opponent Hand Range

The equity calculator uses the assumption that the opponent's unknown hole cards are uniformly distributed between the cards that remain in the deck.

However, in real poker, the player's actions provide valuable information about their hand. For instance, a raise can increase the likelihood of some strong hand significantly more than it was beforehand. In the current environment, this information is disregarded.

Future versions of the environment could estimate the ranges of opponent's hole cards from their betting and use these ranges in their Monte Carlo simulations.



## Future Improvements
As talked about in single-hand optimization, the agent exhibits a high bias towards choosing the all-in action. Since each episode simulates a single hand, during training the agent is only alive for one hand per episode; the agent's objective is to maximize expected reward for a given episode, the agent believes that it only has one hand to play to maximize its stack. As a result, the agent decides that betting its entire stack is the best way to do optimize. It neglects the variance and accepts more risk, adopting an all-or-nothing mentality.

This reckless behavior signals that there is a mismatch between the optimization objective and how poker is successfully played. While observing how poker is played professionally, I noticed players rarely go all-in and this is because going all-in introduces high variance. To minimize this professional players make decisions with long term success in mind - they optimize their risk-adjusted return with smaller bets to preserve their stack, and regulate risk management. As they experience more games, then they start deploying more aggressive strategies based on whatever mathematical model of the game they have in their mind. Conversely, due to the environment design, the agent has no concept of long term survival, thus it is unable to be conservative of its stack, mitigate its variance, and regulate the risk it accepts. As a result, it learns to deploy an "all-in all-the-time strategy," and while this aggressive strategy achieves a win rate of [enter metric] - and produces quick, massive profit when successful - it also leads to the agent overcommitting on hands of poor equity. 

However, there exists a flaw with even the professional players' strategy. The patient "loading stage" that they commit to in early game, only lasts until they download the data on their opponent's playstyle, then they issue their attack; the downside of this is due to the blinds you must sacrifice each game, depending on the length of this loading stage, it can be quite costly. Luckily, the advantage of reinforcement learning in this project is that the training happens in a simulated environment thousands of times before the inference commences - effectively skipping the costly loading stage and skipping straight to the exploitation while your opponent is stuck learning.

I plan to improve this design by reframing the reward function so that the agent can adapt to a stronger long-term profitability rather than maximizing the reward of a single hand. I would implement this by introducing a metric to measure variance of the reward per hand, and penalties for high-variance actions under low win rate conditions. I also could redesign the environment so that an episode reflects consecutive hands rather than just one. I hypothesize that these changes would enact the agent to reserve its all-in actions for ideal conditions.

Although these changes would likely inflict a lower average profit per single hand, the agent will react with stable return, longer gameplay (more fun), better risk tolerance, and improved decision-making.

## License
Distributed under the Apache-2.0 License. See [LICENSE](LICENSE) for more information.

## Contact 
My LinkedIn: https://linkedin.com/in/hayden-lekha \
Project Link: https://github.com/hlekha/PokerAI
