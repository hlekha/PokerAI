# Poker AI


[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%2F%20gymnasium-orange.svg)]()
[![License](https://img.shields.io/badge/License-Apache2.0-green.svg)](LICENSE)
<div align="center">
  
  <img width="640" height="380" alt="Las Vegas Wtf GIF by Looney Tunes" src="https://github.com/user-attachments/assets/6bb93a09-cda1-479e-a70b-f5730a2118af" />

  _**A reinforcement learning agent designed to optimize the performance of heads-up no-limit Texas Hold'em Poker.**_
  
</div>




## Table of Contents

- [**Overview**](#overview)
- [**Key Features**](#key-features)
- [**Repository Structure**](#repository-structure)
- [**The Mechanics**](#the-mechanics) 
  - [_Architecture_](#architecture) 
  - [_Monte Carlo Equity Calculator_](#monte-carlo-equity-calculator) 
  - [_Environment_](#environment) 
  - [_DDQN_](#ddqn) 
  - [_Inference_](#inference)
- [**Training**](#training) 
  - [_Hyperparameters_](#hyperparameters) 
  - [_Reward Function_](#reward-function) 
- [**Results & Performance**](#results--performance)
- [**Current Limitations**](#current-limitations)
- [**Getting Started**](#getting-started)
- [**Future Improvements**](#future-improvements)
- [**License**](#license)
- [**Contact**](#contact)

## Overview
A reinforcement learning agent built with PyTorch and Gymnasium, designed for heads-up Texas Hold'em Poker. It learns decision-making through a Double Deep Q-Network (DDQN), and was built to optimize profit and win rate while automating the decision-making process given the current game state. 

This agent uses a Double Deep Q-Network which combines Q-learning, and  deep neural networks. This README, as well as other documents in this repo relies heavily on knowledge of poker terminology. For relevant and foundational dictionary of these terms, you can refer to the [dictionary](./docs/PokerKnowledge.md).


## Key Features

This project utilizes many technical tools, all of which I will go into more detail in other sections. Nevertheless, here is a brief summary:

- **Custom Gymnasium Environment** — Creates an environment for the agent to live in, while modelling action consequences, retrieving state information, and handling showdown logic.

- **Double Deep Q-Learning** — Utilizes two neural networks to learn action values from its own simulated experiences.

- **Experience Replay & Target Network** — Uses replay memory and soft target network updates to improve training stability and sample efficiency.

- **Monte Carlo Simulations** — Uses probability theory and simulations to estimate the win rate given its current hand and board.

- **Poker State Representation** — Incorporates hand equity, pot odds, effective stack, position, opponent aggression, street, and betting state.

- **Discrete Action Space** — Supports folding, checking, calling, multiple bet sizes, and all-in decisions.

- **Poker Engine** — Models poker rules (for the respective poker variation) with betting rounds, variable stack sizes, position, blinds, and dealing.

- **Reward Shaping** — Uses equity and pot odds to calculate intermediate rewards to guide learning.

- **Playable Trained Model** — Saved model weights can be loaded into an interactive interface for poker decision making.


## Repository Structure
```
PokerAI/
├── docs/                          — Deeper insights on specific components
│   ├── PokerKnowledge.md          
│   ├── ddqn.md                    
│   ├── environment.md             
│   ├── inference.md               
│   └── mc_calculator.md           
│
├── notebook/
│   └── PokerAI.ipynb              — Entire notebook
│
├── src/                    
│   ├── DDQN_Training.py           — DDQN establishment and training thereof
│   ├── environment.py             — Code for the environment
│   ├── inference.py               — Code for the inference
│   └── mc_equity_calc.py          — Code for the Monte Carlo equity calculator 
│
├── tests/                         — Experimentation on certain aspects of code, and for troubleshooting logical errors
│   ├── README.md                  
│   ├── RewardCollapse_Test1.py    — First iteration of solution to problem
│   ├── RewardCollapse_Test2.py    — Second iteration
│   └── RewardCollapse_Test3.py    — Third iteration
│    
├── LICENSE
├── README.md
└── requirements.txt
```


## The Mechanics  

### Architecture
<p align=center>
  <img width="1471" height="644" alt="architecture" src="https://github.com/user-attachments/assets/f1bbc3d6-942d-47d5-b75d-14ec578ef50d" />
</p>

### Monte Carlo Equity Calculator
Equity in poker refers to the probability that a player's hand will win against an opponent's hand given the known board cards. This component estimates that probability using a Monte Carlo simulation. The agent relies on equity as one of its primary signals for determining the strength of its position in the game. Since poker is a game of incomplete information, where the agent knows neither the opponent's cards nor the future community cards, the estimation of equity can help infer missing information with repeated sampling of unknown cards via a Monte Carlo simulation. 

For a single iteration of the MC simulation, the opponent is simulated as well as the remaining pieces of the board. Using the Treys library, we are able to evaluate the two poker hands against the board. The strengths of the two hands are then compared and if the hero's hand is stronger, the count is incremented. For more information on this component of the code go [here](./docs/mc_calculator.md).

### Environment 

Every RL agent needs an environment to live within, this RL agent uses a custom-built environment based on Farama Foundation's (formerly OpenAI's) Gymnasium. The environment features a poker engine that helps to enforce the rules of and model a heads-up Texas Hold'em game, as well as the Gymnasium standard reset, step, and _get_obs functions. The environment's helper functions (the methods excluding the Gymnasium standard functions) manage the game state, card dealing, blinds, stacks, pot size, betting rounds, contributions, positions, and street progression. 

The step function's purpose is to simulate a step along the agent's path to the terminal state - from one state to the next. At each step, the agent selects one of seven discrete actions from fold to check/call to betting different bet sizes, to an all-in. The environment applies the action's consequences to the current game state, and simulates the opponent's reaction; this all aids in shifting the initial state towards the successive state. 

The environment also calculates rewards. This calculation is based on the changes in the agent's chip stack from the initial state to the terminal, and also utilizes reward shaping to reinforce the use of good strategies and aversion of poor ones. 

The purpose of this environment is to expose raw game information to the neural network. It does this by obtaining an observation vector which collects select information about the game state like hand equity. This vector is passed through the DDQN every episode, and is used for training. I delve deeper about this crucial component [here](./docs/environment.md).

### DDQN
The agent learns its poker strategies through a Double Deep Q-Network, where the network is designed to approximate an action-value function, Q(s,a). This function is equal to the expected future reward of taking action, a, from the current state, s.

We use the two Deep Q-Networks, where one is assigned to select the best action - while the other evaluates it. We use two instead of one so that we can avoid overestimating our Q-value. 

The network takes in the 11-dimensional observation vector, goes through a linear combination of weights and biases to two hidden layers of 256 nodes, one of 128 nodes, each applying the ReLU function to their nodes, and finally the output layer is the 7 Q-Values for the respective action space.

For a more in-depth description of this component of the code, including some of the math behind it, go [here](./docs/ddqn.md).

### Inference
The purpose for this component is to test the agent's ability to make optimal decisions and observe if it can handle out of sample data well, while returning similar performance we saw in the training. After training, the policy network weights are saved and loaded into this component so that the agent's learned knowledge can be used independently of the training environment.

The framework of the inference component is  similar to that of the environment, where they both call the Monte Carlo Equity Calculator, use a similar poker engine, and create an 11-Dimensional observation vector. After the observation vector is made, it is fed to the trained policy network which assigns a Q-Value for each action; the max argument is then taken, and translated to the respective poker action.

In inference, the value function becomes deterministic rather than using the epsilon-greedy policy. Hence, the agent does not require exploration, and just chooses the "greedy" action - that is the action with the greatest Q-value. 

This component also provides interactive prompting to gather information describing the current state. The information aids in establishing the environment's numerical observation and consequently the observation vector. After the prompting is finished, all necessary game-state information has been collected, the vector is created and passed through the network's weights, and the recommended action is identified and printed.

For more information about this component, go [here](./docs/inference.md)

## Training


### Hyperparameters



**Batch**: The amount of experiences that is sampled from the memory bank each training step to compute loss update. _Set to 256_ 

**$\gamma$**: The discount rate. How much the agent values total rewards compared to its immediate rewards. The purpose of this is to make the agent consider long-term consequences of its actions. _Set to 0.97_ 

**$\epsilon _0$**: The exploration rate. It is the probability of the agent choosing a random action over the optimal action. The purpose of this parameter is to let the agent explore for new potentially (more) optimal routes while taking advantage of the current optimal route it knows. _Set to 0.90_ 

**$\epsilon _{final}$**: What epsilon will decay to after training is complete. _Set to 0.01_ 

**$\epsilon _{decay}$**: The decay rate for exploration. It controls how fast epsilon decreases from $\epsilon_0$ to $\epsilon{final}$. _Set to 113750_ 

**$\tau$**: The soft updates. How much of the online network's weights are blended in the target's network at each training step. The value is usually really low so that the target network shifts into the online network smoothly. This avoids oscillations and divergence in training. _Set to 0.005_ 

**$\alpha$:** The learning rate. How much the agent considers new information relative to the existing information. The purpose of this parameter is to choose how quickly the agent adapts to new information. The more information is processes the slower the program will be , the lower the value the more conservative it will be. _Set to 3e-4_


These hyperparameters represent the current training setup and should be adjusted to how you want to balance the exploration, stabliilty, and speed of the agent.

### Reward Function

The reward function is built inside the environment but, since it's a crucial step in the training I will talk about it here. The reward function is made up of terminal rewards that the agent receives only when it reaches a terminal state - which in our case is the end of a poker game - and the intermediate rewards which I calculated based on an edge metric and a scale factor of 0.3. The scale factor still needs to be tested for the optimal number, but is used to make the edge metric not too significant where the agent overestimates the value of certain action, but not too inconsequential where the agent underestimates the value. The edge is calculated as the difference from the win rate and the pot odds (the ratio of call amount to the pot plus the call amount). The final reward per game is the sum of the intermediate rewards and the terminal rewards.

The purpose of the intermediate rewards is to reduce the foresight that the agent needs. Since the probability of winning from the start of the poker game to the end is so volatile, and its final payoff (the accompanying reward for winning) is so distant, the agent needs more signals so it understands more complex patterns of the game. This technique of reward shaping also helps to accelerate training time and sample efficiency. 

## Results & Performance
<p align=center>
  <img width="407" height="377" alt="final_reward_matched_resolution" src="https://github.com/user-attachments/assets/35416174-fad4-424c-96b6-e1e68ecafb74" />
</p>

_Disclaimer: Unfortunately, the training plot was generated at a lower resolution. I've implemented the fix, but because the training is so long, I'm not allowed to run it again._

Reward has a relationship to episodes similar to that of a saturation curve, where it initially grows upwards fast, but as it grows its growth rate slows down leading to a convergence. In our graph, we see that same relationship, growing initially fast but then slowing down, where the rolling average stabilizes near 0.75.

This number may seem small, but recall that this number is actually the reward represented as a percentage of the initial stack. We can also observe that in the beginning of training the rewards are widely dispersed, with its rolling mean being very volatile. In contrast, as we see the episode count near its end, that volatility settles down, and the data points less scattered - bunching up mostly around 1.0 (100%).



<p align=center>
  <img width="407" height="377" alt="Screenshot 2026-07-26 095831" src="https://github.com/user-attachments/assets/7ab985e2-d0c4-413e-9ef6-e5280d7396c9" />
</p>
Here we see the performance of the loss function over the course of the 87500 episodes. The loss sharply reaches to about 0.028 for a short amount of time in the beginning of the training, but then rapidly stabilizes to around 0.012-0.014. Although the function remains volatile for the remainder of the training, with no convergence, just bouncing around in the interval [0.012, 0.014], the number is still very marginal as it shows that the difference between the estimated and actual reward remains close to zero for the entire duration of the training. 

## Current Limitations

Although the agent is capable of learning optimal poker actions via reinforcement learning, there are some restrictions in place in the current implementation that influence the strategies in an unfavorable way.

### Single-Hand Optimization

Currently, due to the terminal state being the end of a single poker hand, each individual poker hand is optimized separately. This means that the agent's objective is the maximization of its expected reward for an individual hand rather than maximization of long-term profitability.

One side effect that was observed during training was the agent's tendency to employ very aggressive plays, including all-ins. Even though the agent receives positive expected reward in the current environment by doing so, it does not learn about the long-term effects of taking too many risks since chip preservation and future hands are not included in the agent's objective.


### Simplified Opponent Model

Since the current training opponent selects the actions it wants to take randomly, the agent is more geared towards exploiting a stochastic opponent than to playing against a realistic poker opponent.

Furthermore, the state feature, opponent aggression, lose their meaning since the opponent does not have any consistent behavior. Training against multiple opponent strategies or previous versions of the agent could add variety into the learning process.

### Uniform Opponent Hand Range

The equity calculator uses the assumption that the opponent's unknown hole cards are uniformly distributed between the remaining cards.

However, in real poker, the player's actions provide valuable information about their hand. For instance, a raise can increase the likelihood of strong hands significantly more than it was beforehand. In the current environment, this information is disregarded.


## Getting Started

### Stack
* **Language**: Python 
* **Reinforcement Learning Tools**: PyTorch and Gymnasium 
* **Key Libraries**: NumPy, treys, matplotlib

### Prerequisites
This code relies on the following libraries to be installed: **Gymnasium**, for the environment building; **Matplotlib** to see a graphical representation of the performance; **PyTorch** for constructing the neural network; **Treys** for its hand strength calculator, and simulating a deck of cards; as well as **NumPy** for its respective functions. You must also make sure that you have Python 3 and pip installed.


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
python src/DDQN_Training.py
```
Once training is complete, the policy network weights are saved and can be loaded later for inference.

If you want to run the trained agent then run the following script:
```bash
python src/inference.py
```
This will allow you to prompt the user about information regarding the game, and will continue its prompts until the user ends it. Until then, after the information is sent to the agent, the agent prints out the optimal action for the given scenario.



## Future Improvements

There are several implementations I have planned to make the agent's strategy closer to perfection, and improve its long-term poker performance.

As discussed in Current Limitations, the current agent exhibits a strong bias toward all-in actions. A primary future improvement is therefore to better align the training objective with long-term poker performance rather than maximizing expected reward within a single hand.

The most direct approach would be to redesign the environment so that each episode consists of multiple consecutive hands. This would extend the agent's decision horizon and allow the consequences of stack preservation and repeated risk-taking to influence future rewards.

Another approach worth experimenting with is implementing risk sensitivity into the reward function. This would assess the variance and penalize high-variance decisions under unfavorable conditions. However, this would require thorough testing to ensure that reward shaping does not unintentionally discourage profitable aggression.

My hypothesis is that extending the training horizon will reduce the agent's tendency to select all-in actions, reserving them for situations in which the expected return justifies the associated risk. Future testing of this could compare the resulting policy's profitability, action distribution, and return variance against the current single-hand model.

We must also consider the problem of the opponent hand range being uniformly distributed, and the agent's current lack of ability accounting for the information given by the opponent based on their play style. Future versions of the environment could estimate the range of opponent's hole cards from their betting and use these ranges in their Monte Carlo simulation. This could also help to fix the simple opponent-model.

Although the performance of the agent is good, the opponent model which it battles during training is relatively simple. To solve this problem, some popular strategies professional players use can be coded, or a copy of our current agent can be modified and train them against each other. 

Lastly, I'd like to include more statistics into the project, especially for the Monte Carlo simulation. Although the math is sufficient enough to trust the component itself, these metrics will provide valuable insights to further tune and improve the model and convert it into presentable data. Firstly, I'd like to track action selection of the agent; this will allow me to observe how the action selection evolves as the epsilon-greedy policy develops along with any biases. I also plan to include a plot of the probability throughout k iterations of a given MC simulation, and test it for each street for various hand strengths. This will allow me to see how the probability converges, and prove if the n trials I derived for the Law of Large Numbers to be sufficient is true. Furthermore, I'd like to quantify the "good enough" statement for the n trials using a t-distribution and assessing the p-value.

These future improvements may not be entirely necessary, but it will lead to the agent acquiring more powerful techniques, which in return,  will make my agent even more successful and profitable.

## License
Distributed under the Apache-2.0 License. See [LICENSE](LICENSE) for more information.

## Contact 
My LinkedIn: https://linkedin.com/in/hayden-lekha 

Project Link: https://github.com/hlekha/PokerAI
