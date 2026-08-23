# Poker AI


[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%2F%20gymnasium-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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

## Results & Performance
<p align=center>
  <img width="407" height="377" alt="final_reward_matched_resolution" src="https://github.com/user-attachments/assets/35416174-fad4-424c-96b6-e1e68ecafb74" />
</p>

<p align=center>
  <img width="407" height="377" alt="Screenshot 2026-07-26 095831" src="https://github.com/user-attachments/assets/7ab985e2-d0c4-413e-9ef6-e5280d7396c9" />
</p>



## Strategy

### State Space

### Action Space

For more in depth details for each component of this project go to [docs](./docs/).

## Getting Started

### Stack
* **Language**: Python 
* **Reinforcement Learning Tools**: PyTorch and gymnasium 
* **Key Libraries**: NumpPy, treys, matplotlib

### Prerequisites
This code depends on the following libraries to be installed: gymnasium, for the environment building; matplotlib to see a graphical representation of the performance; torch for constructing the neural network; treys for its hand strength calculator, and simulating a deck of cards; as well as numpy for its respective functions.

```bash
pip install -r requirements.txt
```

### Hyperparameters


<div align="center">


| | Batch | $\gamma$ | $\epsilon _0$ | $\epsilon _{final}$ | $\epsilon _{decay}$ | $\tau$ | $\alpha$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Significance** | Cell 1 | Cell 2 | Cell 3 | Cell 4 | Cell 5 | Cell 6 | Cell 6 |
| **Input** | 256 | 0.97 | 0.90 | 0.01 | 113750 | 0.005 | 3e-4 |
</div>

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
