# Poker AI

## Purpose
Aimed at perfecting the performance of the game Poker, this agent was built to optimize profit and win-rate, while automating the decision-making process given information about the game state. Specifically, this agent is tailored for the heads-up variation of Poker, which acts just like a Texas and Hold 'em game, but is strictly two players - where the small blind plays first only at the preflop stage. 
This agent utilizes the intersection of reinforcement and deep learning - the Double Deep Q-Network (DDQN), which utilizes Q-learning, and two deep neural networks. For more insight about this hop over to the [Key Features (##key-features)] section.

## Prerequisites
This code depends on the following libraries to be installed: gymnasium, for the environment building; matplotlib to see a graphical representation of the performance; torch for constructing the neural network; treys for its hand strength calculator, and simulating a deck of cards; as well as numpy for its respective functions.
```shell
pip install gymnasium
pip install numpy
pip install matplotlib
pip install torch
pip install treys
```

## Usage
Running the program will automatically prompt you to enter the necessary details like the stack amount, the cards in your hand, and the blind amount. From there the hand will enter a loop of recieving information about the board cards, and opponents action, and outputting the respective optimal action. By the end up of it you'll be filthy rich, and will win so much, your opponents will suspect you of cheating.  
## Key Features

## Key Features
The DDQN is an extremely impressive, and intuituve method of building an agent, it is much like training a pet. This neural network esentially solves the Bellman Optimality Equation by instead utilizing methods such as gradient descent, experience replay, an epsilon-greedy policy, and a second neural network that seperates the agent from evaluating an action, and actually selecting it. Paired with a custom environment, as well as a Monte-Carlo hand-equity calculator that calculates the equity of your hand by utilizing Monte Carlo simulations.

## License
Distributed under the Apache-2.0 License. See LICENSE for more information.

## Contact 
My LinkedIn: linkedin.com/in/hayden-lekha
My Email: haydenlekha@gmail.com
Project Link: https://github.com/hlekha/PokerAI
