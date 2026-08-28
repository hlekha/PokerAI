# Environment
If you are not familiar with poker terminology I recommend that you refer to [Poker Knowledge](PokerKnowledge.md).

Utilizing (formerly OpenAI's gym) gymnasium's API, I developed a custom reinforcement learning environment from which my agent can live, commit actions, and record its observations. Through the use of gymnasium's framework, I was able to implement a poker engine, and use it to simulate the heads-up poker game and assist the functionality of gymnasium's reset and step functions. Gymnasium's framework helps to collect data on the current state of the agent, the consequences of the agents actions, and the accompanying total expected reward thereof.  

This environment ensures that the agent can learn from interacting with the environment across tens of thousands of episodes while consistently bounding it by betting mechanics, game progression, state transitions, and reward calculations. This implies the environment is also responsible for updating the game state and enforcing poker rules.

## State and Action Spaces

At its core is an observation/state space of 11 elements, and a discrete action space consisting of 7 elements. We use spaces - a tool from the gymnasium API that help to represent mathematical sets - as the foundation of the environment. 

The state space is used to determine what information the agent has access to and what variables should be updated, tracked, and reset over the course of the agent's training. 

<div align="center">
  
| State Feature | Range | Description |
| -------- | -------- | -------- |
| Hand Equity    | 0-1     | Probability of having the winning hand     |
| Starting Stack    | 0-1     | Initial stack size, normalized from the environment's range for the stack     |
| Pot Size    | 0-1     | Normalized pot size relative to the starting stack     |
| Hero Stack    | 0-1     | Agent's current stack relative to its starting stack     |
| Effective Stack    | 0-1     | The smaller of the hero and opponent stacks, normalized by starting stack     |
| Position    | 0 or 1     | If agent is small or big blind     |
| Pot Odds   | 0-1     | Ratio between amount required to call and the pot + that call amount     |
| Opponent Aggression   | 0-5     | Ratio of opponent raises to calls     |
| Street    | 0-1     | Betting street normalized     |
| Opponent Last Action   | 0-6     | Representation of opponent's actions (same as action space)     |
| Betting Ongoing   | 0 or 1     | Boolean flag of if betting round is active or not     |
  
</div>

The action space is used to represent the actions the agent is allowed to take, with each integer of the set corresponding to a poker action.

<div align="center">
  
| Action | Decision | Description |
| :--- | :--- | :--- |
| 0 | Fold | Concede the game and the pot |
| 1 | Check/Call | Matches the outstanding bet amount (check = 0)|
| 2 | 33% Pot | Calls any outstanding bet and bets/raises 33% if the pot |
| 3 | 67% Pot | Calls any outstanding bet and bets/raises 67% if the pot |
| 4 | 100% Pot | Calls any outstanding bet and bets/raises 100% if the pot |
| 5 | 125% Pot | Calls any outstanding bet and bets/raises 125% if the pot |
| 6 | All-in | Commits the agent's entire stack |

</div>

At first, I thought a continuous action space was best, since the agent can bet any integer amount from 1-amount of current stack, but realized this to be inefficient. The simplification of the action space using a discrete set of pot-relative bet rises leads to simpler - hence easier for the agent - learning; particularly when assigning a Q-value for the action space, if the space were continuous the agent would need to assign a Q-value for every possible chip amount, but with the discrete space the agent only has to estimate the expected value of the seven elements.

## Helper Functions

Embedded in the class are several helper functions, together these functions act as a poker engine which helps simulate the environment for the agent which is essentially the poker game itself. We have the advance_street function which advances the street and resets variables that only live within and are changed in a given street; the compute_win_rate function which calls the Monte Carlo Equity Calculator, gives it the respective information - like street, your hand, and board cards - and updates the instance variable, self.win_rate to the current win rate of the player; the post_blinds function checks the position and street of the player, and respectively posts blinds for the player and opponent; since in Heads-Up Poker the first person to act depends on who has the button, and what street the players are at, the first_to_act function reads the information and returns who should be first to act; the whos_turn function checks what street it is, and calls the first_to_act function to find out who's turn it is in a given street; given the agent's action the done function updates the variables that tracks if the betting round is live or not; the chip_mover function updates stacks and pot depending on the player and chip amount - which are both parameters of this function; the opp function models the opponents actions, and executes the respective consequence using the chip_mover function; the apply_action applies the consequence of the agent's action; the showdown function evaluates the opponent's hand and agent's hand, and compares the two to return the winner; the reward_shaping function uses the pot odd metric, and a scale factor to reinforce good strategy, while penalize poor ones; lastly, the get_reward function rewards the winner the pot. I also have an action mask function designed to illegalize moves that disobey the Heads-Up Poker rules, but, for now, this is just acting as a placeholder for now as the poker engine practically does that job already. 

Additionally, the _get_obs function helps to collect the information you want fed into the neural network. For this project, I've selected the input to be fairly small, containing the player’s win rate, the stack (normalized against a selected range), the pot normalized, the player’s stack (normalized against its starting stack), the player’s effective stack normalized, the player’s position (if they are small or big blind), pot odds, opponents aggression, the normalized street, the opponents action, and if betting is ongoing. This function returns a vector of that information, and feeds it into the DDQN to which to act as the input layer.  


## Gymnasium Functions

Traditionally,  in the gymnasium environment lives the constructor, step, and reset function. The constructor function creates all of the instance variables that we will be tracking across the environment. Furthermore, the step and reset functions are standard functions for a gymnasium environment. The step function’s purpose is to move the agent from its current state to its respective successive state based on the chosen action and call the observation function to observe and record the effect; the reset function resets the variables once the agent reaches the terminal state. 


