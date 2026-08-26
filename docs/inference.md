# Inference

A console-driven input of observable game state and opponents’ actions. Thank to the trained policy_net, we are gifted its saved weights, which we utilize to guide the hero’s actions.
The inference is so that the DDQN can be applied to real word games, so that a player can actually use it. 
## How It’s Made
It is very similar to the Poker class (the environemnt), but has many structural differences.   
To start, the bet history deque we see in the Play class is a substitute for the contribution logic that we observe with hero_contribution and opp_contribution variables in the Poker class. The deque is used to track bet amounts and to check how much call amount is as well as if the amount owed for each player is the same.
