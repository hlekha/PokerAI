# Inference

A console-driven input of observable game state and opponents’ actions. Thank to the trained policy_net, we are gifted its saved weights, which we utilize to guide the hero’s actions.
The inference's purpose is making the DDQN applicable to real word games, so that a player can actually use it. 

## How It’s Made

It is very similar to the Poker class (the environment), but has many structural differences.   
To start, the bet history deque we see in the Play class is a substitute for the contribution logic that we observe with hero_contribution and opp_contribution variables in the Poker class. The deque is used to track bet amounts and to check how much call amount is as well as if the amount owed for each player is the same.

Furthermore, instead of an rule-based opponent model that we see in the environment, we have what_opp() which parses text input from the user telling us the opponent's action, and converts the string to an integer. This integer maps to the same values as the action seen in our action space (0-6). If the action was a bet, then the user is also prompted to communicate with the agent the amount, where the agent then filters that amount into the bet-size buckets we see in the action space; transforming its raw dollar amount to a fraction of the pot. We also get direct input to get info about the community cards, where a simulated deck is made to copy the framework of the game, and the community cards are then drawn and removed from said deck. 

Lastly, showdown changes from a simulated opponent's hand randomly chosen from the deck and the comparison thereof to the hero's hand, to a simple yes or no from the user.

## main()

In the inference component, we have a main loop which serves as the master function of the Play class, which puts the actual code to work after the user runs it. The loop assumes that once the program runs, the user is playing until the user says stop, or either player runs out of chips. Until then, the game progresses with each poker hand played - meaning the chips carry over, the button is switched and the respective consequences of that ensue. 

For the first iteration of the program, (first iteration meaning first poker hand played), the program collects who has the button first and the initial starting stack. The program does this the one time only, and this data is used to initialize the variables that will carry over throughout the rounds, like the player's position and how their stack changes. The resolution of each poker hand is determined by a check of if the game has reached street 3 and if betting ongoing is False (a Boolean flag to check if the betting round is over yet), or if either player folded.

Using the similar helper functions we see in the framework of the environment, we update the game states per iteration (per hand played). The user can also signal that he will opt out of the game at any time with an input to the following question from the agent: "Will that be all sir?"

