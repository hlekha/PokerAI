Since this is a heads-up No-Limit Texas Hold'em RL project, with a poker engine in its environment, it is essential to know poker terminology, as well as poker mechanics. The distinction between the two is how I chose to separate variables from functions. I adopted poker terminology into my program as variables with the same purpose as its respective poker term, and - using those variables - replicated poker mechanics by setting up functions that shaped that nature in which the agent should act. 

This document assumes that you have zero knowledge and will help you to understand both sides of the game - terminology and game mechanics. The poker terminology will also help you understand some of the documents and md files in this repo. As a result, I've bolded the terminology that exists in the code.

## Poker Terminology 

Hole Cards: the two cards that are dealt to each player.  

**Board Cards**: the shared cards everyone can use. It consists of five cards that are flipped over in three stages. The flop which is 3 cards, the turn which is a single card, and the river - the final card.  

**Street**: the stage of the game. Reflects the stages which the cards are revealed. The flop, then the turn, then the river. A given street ends when the betting round is finished or either player folds.  

**Betting Round**: a loop of betting and raising that players enter where they risk their chips to feed the pot. This loop ends when both players check, one player folds, one player calls, or when an all-in is called.  

**Pot**: an entity in the game which pools the chips that are bet by both opponent and player. The pot is rewarded to whoever wins in showdown.  

**Showdown**: once players have reached the final street, and the betting round is finished, the players reveal their cards and the winner is revealed.  

Bluff: a bet placed in hopes of intimidating the opponent into an eventual fold.

**Fold**: player concedes and they forfeit the pot.  

**Check**: bets zero, attempting to pass through the betting round without sacrificing any chips.  

**Call**: betting the amount of chips that is needed to continue the game.  

Cold Call: calling without previously investing

**Bet**: the first chips put into a street.  

**Raise**: an increase in the existing bet.  

Re-raise: raising a raise.  

**All-in**: when a player bets their entire stack. 

**Equity**: the probability that your hand wins, given your opponent and you reach showdown.  

**Button**: when you have the button, you are small blind allowing you to act first in the first round.  

**Small Blind**: The forced bet that you must give for playing the game.  

**Big Blind**: The forced bet you must give for playing the game, it equals to the small blind times two.  

Limp: calling the big blind.  

**Stack**: The amount of chips a player has,

**Effective Stack**: The smaller stack of the hero and villain.

**Starting Stack**: The amount of chips that the player begins win.

In Position: acts after opponent.  

Out of Position: Acts before opponent.  

Implied Odds: the chips expected to win later.  

Reverse Implied Odds: the chips expected to lose.

Fold Equity: probability of opponent folding.

Board Texture: describes how strong the board cards are together.

Dry Board: cards that are far apart in terms of rank and have different suits.

Wet Board: highly connected cards that are close in rank or have the same suits.

Rainbow: every card is a different suit.

Two-Tone: two cards of the same suit exist.  

Monotone: all cards are of the same suits.  

## Poker Mechanics
