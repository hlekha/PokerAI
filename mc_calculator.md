# Monte Carlo Equity Calculator

In this component of the project, I implemented Monte Carlo simulations to estimate the probability of a given hand winning at showdown. This component utilizes the Law of Large Numbers which states the more random experiments, the closer the average of the results will be to the true expected value.

## Purpose

Its purpose is to communicate with the agent its win rate at any given state. This win rate is one of the elements that make up the observation vector, and, as a result, will be directly fed into the network. The win rate will also help with various computations like assigning intermittent rewards - helping the agent perceive hand strength and how dominant it is in the current state. 

## Design

This part of the project was split up into three functions each for the respective street it belongs to. There exists simulate_preflop, simulate_preturn, and simulate_preriver, where each function takes in the player’s card, and the available board cards as the parameters. The only part that varies throughout the different functions are the board cards that are added per street.  Each function uses a Monte Carlo Simulation that fixes the players hand, removes it from the deck, and simulates different possibilities of what the board would look like. It then draws two cards for the opponent’s hand and compares the hand strength of this hypothetical hand to the players hand, and adds one to the counter if the player’s hand strength is stronger. By the end of however much iterations the counter is then divided by the total amount of iterations to retrieve the win rate. In the case of a tie or loss, the count remains untouched for that iteration.



## Mechanics

A lot of the mechanics of the simulation are derived from the treys library. With the treys library, I was able to simulate a hypothetical deck, from which cards can be drawn and assigned, as well as removed from the deck. I utilized this feature to create a deck on each iteration, and deal the player's - and the hypothetical opponent's - cards and the board cards (respective to the street). Since when you draw a card, the deck is unchanged I had to remove each card that I drew so that when the MC simulations begun, the simulation wasn't pulling from a deck containing cards that were in the player's hand or on the board. Furthermore, I also utilized the strength evaluator feature, which assesses the strength of hands against the board cards. I used this feature to quantify the player's hand, as well as the opponent's hand on each iteration; once these values were retrieved, it was compared to see who would be the winner of a hypothetical showdown in that street. 

Since Monte Carlo simulations, by themselves, consume a decent amount of performance time, I minimized the amount of trials needed since this would cost great computing power as well as time during training if n was set too large. I therefore optimized this constraint by using Chebyshev's Inequality. This states that for any random variable, the probability that the random variable straying away from its mean by an error of $\epsilon$ or more is bounded by its variance squared divided by the error squared. 
<p align=center>
  $P(|X-\mu_X| \geq \epsilon ) \leq \frac {\sigma _X ^2}{\epsilon ^2}$
</p>

This rule is especially powerful because since our experiment consists of n trials which are independent of each other, we can substitute the sample mean with $\mu$ and $\sigma ^2$ with the variance of the sample mean - this is equal to $\frac {\sigma ^2}  {n}$. On the right side of the inequality, we are left with $\frac {\sigma ^2}{n\epsilon ^2}$. We then state that the probability of the error between the sample mean and the actual mean is equal to $\epsilon$ or more has an upper bound up $\delta$. 
<p align=center>
  $P(|\overline {X_n}-\mu_X| \geq \epsilon ) \leq \delta$
</p>

We then recall our worst case scenario of $\frac {\sigma ^2}{n\epsilon ^2}$ and replace the left side of the inequality with this quantity, and solve for n.

<p align=center>
  $\frac {\sigma ^2}{n\epsilon ^2} \leq \delta$
<p align=center>
  $\frac {\sigma ^2}{\epsilon ^2} \leq n\delta$
<p align=center>
  $\frac {\sigma ^2}{\epsilon ^2 \delta} \leq n$

With this final derivation,  with set parameters of $\delta$ and $\epsilon$, we are able to find n that is small yet sufficient enough to estimate the probability of winning the given hand. 
Since we can evaluate this random variable as having binary trials, where each trial results in either a win or loss, we can set maximum variance ($\sigma ^2$) to 0.25; along with an error margin of 0.05, and a probability of failure (1 minus our confidence) of 0.05 we are able to evaluate n to be at least 2000 trials. This means the simulation needs at least 2000 trials to be within 5% of the actual equity value, with 95% confidence.

## Integration Into Main Program

In the main program, we call this function throughout the environment (and therefore the training), as well as the inference; given its appearance all throughout the main program, it's conclusive to say that the Monte Carlo Equity Calculator is one of the most crucial parts of this project. Despite our optimal minimization of the number of trials, the drawback remains the same. This component is called inside training and inference, and every time that it is called, the program has to run 2000 iterations of simulating board and opponent hand combinations. As a rough estimate, since it is called at least once per training episode and there are 87500 episodes, the program would have iterated over 175,000,000 times - and that's a strong low-ball! To conclude, the MC Equity Calculator remains a crucial foundational step of this AI, but also a very costly step in performance.
