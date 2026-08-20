#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from DDQN_Training import PokerDQN
from environment import Poker
import numpy as np
import torch
from collections import deque

env = Poker()
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
observation, null = env.reset()

policy_net = PokerDQN(len(observation), env.action_space.n).to(device)
policy_net.load_state_dict(torch.load("policy_net_weights.pth", map_location = device))
policy_net.eval()

def choose_action(obs):
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(obs)
        mask = torch.tensor(env.action_mask(), device=device)
        q_values[:,~mask] = -1e10
        action = q_values.argmax(dim=1).item()

    return action


class Play:

    def __init__(self):

        self.button = None
        self.stack = None
        self.starting_stack = None

        self.deck = None
        self.street = None
        self.board = None
        self.h1 = None
        self.h2 = None

        self.opp_stack = None
        self.opp_raises = None
        self.opp_calls = None
        self.opp_action = None

        self.bet_history = None

        self.pot = None

        self.hero_stack = None
        self.win_rate = None

        self.betting_ongoing = None
        self.position = None


    def obs_vector(self):
        call_amount = abs(self.bet_history[1] - self.bet_history[0])

        if call_amount == 0:
            pot_odds = 0
        else:
            pot_odds = call_amount / (self.pot + call_amount)

        opp_agg = min(self.opp_raises / max(self.opp_calls, 1), 5)
        return np.array([
            self.win_rate, 
            self.stack / self.starting_stack,
            self.pot / (2*self.stack), # normalized pot
            self.hero_stack / self.stack, # normalized hero stack
            min(self.hero_stack, self.opp_stack) / self.stack, # normalized effective stack
            self.position,
            pot_odds, # pot odds (doesn't inlcude opponent last bet as it is already accounted for in the pot)
            opp_agg, 
            self.street / 3,
            self.opp_action,
            self.betting_ongoing
        ], dtype = np.float32)


    def advance_street(self):

        self.street += 1
        self.bet_history = deque([0,0],maxlen=2)

        if self.street == 1:
            flop1, flop2, flop3 = input("What are the flop cards sir?").split()
            cards = [Card.new(flop1), Card.new(flop2), Card.new(flop3)]
            self.board.extend(cards)
            self.remove(flop1, flop2, flop3)
        elif self.street == 2:
            turn = input("What is the turn card sir?")
            self.board.append(Card.new(turn))
            self.remove(turn)
        elif self.street == 3:
            river = input("What is the river card sir?")
            self.board.append(Card.new(river))
            self.remove(river)


    def compute_win_rate(self):
        if self.street == 0:
            self.win_rate = simulate_preflop(self.h1, self.h2)
            return self.win_rate

        elif self.street == 1:
            board = [Card.int_to_str(card) for card in self.board]
            self.win_rate = simulate_preturn(self.h1, self.h2, board[0], board[1], board[2])
            return self.win_rate

        elif self.street == 2:
            board = [Card.int_to_str(card) for card in self.board]
            self.win_rate = simulate_preriver(self.h1, self.h2, board[0], board[1], board[2], board[3])
            return self.win_rate
        else:
            board = [Card.int_to_str(card) for card in self.board]
            self.win_rate = simulate_prereveal(self.h1, self.h2, board[0], board[1], board[2], board[3], board[4])
            return self.win_rate



    def done(self, action):
        call_amount = self.bet_history[1] - self.bet_history[0]
        if call_amount == 0 or self.opp_action == 0 or action == 0:
            self.betting_ongoing = False
        else:
            self.betting_ongoing = True


    def chip_mover(self, player, amount):
        if player == "hero":
            self.hero_stack -= amount
            self.pot += amount
        else:
            self.opp_stack -= amount
            self.pot += amount



    def apply_action(self, action):

        if action == 0:
            return

        elif action == 1:
            call_amount = self.bet_history[1] - self.bet_history[0]
            self.chip_mover("hero", call_amount)

        elif action in (2, 3, 4, 5):
            call_amount = self.bet_history[1] - self.bet_history[0]

            if action == 2:
                bet_amount = call_amount + 0.33*self.pot
                self.bet_history.append(bet_amount)

            elif action == 3:
                bet_amount = call_amount + 0.67*self.pot
                self.bet_history.append(bet_amount)

            elif action == 4:
                bet_amount = call_amount + self.pot
                self.bet_history.append(bet_amount)

            elif action == 5:
                bet_amount = call_amount + 1.25*self.pot
                self.bet_history.append(bet_amount)

            self.chip_mover("hero", bet_amount)

        elif action == 6:
            self.chip_mover("hero", self.hero_stack)
            self.bet_history.append(self.hero_stack)



    def showdown(self):
        ui = input("Did you win or lose?")
        if ui == "w":
            return True
        else:
            return False


    def what_opp(self):
        action = input("What is your opponent's action sir?")
        if action == "fold":
            action = 0 

        elif action == "call" or action == "check":
            self.opp_calls += 1

            if action == "call":
                amount = abs(self.bet_history[1] - self.bet_history[0])

            else:
                amount = 0
            self.chip_mover("opp", amount)
            self.bet_history.append(self.bet_history[1])
            action = 1

        elif action == "bet" or action == "raise":
            self.opp_raises += 1
            amount = float(input("What is the the amount sir?"))
            self.chip_mover("opp", amount)
            self.bet_history.append(amount)

            if amount <= 0.33*self.pot:
                action = 2

            elif amount <= 0.67*self.pot:
                action = 3

            elif amount <= self.pot:
                action = 4

            else:
                action = 5

        elif action == "all-in":
            self.bet_history.append(self.opp_stack)
            self.chip_mover("opp", self.opp_stack)
            action = 6

        return action


    def remove(self, *args):
        for c in args:
            self.deck.cards.remove(Card.new(c))


    def what_action(self, action):
        if action == 0:
            self.apply_action(action)
            return "fold"
        elif action == 1:
            self.apply_action(action)
            return "call"
        elif action == 2:
            self.apply_action(action)
            return f"bet/raise {self.pot*0.33}"
        elif action == 3:
            self.apply_action(action)
            return f"bet/raise {self.pot*0.67}"
        elif action == 4:
            self.apply_action(action)
            return f"bet/raise {self.pot}"
        elif action == 5:
            self.apply_action(action)
            return f"bet/raise {self.pot*1.25}"
        else:
            self.apply_action(action)
            return "all-in"


    def main(self):

        done = False

        blind = input("As of right now, are you the small blind sir?")
        small_blind = float(input("How much is small blind?"))
        self.button = 0 if blind == "yes" else 1
        big_blind = 2*small_blind
        stack = float(input("What will stack amount be sir?"))
        self.starting_stack = stack
        self.stack = stack
        self.hero_stack = stack
        self.opp_stack = stack
        count = 0

        while not done:      

            hand_over = False
            count += 1
            self.h1 = None
            self.h2 = None
            self.deck = Deck()
            self.h1, self.h2 = input("Your cards sir?").split()
            self.remove(self.h1, self.h2)

            self.pot = small_blind + big_blind
            if (count % 2 != 0 and self.button == 0) or (count % 2 == 0 and self.button == 1):
                self.bet_history = deque([small_blind, big_blind], maxlen=2)
                self.position = 0
            else:
                self.bet_history = deque([big_blind, small_blind], maxlen=2)
                self.position = 1

            self.street = 0
            self.stack = self.hero_stack
            self.board = []
            self.opp_calls = 0
            self.opp_raises = 0
            self.opp_action = 1
            self.betting_ongoing = True
            while not hand_over:

                if self.street == 0:
                    self.compute_win_rate()

                    if self.position == 0:
                        while self.betting_ongoing:
                            obs = self.obs_vector()
                            action = choose_action(obs)
                            action_int = action
                            action = self.what_action(action_int)
                            print(action)
                            self.done(action_int)
                            if self.betting_ongoing == False:
                                break

                            self.opp_action = self.what_opp()
                            self.done(action_int)

                    else:
                        while self.betting_ongoing:
                            self.opp_action = self.what_opp()
                            self.done(action_int)
                            if self.betting_ongoing == False:
                                break

                            obs = self.obs_vector()
                            action = choose_action(obs)
                            action_int = action
                            action = self.what_action(action_int)
                            print(action)

                            self.done(action_int)

                elif self.street == 1:
                    self.compute_win_rate()

                    if self.position == 1:
                        while self.betting_ongoing:
                            obs = self.obs_vector()
                            action = choose_action(obs)
                            action_int = action
                            action = self.what_action(action_int)
                            print(action)
                            self.done(action_int)
                            if self.betting_ongoing == False:
                                break

                            self.opp_action = self.what_opp()
                            self.done(action_int)

                    else:
                        while self.betting_ongoing:
                            self.opp_action = self.what_opp()
                            self.done(action_int)
                            if self.betting_ongoing == False:
                                break

                            obs = self.obs_vector()
                            action = choose_action(obs)
                            action_int = action
                            action = self.what_action(action_int)
                            print(action)

                            self.done(action_int)  

                elif self.street == 2:
                    self.compute_win_rate()

                    if self.position == 1:
                        while self.betting_ongoing:
                            obs = self.obs_vector()
                            action = choose_action(obs)
                            action_int = action
                            action = self.what_action(action_int)
                            print(action)
                            self.done(action_int)
                            if self.betting_ongoing == False:
                                break

                            self.opp_action = self.what_opp()
                            self.done(action_int)

                    else:
                        while self.betting_ongoing:
                            self.opp_action = self.what_opp()
                            self.done(action_int)
                            if self.betting_ongoing == False:
                                break

                            obs = self.obs_vector()
                            action = choose_action(obs)
                            action_int = action
                            action = self.what_action(action_int)
                            print(action)

                            self.done(action_int) 

                elif self.street == 3:
                    self.compute_win_rate()

                    if self.position == 1:
                        while self.betting_ongoing:
                            obs = self.obs_vector()
                            action = choose_action(obs)
                            action_int = action
                            action = self.what_action(action_int)
                            print(action)
                            self.done(action_int)
                            if self.betting_ongoing == False:
                                break

                            self.opp_action = self.what_opp()
                            self.done(action_int)

                    else:
                        while self.betting_ongoing:
                            self.opp_action = self.what_opp()
                            self.done(action_int)
                            if self.betting_ongoing == False:
                                break

                            obs = self.obs_vector()
                            action = choose_action(obs)
                            action_int = action
                            action = self.what_action(action_int)
                            print(action)

                            self.done(action_int)


                if action == 0:
                    self.opp_stack += self.pot
                    hand_over = True

                elif self.opp_action == 0:
                    self.hero_stack += self.pot
                    hand_over = True

                elif self.street == 3 or self.opp_stack <= 0 or self.hero_stack <= 0:
                    win = self.showdown()
                    if win is True:
                        self.hero_stack += self.pot
                    else:
                        self.opp_stack += self.pot

                    hand_over = True


                if not hand_over:
                    self.advance_street()

            done = True if input("Will that be all sir?") == "yes" else False

