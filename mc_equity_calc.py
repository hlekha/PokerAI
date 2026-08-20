#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from treys import Card, Deck, Evaluator

evalu = Evaluator()

# -----------------------------------------------------------
# Monte Carlo Equity Calculator
# -----------------------------------------------------------

def simulate_preflop(h1, h2):

    your_hand = [Card.new(h1), Card.new(h2)]

    count = 0


    N = 2000  

    for i in range(N):
        deck = Deck()

        # remove hero hand from deck
        deck.cards.remove(your_hand[0])
        deck.cards.remove(your_hand[1])

        villain_hand = deck.draw(2)
        board = deck.draw(5)

        hero_score = evalu.evaluate(board, your_hand)
        villain_score = evalu.evaluate(board, villain_hand)


        if villain_score > hero_score:
            count += 1
        else:
            pass

    win_rate = count / N


    return win_rate

def simulate_preturn(h1, h2, b1, b2, b3):

    your_hand = [Card.new(h1), Card.new(h2)]
    flop = [Card.new(b1), Card.new(b2), Card.new(b3)]

    count = 0

    N = 2000

    for i in range(N):
        deck = Deck()

        # remove hero cards and flop cards
        deck.cards.remove(your_hand[0])
        deck.cards.remove(your_hand[1])
        for c in flop:
            deck.cards.remove(c)

        villain_hand = deck.draw(2)
        turn_river = deck.draw(2)
        board = flop + turn_river

        hero_score = evalu.evaluate(board, your_hand)
        villain_score = evalu.evaluate(board, villain_hand)


        if villain_score > hero_score:
            count += 1
        else:
            pass

    win_rate = count / N


    return win_rate

def simulate_preriver(h1, h2, b1, b2, b3, b4):

    your_hand = [Card.new(h1), Card.new(h2)]
    flop_turn = [
        Card.new(b1), 
        Card.new(b2), 
        Card.new(b3), 
        Card.new(b4)
    ]

    count = 0

    N = 2000

    for i in range(N):
        deck = Deck()

        deck.cards.remove(your_hand[0])
        deck.cards.remove(your_hand[1])

        for c in flop_turn:
            deck.cards.remove(c)

        villain_hand = deck.draw(2)
        river = deck.draw(1)
        board = flop_turn + river

        hero_score = evalu.evaluate(board, your_hand)
        villain_score = evalu.evaluate(board, villain_hand)


        if villain_score > hero_score:
            count += 1
        else:
            pass

    win_rate = count / N

    return win_rate

def simulate_prereveal(h1, h2, b1, b2, b3, b4, b5):

    your_hand = [Card.new(h1), Card.new(h2)]
    board = [
        Card.new(b1),
        Card.new(b2),
        Card.new(b3),
        Card.new(b4),
        Card.new(b5)
    ]

    hero_score = evalu.evaluate(board, your_hand)

    count = 0

    N = 2000

    for i in range(N):
        deck = Deck()
        deck.cards.remove(your_hand[0])
        deck.cards.remove(your_hand[1])
        for card in board:
            deck.cards.remove(card)

        villain_hand = deck.draw(2)

        villain_score = evalu.evaluate(board, villain_hand)

        if villain_score > hero_score:
            count += 1
        else:
            pass

    win_rate = count / N

    return win_rate

