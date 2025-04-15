import random
from cardsdb import ALL_CARDS

class Deck:
    def __init__(self):
        self.deck = ALL_CARDS
        self.shuffle()
        self.discard_pile = []

    def shuffle(self):
        random.shuffle(self.deck)
    
    def draw(self):
        if len(self.deck) == 0 and len(self.discard) > 0:
            self.deck = self.discard
            self.discard = []
            self.shuffle()
            
        return self.deck.pop(0)

    def getCards(self, n):
        cards = []
        for i in range(n):
            cards.append(self.draw())

        return cards

    def discardCard(self, card):
        self.discard_pile.append(card)