from property_set import *
from card import *

class Player:
    def __init__(self, name, deck):
        # initialise name, empty hand, empty properties, empty money
        self.name = name
        self.hand = []
        self.money = []
        self.deck = deck

        # draw 5 cards to hand
        cards = self.deck.getCards(5)
        self.hand += cards

        # initialise properties
        property_set_number = {
            "Dark Blue": 2,
            "Brown": 2,
            "Light Green": 2,
            "Green": 3,
            "Light Blue": 3,
            "Red": 3,
            "Yellow": 3,
            "Orange": 3,
            "Pink": 3,
            "Black": 4
        }

        self.sets = {
            colour: [PropertySet(colour,maxSize),PropertySet(colour,maxSize)] for colour,maxSize in property_set_number.items()
        }

    def drawTwo(self):
        cards = self.deck.getCards(2)
        self.hand += cards
    
    def hasAtLeastOnePropertyOnBoard(self):
        for colour,pSets in self.sets.items():
            for pSet in pSets:
                if pSet.properties:
                    return True
        return False
    
    def hasAtLeastOneNonWildPropertyOnBoard(self):
        for colour,pSets in self.sets.items():
            for pSet in pSets:
                if pSet.properties and not pSet.isOnlyWild():
                    return True
        return False
    
    def hasAtLeastOneMoneyOnBoard(self):
        if self.money():
            return True
        return False
    
    def hasAtLeastOneSetOnBoard(self):
        for colour,pSets in self.sets.items():
            for pSet in pSets:
                if pSet.isCompleted():
                    return True
        return False

    def whichColoursOnBoard(self):
        colours = set()
        for colour,pSets in self.sets.items():
            for pSet in pSets:
                if not pSet.isEmpty() and not pSet.isOnlyWild():
                    colours.add(colour)
                    break
        return colours

    def hasMoneyInHand(self):
        for card in self.hand:
            if isinstance(card,MoneyCard):
                return True
        return False
        
    def hasPropertyInHand(self):
        for card in self.hand:
            if isinstance(card,PropertyCard):
                return True
        return False
    
    def hasWildPropertyInHand(self):
        for card in self.hand:
            if isinstance(card,PropertyCard) and card.isWild():
                return True
        return False

    def hasSlyDeal(self):
        for card in self.hand:
            if card.name == "Sly Deal":
                return True
        return False
    
    def hasForcedDeal(self):
        for card in self.hand:
            if card.name == "Forced Deal":
                return True
        return False

    def hasDebtCollector(self):
        for card in self.hand:
            if card.name == "Debt Collector":
                return True
        return False

    def hasItsMyBirthday(self):
        for card in self.hand:
            if card.name == "It's My Birthday":
                return True
        return False

    def hasDealBreaker(self):
        for card in self.hand:
            if card.name == "Deal Breaker":
                return True
        return False

    def whichRentColoursInHand(self):
        colours = set()
        for card in self.hand:
            if isinstance(card,RentCard):
                for c in card.colours:
                    colours.add(c)
        
        return colours