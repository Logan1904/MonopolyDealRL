import copy

from property_set import *
from card import *
from mappings import *

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

        self.sets = {
            colour: [PropertySet(colour,maxSize),PropertySet(colour,maxSize),PropertySet(colour,maxSize),
                     PropertySet(colour,maxSize),PropertySet(colour,maxSize),PropertySet(colour,maxSize),
                     PropertySet(colour,maxSize),PropertySet(colour,maxSize),PropertySet(colour,maxSize)] for colour,maxSize in SET_LENGTH.items()
        }

    def __repr__(self):
        return self.name

    def drawTwo(self):
        cards = self.deck.getCards(2)
        self.hand += cards

    def removeHandCardById(self, card_id):
        for i,card in enumerate(self.hand):
            if card.id == card_id:
                return self.hand.pop(i)
    
    def removeHandCard(self, card):
        for hand_card in self.hand:
            if hand_card.id == card.id:
                self.hand.remove(hand_card)
                break

    def getHandCardById(self, card_id):
        for i,card in enumerate(self.hand):
            if card.id == card_id:
                return copy.deepcopy(card)

    def removeProperty(self, colour, set_index, card):
        pSet = self.sets[colour][set_index]
        pSet.removeProperty(card)

    def addProperty(self, colour, set_index, card):
        pSet = self.sets[colour][set_index]
        pSet.addProperty(card)

    def removePropertyById(self, colour, set_index, card):
        pSet = self.sets[colour][set_index]
        for i,pCard in enumerate(pSet.properties):
            if pCard.id == card:
                return pSet.properties.pop(i)

    def getPropertyById(self, colour, set_index, card):
        pSet = self.sets[colour][set_index]
        for i,pCard in enumerate(pSet.properties):
            if pCard.id == card:
                return copy.deepcopy(pCard)

    def removeSetByID(self, colour, set_index):
        pSet = self.sets[colour][set_index]
        self.sets[colour][set_index].clearSet()
        return pSet

    def addMoney(self, card):
        self.money.append(card)

    def removeMoney(self, card):
        self.money.remove(card)

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
        if self.money:
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