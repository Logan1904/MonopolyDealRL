
class Card:
    def __init__(self, id, name, value):
        self.id = id
        self.name = name
        self.value = value

    def __repr__(self):
        return self.name
    
class MoneyCard(Card):
    def __init__(self, id, name, value):
        super().__init__(id, name, value)
    
class ActionCard(Card):
    def __init__(self, id, name, value, action):
        super().__init__(id, name, value)
        self.action = action

class PropertyCard(Card):
    def __init__(self, id, name, value, colours):
        super().__init__(id, name, value)
        self.colours = colours
    
    def isWild(self):
        return self.colours[0] == "Wild"
    
class RentCard(Card):
    def __init__(self, id, name, value, colours):
        super().__init__(id, name, value)
        self.colours = colours
    
    def isWild(self):
        return self.colours[0] == "Wild"
