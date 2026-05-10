from Card import *

class PropertySet:
    def __init__(self, colour, maxSize):
        self.properties = []
        self.colour = colour
        self.maxSize = maxSize
        self.hasHouse = False
        self.hasHotel = False

    def __repr__(self):
        return str([repr(card) for card in self.properties])

    def addProperty(self, property):
        if self.canAddProperty(property):
            self.properties.append(property)

    def canAddProperty(self, property):
        # Wilds may be placed into empty sets — caveat in rentValue: a set
        # whose contents are entirely wild has no rent value.
        correct_colour = self.colour in property.colours or "Wild" in property.colours
        is_not_full = not (self.isCompleted())

        return correct_colour and is_not_full

    def removeProperty(self, property):
        for p in self.properties:
            if p.id == property.id:
                return self.properties.remove(p)

    def clearSet(self):
        for property in self.properties:
            self.removeProperty(property)

    def rentValue(self):
        # Caveat: wild-only sets (empty or all pure-wild contents) earn no rent.
        # A pure wild placed alone in a coloured bucket doesn't commit to that
        # colour for rent purposes.
        if self.isEmpty() or self.isOnlyWild():
            return 0

        if self.colour == "Blue":
            rent = 3 if len(self.properties) == 1 else 8
        elif self.colour == "Brown":
            rent = len(self.properties)
        elif self.colour == "Light Green":
            rent = len(self.properties)
        elif self.colour == "Green":
            rent = 2 * len(self.properties) if len(self.properties) <= 2 else 7
        elif self.colour == "Light Blue":
            rent = len(self.properties)
        elif self.colour == "Red":
            rent = len(self.properties) + 1 if len(self.properties) <= 2 else 6
        elif self.colour == "Yellow":
            rent = 2 * len(self.properties)
        elif self.colour == "Orange":
            rent = 2 * len(self.properties) - 1
        elif self.colour == "Pink":
            rent = 2 ** (len(self.properties) - 1)
        elif self.colour == "Black":
            rent = len(self.properties)
        elif self.colour == "Wild":
            rent = 0
    
        return rent + (3 if self.hasHouse else 0) + (4 if self.hasHotel else 0)
    
    def isCompleted(self):
        if self.colour != "Wild":
            return len(self.properties) >= self.maxSize
        else:
            return False
    
    def isOnlyWild(self):
        for p in self.properties:
            if not p.isWild():
                return False
        return True
    
    def isEmpty(self):
        return not self.properties
