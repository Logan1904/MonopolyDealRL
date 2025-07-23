from card import *

# Array where all the cards of the deck will be stored
ALL_CARDS = []

ALL_CARDS.append(PropertyCard(10, "Light Green/Black", 2, ["Black","Light Green"]))
ALL_CARDS.append(PropertyCard(12, "Brown/Light Blue", 1, ["Black","Light Green"]))
ALL_CARDS.append(PropertyCard(14, "Black/Light Blue", 4, ["Black","Light Green"]))
ALL_CARDS.append(PropertyCard(15, "Green/Black", 4, ["Black","Light Green"]))
ALL_CARDS.append(PropertyCard(16, "Green/Dark Blue", 4, ["Black","Light Green"]))

ALL_CARDS.append(MoneyCard(39, "10M", 10))

for i in range(2):
    ALL_CARDS.append(PropertyCard(0, "Dark Blue", 4, ["Dark Blue"]))
    ALL_CARDS.append(PropertyCard(1, "Brown", 1, ["Brown"]))
    ALL_CARDS.append(PropertyCard(2, "Light Green", 2, ["Light Green"]))

    ALL_CARDS.append(PropertyCard(13, "Pink/Orange", 2, ["Pink", "Orange"]))
    ALL_CARDS.append(PropertyCard(11, "Red/Yellow", 3, ["Red", "Yellow"]))
    ALL_CARDS.append(PropertyCard(17, "Wild", 0, ["Wild"]))

    ALL_CARDS.append(ActionCard(19, "Hotel", 4, "Add on top of a house"))
    ALL_CARDS.append(ActionCard(20, "Double The Rent", 1, "Double the rent"))
    ALL_CARDS.append(ActionCard(26, "Deal Breaker", 5, "Steal a completed set"))

    ALL_CARDS.append(RentCard(28, "Red/Yellow Rent", 1, ["Red", "Yellow"]))
    ALL_CARDS.append(RentCard(29, "Green/Dark Blue Rent", 1, ["Green", "Dark Blue"]))
    ALL_CARDS.append(RentCard(30, "Pink/Orange Rent", 1, ["Pink", "Orange"]))
    ALL_CARDS.append(RentCard(31, "Black/Light Green Rent", 1, ["Black", "Light Green"]))
    ALL_CARDS.append(RentCard(32, "Brown/Light Blue Rent", 1, ["Brown", "Light Blue"]))
                     
    ALL_CARDS.append(MoneyCard(38, "5M", 5))

for i in range(3):
    ALL_CARDS.append(PropertyCard(3, "Green", 4, ["Green"]))
    ALL_CARDS.append(PropertyCard(4, "Light Blue", 1, ["Light Blue"]))
    ALL_CARDS.append(PropertyCard(5, "Red", 3, ["Red"]))
    ALL_CARDS.append(PropertyCard(6, "Yellow", 3, ["Yellow"]))
    ALL_CARDS.append(PropertyCard(7, "Orange", 2, ["Orange"]))
    ALL_CARDS.append(PropertyCard(8, "Pink", 2, ["Pink"]))

    ALL_CARDS.append(ActionCard(18, "House", 3, "Add on top of a property"))
    ALL_CARDS.append(ActionCard(21, "Forced Deal", 3, "Swap properties"))
    ALL_CARDS.append(ActionCard(22, "Debt Collector", 3, "Steal 5M"))
    ALL_CARDS.append(ActionCard(23, "Sly Deal", 3, "Steal a property"))
    ALL_CARDS.append(ActionCard(24, "It's My Birthday", 2, "All players give you 2M"))
    ALL_CARDS.append(ActionCard(27, "Just Say No", 4, "Cancel an action"))

    ALL_CARDS.append(RentCard(33, "Wild Rent", 3, ["Wild"]))

    ALL_CARDS.append(MoneyCard(36, "3M", 3))
    ALL_CARDS.append(MoneyCard(37, "4M", 4))

for i in range(4):
    ALL_CARDS.append(PropertyCard(9, "Black", 2, ["Black"]))

for i in range(5):
    ALL_CARDS.append(MoneyCard(35, "2M", 2))

for i in range(6):
    ALL_CARDS.append(MoneyCard(34, "1M", 1))

for i in range(8):
    ALL_CARDS.append(ActionCard(25, "Pass Go", 1, "Take 2 cards"))

    