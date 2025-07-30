import numpy as np

from Card import *
from mappings import *


class ActionMask():
    def __init__(self):
        self.initialise_action_mask()
        
    def initialise_action_mask(self):
        self.action_mask = {
            "action_ID": np.zeros(NUM_ACTIONS, dtype=np.int8),
            "hand_card": np.zeros(NUM_UNIQUE_CARDS, dtype=np.int8),
            "opponent_ID": np.zeros(NUM_OPPONENTS+1, dtype=np.int8),
            "property_card": {
                "colour": np.zeros(NUM_UNIQUE_COLOURS, dtype=np.int8),
                "set_index": np.zeros(MAX_SETS_PER_PROPERTY, dtype=np.int8),
                "card": np.zeros(NUM_UNIQUE_PROPERTY_CARDS, dtype=np.int8)
            },
            "set": {
                "colour": np.zeros(NUM_UNIQUE_COLOURS, dtype=np.int8),
                "set_index": np.zeros(MAX_SETS_PER_PROPERTY, dtype=np.int8)
            }
        }

    def set_action_ID(self, internal_state):
        # set action mask based on cards in hand

        players, agents, agent_selection, deck, action_context = internal_state
        player = players[agent_selection]

        # can always skip
        self.action_mask["action_ID"][0] = 1

        # move property, at least 1 property on the board (can be wild)
        self.action_mask["action_ID"][1] = player.hasAtLeastOnePropertyOnBoard()
        
        # play money
        self.action_mask["action_ID"][2] = player.hasMoneyInHand()

        # play property
        self.action_mask["action_ID"][3] = player.hasPropertyInHand()

        # play wild property, at least 1 property on the board (cannot be wild)
        self.action_mask["action_ID"][4] = player.hasWildPropertyInHand() and player.hasAtLeastOneNonWildPropertyOnBoard()

        # sly deal, at least one opponent must have at least 1 property on the board
        if player.hasSlyDeal():
            for opponent in players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOnePropertyOnBoard():
                    self.action_mask["action_ID"][5] = 1
                    break
        
        # forced deal, at least one opponent AND player must have at least 1 property on the board
        if player.hasForcedDeal():
            for opponent in players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOnePropertyOnBoard():
                    self.action_mask["action_ID"][6] = 1
                    break
        
        # debt collector, at least one opponent must have >0 money on the board
        if player.hasDebtCollector():
            for opponent in players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOneMoneyOnBoard():
                    self.action_mask["action_ID"][7] = 1
                    break

        # its my birthday, at least one opponent must have >0 money on the board
        if player.hasItsMyBirthday():
            for opponent in players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOneMoneyOnBoard():
                    self.action_mask["action_ID"][8] = 1
                    break

        # deal breaker, at least one opponent must have at least one set on the board
        if player.hasDealBreaker():
            for opponent in players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOneSetOnBoard():
                    self.action_mask["action_ID"][9] = 1
                    break
        
        # rent, at least one property of that colour AND at least one opponent with >0 money
        for opponent in players.values():
            if opponent == player:
                continue
            if opponent.hasAtLeastOneMoneyOnBoard():
                pColours = player.whichColoursOnBoard()
                rColours = player.whichRentColoursInHand()

                validColours = pColours.intersection(rColours)

                if "Red" in validColours:
                    self.action_mask["action_ID"][10] = 1
                if "Green" in validColours:
                    self.action_mask["action_ID"][11] = 1
                if "Pink" in validColours:
                    self.action_mask["action_ID"][12] = 1
                if "Black" in validColours:
                    self.action_mask["action_ID"][13] = 1
                if "Brown" in validColours:
                    self.action_mask["action_ID"][14] = 1
                if "Wild" in rColours:
                    self.action_mask["action_ID"][15] = 1
        
        # just say no, masked 
        self.action_mask["action_ID"][16] = 0

    def set_hand_card(self, internal_state):
        # set action mask based on cards in hand and action_ID

        players, agents, agent_selection, deck, action_context = internal_state
        player = players[agent_selection]
        action_ID = action_context["action"]
    
        if action_ID == 2:      # money
            for card in player.hand:
                self.action_mask["hand_card"][card.id] = 1

        elif action_ID == 3:    # property
            for card in player.hand:
                if isinstance(card,PropertyCard):
                    self.action_mask["hand_card"][card.id] = 1

        elif action_ID == 4:    # wild property
            self.action_mask["hand_card"][17] = 1

        elif action_ID == 5:    # sly deal
            self.action_mask["hand_card"][23] = 1

        elif action_ID == 6:    # forced deal
            self.action_mask["hand_card"][21] = 1

        elif action_ID == 7:    # debt collector
            self.action_mask["hand_card"][22] = 1

        elif action_ID == 8:    # its my birthday
            self.action_mask["hand_card"][24] = 1

        elif action_ID == 9:    # deal breaker
            self.action_mask["hand_card"][26] = 1

        elif action_ID == 10:    # rent, red/yellow
            self.action_mask["hand_card"][28] = 1

        elif action_ID == 11:    # rent, green/dark blue
            self.action_mask["hand_card"][29] = 1

        elif action_ID == 12:    # rent, pink/orange
            self.action_mask["hand_card"][30] = 1

        elif action_ID == 13:    # rent, black/light green
            self.action_mask["hand_card"][31] = 1

        elif action_ID == 14:    # rent, brown/light blue
            self.action_mask["hand_card"][32] = 1

        elif action_ID == 15:    # rent, wild
            self.action_mask["hand_card"][33] = 1

    def set_opponent(self, internal_state):
        players, agents, agent_selection, deck, action_context = internal_state

        self.action_mask["opponent_ID"] = np.ones(NUM_OPPONENTS+1, dtype=np.int8)
        self.action_mask["opponent_ID"][agents.index(str(agent_selection))] = 0

    def set_property_colour(self, internal_state, target_opponent):
        # set action mask based on agents properties on the board

        players, agents, agent_selection, deck, action_context = internal_state
        if target_opponent:
            opponent = agents[action_context["opponent_ID"]]  
            target = players[opponent]
        else:
            target = players[agent_selection]

        # get indices of properties held by player
        for cind,pSets in enumerate(target.sets.values()):
            for pSet in pSets:
                if not pSet.isEmpty() and not pSet.isOnlyWild():
                    self.action_mask["property_card"]["colour"][cind] = 1
                    break

    def set_property_set_index(self, internal_state, target_opponent):
        # set action mask based on agents properties on the board

        players, agents, agent_selection, deck, action_context = internal_state
        if target_opponent:
            opponent = agents[action_context["opponent_ID"]] 
            target = players[opponent]
            colour_ID = action_context["opponent_property"]["colour"]
        else:
            target = players[agent_selection]
            colour_ID = action_context["my_property"]["colour"]
        
        colour = decode_colour(colour_ID)
        
        # get indices of properties held by player
        for pind,pSet in enumerate(target.sets[colour]):
            if not pSet.isEmpty() and not pSet.isOnlyWild():
                self.action_mask["property_card"]["set_index"][pind] = 1
                break

    def set_property_card(self, internal_state, target_opponent):
        # set action mask based on agents properties on the board

        players, agents, agent_selection, deck, action_context = internal_state
        if target_opponent:
            opponent = agents[action_context["opponent_ID"]] 
            target = players[opponent]
            colour_ID = action_context["opponent_property"]["colour"]
            set_index = action_context["opponent_property"]["set_index"]
        else:
            target = players[agent_selection]
            colour_ID = action_context["my_property"]["colour"]
            set_index = action_context["my_property"]["set_index"]

        colour = decode_colour(colour_ID)
        
        for card in target.sets[colour][set_index].properties:
            self.action_mask["property_card"]["card"][card.id] = 1

    def set_set_colour(self, internal_state, target_opponent):
        # set action mask on set colour

        players, agents, agent_selection, deck, action_context = internal_state
        if target_opponent:
            opponent = agents[action_context["opponent_ID"]] 
            target = players[opponent]
        else:
            target = players[agent_selection]

        # playing a property into a set
        if action_context["action"] < 9:
            if action_context["action"] in [1]:
                # move property
                my_property = action_context["my_property"]

                # decode colour
                colour = decode_colour(my_property["colour"])

                # get property
                pCard = target.getPropertyById(colour,my_property["set_index"],my_property["card"])
            elif action_context["action"] in [3,4]:
                # play property or wild property
                card_ID = action_context["hand_card"]
                
                # get property
                pCard = target.getHandCardById(card_ID)
            elif action_context["action"] in [5,6]:
                # sly deal or forced deal
                opponent = players[agents[action_context["opponent_ID"]]]
                opponent_property = action_context["opponent_property"]

                # decode colour
                colour = decode_colour(opponent_property["colour"])

                # get property
                pCard = opponent.getPropertyById(colour,opponent_property["set_index"],opponent_property["card"])

            for cind,(colour,pSets) in enumerate(target.sets.items()):
                for pSet in pSets:
                    if pSet.canAddProperty(pCard):
                        self.action_mask["set"]["colour"][cind] = 1
                        break
        else:
            if action_context["action"] == 9:
                # deal breaker, unmask all full sets
                for cind,(colour,pSets) in enumerate(target.sets.items()):
                    for pSet in pSets:
                        if pSet.isCompleted():
                            self.action_mask["set"]["colour"][cind] = 1
                            break
            else:
                # rent, unmask based on rent card
                card_ID = action_context["hand_card"]
                # get rent card
                rCard = target.getHandCardById(card_ID)

                if rCard.isWild():
                    for cind,(colour,pSets) in enumerate(target.sets.items()):
                        for pSet in pSets:
                            if not pSet.isEmpty():
                                self.action_mask["set"]["colour"][cind] = 1
                                break
                else:
                    for cind,(colour,pSets) in enumerate(target.sets.items()):
                        if colour not in rCard.colours:
                            continue
                        for pSet in pSets:
                            if not pSet.isEmpty():
                                self.action_mask["set"]["colour"][cind] = 1
                                break

    def set_set_index(self, internal_state, target_opponent):
        # set action mask on set index

        players, agents, agent_selection, deck, action_context = internal_state
        if target_opponent:
            opponent = agents[action_context["opponent_ID"]] 
            target = players[opponent]
        else:
            target = players[agent_selection]

        # playing a property into a set
        if action_context["action"] < 9:
            if action_context["action"] in [1]:
                # move property
                my_property = action_context["my_property"]

                # decode colour
                colour = decode_colour(my_property["colour"])

                # get property
                pCard = target.getPropertyById(colour,my_property["set_index"],my_property["card"])
            elif action_context["action"] in [3,4]:
                # play property or wild property
                card_ID = action_context["hand_card"]
                
                # get property
                pCard = target.getHandCardById(card_ID)
            elif action_context["action"] in [5,6]:
                # sly deal or forced deal
                opponent = players[agents[action_context["opponent_ID"]]]
                opponent_property = action_context["opponent_property"]

                # decode colour
                colour = decode_colour(opponent_property["colour"])

                # get property
                pCard = opponent.getPropertyById(colour,opponent_property["set_index"],opponent_property["card"])

            for cind,(colour,pSets) in enumerate(target.sets.items()):
                for pind,pSet in enumerate(pSets):
                    if pSet.canAddProperty(pCard):
                        self.action_mask["set"]["set_index"][pind] = 1
        else:
            if action_context["action"] == 9:
                # deal breaker, unmask all full sets
                for cind,(colour,pSets) in enumerate(target.sets.items()):
                    for pind,pSet in enumerate(pSets):
                        if pSet.isCompleted():
                            self.action_mask["set"]["set_index"][pind] = 1
            else:
                # rent, unmask based on rent card
                card_ID = action_context["hand_card"]

                # get rent card
                rCard = target.getHandCardById(card_ID)

                if rCard.isWild():
                    for cind,(colour,pSets) in enumerate(target.sets.items()):
                        for pind,pSet in enumerate(pSets):
                            if not pSet.isEmpty():
                                self.action_mask["set"]["set_index"][pind] = 1
                else:
                    for cind,(colour,pSets) in enumerate(target.sets.items()):
                        if colour not in rCard.colours:
                            continue
                        for pind,pSet in enumerate(pSets):
                            if not pSet.isEmpty():
                                self.action_mask["set"]["set_index"][pind] = 1