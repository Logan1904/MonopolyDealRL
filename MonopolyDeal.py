import functools

import gymnasium as gym
import numpy as np
import copy

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from deck import *
from player import *
from card import *



def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = MonopolyDeal(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class MonopolyDeal(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "MD"}

    def __init__(self, render_mode=None):
        
        # IMPORTANT CONSTANTS
        self.num_players = 4                        # Number of players including agent
        self.num_opponents = self.num_players - 1

        self.max_hand_size = 13                     # Maximum number of cards in hand (start with 7, play 3 Pass Go's)
        self.max_money_size = 6                     # Maximum possible number of money cards in hand (6 x 1m)
        self.max_sets_per_property = 3              # Maximum possible number of sets per property colour
        self.max_cards_per_set = 3                  # Maximum possible number of cards per property set
        self.max_any_card = 8                       # Maximum possible number of any card (8 x Pass Go)

        self.num_unique_colours = 10                # Number of unique colours
        self.num_unique_property_cards = 18         # Number of property cards
        self.num_unique_money = 6                   # Number of unique money cards
        self.num_unique_cards = 40                  # Number of unique cards in deck

        self.num_actions = 17                       # Number of actions
        self.max_decisions = 4                      # Number of decisions (choose opponent, choose my property, choose your property, choose set)

        self.property_set_number = {
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

        self.possible_agents = ["player_" + str(r) for r in range(self.num_players)]

        # a mapping between agent name and ID
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.render_mode = render_mode

    # Observation space should be defined here.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Define observation space
        """
        
        return gym.spaces.Dict({
            "hand": gym.spaces.Box(low=0, high=self.max_any_card, shape=(self.num_unique_cards,), dtype=np.int8),
            "property": gym.spaces.Dict({
                colour: gym.spaces.Dict({
                    "cards": gym.spaces.Box(low=-1, high=self.num_unique_property_cards, shape=(self.max_sets_per_property,max_cards), dtype=np.int8),
                    "full_set": gym.spaces.MultiBinary(self.max_sets_per_property)
                }) for colour,max_cards in self.property_set_number.items()
            }),
            "money": gym.spaces.Box(low=0, high=self.max_money_size, shape=(self.num_unique_money,), dtype=np.int8),
            "opponent_property": gym.spaces.Dict({
                colour: gym.spaces.Dict({
                    "cards": gym.spaces.Box(low=-1, high=self.num_unique_property_cards, shape=(self.num_opponents,self.max_sets_per_property,max_cards), dtype=np.int8),
                    "full_set": gym.spaces.MultiBinary([self.num_opponents,self.max_sets_per_property])
                }) for colour,max_cards in self.property_set_number.items()
            }),
            "opponent_money": gym.spaces.Box(low=0, high=self.max_money_size, shape=(self.num_opponents,self.num_unique_money,), dtype=np.int8),
            "actions_left": gym.spaces.Discrete(4),
            "discard_pile": gym.spaces.Box(low=0, high=self.max_any_card, shape=(self.num_unique_cards,), dtype=np.int8),
            "action_context": gym.spaces.Dict({
                "action": gym.spaces.Discrete(self.num_actions+1, start=-1),
                "decision": gym.spaces.Discrete(self.max_decisions+1, start=-1),
                "opponent": gym.spaces.Discrete(self.num_opponents+2, start=-1),
                "opponent_property": gym.spaces.Dict({
                    "colour": gym.spaces.Discrete(self.num_unique_property_cards+1, start=-1),
                    "set_index": gym.spaces.Discrete(self.max_sets_per_property+1, start=-1),
                    "card": gym.spaces.Discrete(self.num_unique_property_cards+1, start=-1)
                }),
                "opponent_set": gym.spaces.Dict({
                    "colour": gym.spaces.Discrete(self.num_unique_property_cards+1, start=-1),
                    "set_index": gym.spaces.Discrete(self.max_sets_per_property+1, start=-1)
                }),
                "my_property": gym.spaces.Dict({
                    "colour": gym.spaces.Discrete(self.num_unique_property_cards+1, start=-1),
                    "set_index": gym.spaces.Discrete(self.max_sets_per_property+1, start=-1),
                    "card": gym.spaces.Discrete(self.num_unique_property_cards+1, start=-1)
                }),
                "my_set": gym.spaces.Dict({
                    "colour": gym.spaces.Discrete(self.num_unique_property_cards+1, start=-1),
                    "set_index": gym.spaces.Discrete(self.max_sets_per_property+1, start=-1)
                })
            })
        })

    # Action space should be defined here.
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Define action space
        0: Skip
        1: Move property
        2: Play money
        3: Play property
        4: Play wild property
        5: Sly Deal
        6: Forced Deal
        7: Debt Collector
        8: It's My Birthday
        9: Deal Breaker
        10: Rent, Red/Yellow
        11: Rent, Green/Dark Blue
        12: Rent, Pink/Orange
        13: Rent, Black/Light Green
        14: Rent, Brown/Light Blue
        15: Rent, Wild
        16: Counter (JSN)
        """

        return gym.spaces.Dict(
            {   
                "action_ID": gym.spaces.Discrete(self.num_actions),                         # Choose an action
                "hand_card": gym.spaces.Discrete(self.num_unique_cards),                    # Choose a card from hand
                "opponent": gym.spaces.Discrete(self.num_opponents+1),                      # Choose an opponent
                "property_card": gym.spaces.Dict({                                          # Choose a card
                    "colour": gym.spaces.Discrete(self.num_unique_colours),
                    "set_index": gym.spaces.Discrete(self.max_sets_per_property),
                    "card": gym.spaces.Discrete(self.num_unique_property_cards)
                }),
                "set_index": gym.spaces.Dict({                                              # Choose a set
                    "colour": gym.spaces.Discrete(self.num_unique_colours),
                    "set_index": gym.spaces.Discrete(self.max_sets_per_property)
                })
            }
        )

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        # initialise list of agents, shuffle for random order
        self.agents = self.possible_agents[:]
        np.random.shuffle(self.agents)

        # Our agent_selector utility allows easy cyclic stepping through the agents list.
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # initialise rewards, cumulative rewards
        self.rewards = {agent: 0 for agent in self.agents}
        self.cumulative_rewards = {agent: 0 for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        # intialise termination
        self.terminations = {agent: False for agent in self.agents}

        # initialise dummy infos
        self.infos = {agent: {} for agent in self.agents}

        # initialise state
        self.deck = Deck()
        self.players = {agent: Player(agent,self.deck) for agent in self.agents}
        self.actions_left = {agent: 3 for agent in self.agents}
        self.reset_action_context()

        # initialise observation dictionary
        self.observations = {
            agent: {
                "observation": self.observe(agent),
                "action_mask": self.initialise_action_mask()
            } for agent in self.agents
        }

        # draw 2 cards for first agent
        self.players[self.agent_selection].drawTwo()

        # set action mask for first agent
        self.observations[self.agent_selection]["action_mask"] = self.set_action_mask_action_ID(self.players[self.agent_selection],self.observations[self.agent_selection]["action_mask"])
        
        return self.observations, self.infos

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        # extract useful values
        agent = self.agent_selection
        player = self.players[agent]

        action_mask = self.initialise_action_mask()

        if self.action_context["action"] == -1:
            # action chosen → choose hand card
            action_ID = action["action_ID"]
            self.action_context["action"] = np.int8(action_ID)
            
            if action_ID == 0:      
                # skip
                self.action_context["decision"] = -1
            elif action_ID == 1:    
                # move property → unmask property_card action
                self.set_action_mask_property_card(player, action_mask)
                self.action_context["decision"] = 2
            else:                   
                # the rest → unmask hand_card action
                self.set_action_mask_hand_card(player, action_mask, action_ID)
                

        elif self.action_context["action"] > -1:
            # hand card chosen → choose opponent, property or set based on decision stage
            action_ID = self.action_context["action"]

            if self.action_context["decision"] == -1:
                # end of turn, perform actions and change state accordingly

                if action_ID == 0:
                    # skip
                    pass
                elif action_ID == 1:
                    # move property
                    my_property = self.action_context["my_property"]
                    pCard = self.get_property_card_from_id(my_property["colour"],my_property["set_index"],my_property["card"])
                    my_set = self.action_context["my_set"]
                    
                    player.removeProperty(my_property["colour"],my_property["set_index"],pCard)
                    player.addProperty(my_set["colour"],my_set["set_index"],pCard)
                elif action_ID == 2:
                    # play money
                    hand_card = action["hand_card"]
                    mCard = self.get_hand_card_from_id(hand_card)

                    player.removeHandCard(mCard)
                    player.addMoney(mCard)
                elif action_ID == 3 or action_ID == 4:
                    # play property or wild property
                    hand_card = action["hand_card"]
                    pCard =  self.get_hand_card_from_id(hand_card)
                    my_set = self.action_context["my_set"]

                    player.addProperty(my_set["colour"],my_set["set_index"],pCard)
                elif action_ID == 5:
                    # sly deal
                    opponent = self.players[self.agents[self.action_context["opponent"]]]
                    opponent_property = self.action_context["opponent_property"]
                    pCard = self.get_property_card_from_id(opponent_property["colour"],opponent_property["set_index"],opponent_property["card"])
                    my_set = self.action_context["my_set"]

                    opponent.removeProperty(opponent_property["colour"],opponent_property["set_index"],pCard)
                    player.addProperty(my_set["colour"],my_set["set_index"],pCard)
                elif action_ID == 6:
                    # forced deal
                    opponent = self.players[self.agents[self.action_context["opponent"]]]
                    opponent_property = self.action_context["opponent_property"]
                    opppCard = self.get_property_card_from_id(opponent_property["colour"],opponent_property["set_index"],opponent_property["card"])
                    my_property = self.action_context["my_property"]
                    mypCard = self.get_property_card_from_id(my_property["colour"],my_property["set_index"],my_property["card"])
                    my_set = self.action_context["my_set"]

                    opponent.removeProperty(opponent_property["colour"],opponent_property["set_index"],opppCard)
                    player.removeProperty(my_property["colour"],my_property["set_index"],mypCard)
                    player.addProperty(my_set["colour"],my_set["set_index"],opppCard)
                    
                    # TODO: opponent needs to choose set to add his new property to
                    # will be part of logic for rebuttals (JSN)
                elif action_ID == 7:
                    # debt collector
                    # TODO: opponent needs to choose cards worth $5M
                    pass
                elif action_ID == 8:
                    # it's my birthday
                    # TODO: each opponent gives $2M
                    pass
                elif action_ID == 9:
                    # deal breaker
                    opponent = self.players[self.agents[self.action_context["opponent"]]]
                    opponent_set = self.action_context["opponent_set"]

                    pSet_taken = opponent.removeSet(opponent_set["colour"],opponent_set["set_index"])

                    for pSet in player.sets[opponent_set["colour"]]:
                        if pSet.isEmpty():
                            pSet = pSet_taken
                            replaced = True
                    if not replaced:
                        player.sets[opponent_set["colour"]].append(pSet_taken)
                elif action_ID > 9 and action_ID < 15:
                    # rent
                    my_set = self.action_context["my_set"]
                    # TODO: each opponent gives money based on rent
                elif action_ID == 15:
                    # wild rent
                    my_set = self.action_context["my_set"]
                    opponent = self.players[self.agents[self.action_context["opponent"]]]

                    # TODO: opponent gives money based on rent
                elif action_ID == 16:
                    # just say no

                    # TODO: implement rebuttal stuff  
                    pass
            elif self.action_context["decision"] == 0:
                # choose opponent
                self.set_action_mask_opponent(agent, action_mask)
            elif self.action_context["decision"] == 1:
                # choose opponent property
                opponent = self.action_context["opponent"]
                self.set_action_mask_property_card(player, action_mask)
            elif self.action_context["decision"] == 2:
                # choose my property
                self.set_action_mask_property_card(player, action_mask)
            elif self.action_context["decision"] == 3:
                # choose set
                self.set_action_mask_set_index(player, action_mask, )

        if self.render_mode == "human":
            self.render()

    def observe(self,agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """

        player = self.players[agent]

        # Observe hand
        hand = np.zeros((self.num_unique_cards), dtype=np.int8)
        for card in player.hand:
            hand[card.id] += 1

        # Observe properties
        property = {}
        for colour,size in self.property_set_number.items():
            property[colour] = {}
            cards = np.full((self.max_sets_per_property,size), -1, dtype=np.int8)
            full_set = np.zeros((self.max_sets_per_property), dtype=np.int8)

            for pind,pSet in enumerate(player.sets[colour]):
                for cind,card in enumerate(pSet.properties):
                    cards[pind,cind] = card.id
                full_set[pind] = pSet.isCompleted()

            property[colour]["cards"] = cards
            property[colour]["full_set"] = full_set
        
        # Observe money
        money = np.zeros((self.num_unique_money), dtype=np.int8)
        for card in player.money:
            money[card.id-34] += 1  # -34 to make 1M index go to 0

        # Observe opponent properties
        opponent_property = {}
        opponents = [a for a in self.agents if a != agent]
        for colour,size in self.property_set_number.items():
            opponent_property[colour] = {}
            cards = np.full((self.num_opponents,self.max_sets_per_property,size), -1, dtype=np.int8)
            full_set = np.zeros((self.num_opponents,self.max_sets_per_property), dtype=np.int8)
            
            for oind,opponent in enumerate(opponents):
                for pind,pSet in enumerate(self.players[opponent].sets[colour]):
                    for cind,card in enumerate(pSet.properties):
                        cards[oind,pind,cind] = card.id
                    full_set[oind,pind] = pSet.isCompleted()
                
            opponent_property[colour]["cards"] = cards
            opponent_property[colour]["full_set"] = full_set
        
        # Observe opponent money
        opponent_money = np.zeros((self.num_opponents,self.num_unique_money), dtype=np.int8)
        for oind,opponent in enumerate(opponents):
            for card in self.players[opponent].money:
                opponent_money[oind,card.id-34] += 1
        
        # Observe actions left
        actions_left = self.actions_left[agent]

        # Observe discard pile
        discard_pile = np.zeros((self.num_unique_cards), dtype=np.int8)
        for card in self.deck.discard_pile:
            discard_pile[card.id] += 1
        
        # Observe action context
        action_context = self.action_context

        observation = {
            "hand": hand,
            "property": property,
            "money": money,
            "opponent_property": opponent_property,
            "opponent_money": opponent_money,
            "actions_left": actions_left,
            "discard_pile": discard_pile,
            "action_context": action_context
        }
        return observation
        
    def reset_action_context(self):
        self.action_context = {
            "action": np.int8(-1),
            "decision": np.int8(-1),
            "opponent": np.int8(-1),
            "opponent_property": {
                "colour": np.int8(-1),
                "set_index": np.int8(-1),
                "card": np.int8(-1)
            },
            "opponent_set": {
                "colour": np.int8(-1),
                "set_index": np.int8(-1)
            },
            "my_property": {
                "colour": np.int8(-1),
                "set_index": np.int8(-1),
                "card": np.int8(-1)
            },
            "my_set": {
                "colour": np.int8(-1),
                "set_index": np.int8(-1)
            }
        }

    def initialise_action_mask(self):
        action_mask = {
            "action_ID": np.zeros(self.num_actions, dtype=np.int8),
            "hand_card": np.zeros(self.num_unique_cards, dtype=np.int8),
            "opponent": np.zeros(self.num_opponents, dtype=np.int8),
            "property_card": {
                "colour": np.zeros(self.num_unique_colours, dtype=np.int8),
                "set_index": np.zeros(self.max_sets_per_property, dtype=np.int8),
                "card": np.zeros(self.num_unique_property_cards, dtype=np.int8)
            },
            "set": {
                "colour": np.zeros(self.num_unique_colours, dtype=np.int8),
                "set_index": np.zeros(self.max_sets_per_property, dtype=np.int8)
            }
        }

        return action_mask

    def set_action_mask_action_ID(self, player, action_mask):
        # set action mask based on cards in hand

        # can always skip
        action_mask["action_ID"][0] = 1

        # move property, at least 1 property on the board (can be wild)
        action_mask["action_ID"][1] = player.hasAtLeastOnePropertyOnBoard()
        
        # play money
        action_mask["action_ID"][2] = player.hasMoneyInHand()

        # play property
        action_mask["action_ID"][3] = player.hasPropertyInHand()

        # play wild property, at least 1 property on the board (cannot be wild)
        action_mask["action_ID"][4] = player.hasWildPropertyInHand() and player.hasAtLeastOneNonWildPropertyOnBoard()

        # sly deal, at least one opponent must have at least 1 property on the board
        if player.hasSlyDeal():
            for opponent in self.players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOnePropertyOnBoard():
                    action_mask["action_ID"][5] == 1
                    break
        
        # forced deal, at least one opponent AND player must have at least 1 property on the board
        if player.hasForcedDeal():
            for opponent in self.players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOnePropertyOnBoard():
                    action_mask["action_ID"][6] == 1
                    break
        
        # debt collector, at least one opponent must have >0 money on the board
        if player.hasDebtCollector():
            for opponent in self.players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOneMoneyOnBoard():
                    action_mask["action_ID"][7] == 1
                    break

        # its my birthday, at least one opponent must have >0 money on the board
        if player.hasItsMyBirthday():
            for opponent in self.players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOneMoneyOnBoard():
                    action_mask["action_ID"][8] == 1
                    break

        # deal breaker, at least one opponent must have at least one set on the board
        if player.hasDealBreaker():
            for opponent in self.players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOneSetOnBoard():
                    action_mask["action_ID"][9] == 1
                    break
        
        # rent, at least one property of that colour AND at least one opponent with >0 money
        for opponent in self.players.values():
            if opponent == player:
                continue
            if opponent.hasAtLeastOneMoneyOnBoard():
                pColours = player.whichColoursOnBoard()
                rColours = player.whichRentColoursinHand()

                validColours = pColours.intersect(rColours)

                if "Red" in validColours:
                    action_mask["action_ID"][10] == 1
                if "Green" in validColours:
                    action_mask["action_ID"][11] == 1
                if "Pink" in validColours:
                    action_mask["action_ID"][12] == 1
                if "Black" in validColours:
                    action_mask["action_ID"][13] == 1
                if "Brown" in validColours:
                    action_mask["action_ID"][14] == 1
                if "Wild" in rColours:
                    action_mask["action_ID"][15] == 1
        
        # just say no, masked 
        action_mask["action_ID"][16] = 0

        return action_mask
        
    def set_action_mask_property_card(self, player, action_mask):
        # set action mask based on agents properties on the board

        # get indices of properties held by player
        for cind,pSets in enumerate(player.sets.values()):
            for pind,pSet in enumerate(pSets):
                if not pSet.isEmpty() and not pSet.isOnlyWild():
                    for card in pSet.properties:
                        action_mask["property_card"]["card"][card.ID] = 1
                    action_mask["property_card"]["set_index"][pind] = 1
                    action_mask["property_card"]["colour"][cind] = 1
                    break
        
    def set_action_mask_hand_card(self, player, action_mask, action_ID):
        # set action mask based on cards in hand and action_ID
    
        if action_ID == 2:      # money
            for card in player.hand:
                if isinstance(card,MoneyCard):
                    action_mask["hand_card"][card.id] = 1
            self.action_context["decision"] = -1

        elif action_ID == 3:    # property
            for card in player.hand:
                if isinstance(card,PropertyCard):
                    action_mask["hand_card"][card.id] = 1

            self.action_context["decision"] = -1

        elif action_ID == 4:    # wild property
            action_mask["hand_card"][17] = 1
            self.action_context["decision"] = -1

        elif action_ID == 5:    # sly deal
            action_mask["hand_card"][23] = 1
            self.action_context["decision"] = 0

        elif action_ID == 6:    # forced deal
            action_mask["hand_card"][21] = 1
            self.action_context["decision"] = 0

        elif action_ID == 7:    # debt collector
            action_mask["hand_card"][22] = 1
            self.action_context["decision"] = 0

        elif action_ID == 8:    # its my birthday
            action_mask["hand_card"][24] = 1
            self.action_context["decision"] = -1

        elif action_ID == 9:    # deal breaker
            action_mask["hand_card"][26] = 1
            self.action_context["decision"] = 0

        elif action_ID == 10:    # rent, red/yellow
            action_mask["hand_card"][28] = 1
            self.action_context["decision"] = 3

        elif action_ID == 11:    # rent, green/dark blue
            action_mask["hand_card"][29] = 1
            self.action_context["decision"] = 3

        elif action_ID == 10:    # rent, pink/orange
            action_mask["hand_card"][30] = 1
            self.action_context["decision"] = 3

        elif action_ID == 10:    # rent, black/light green
            action_mask["hand_card"][31] = 1
            self.action_context["decision"] = 3

        elif action_ID == 10:    # rent, brown/light blue
            action_mask["hand_card"][32] = 1
            self.action_context["decision"] = 3

        elif action_ID == 10:    # rent, wild
            action_mask["hand_card"][33] = 1
            self.action_context["decision"] = 3

    def set_action_mask_opponent(self, agent, action_mask):
        action_mask["opponent"] = np.ones(self.num_opponents, dtype=np.int8)
        action_mask["opponent"][self.agents.index(agent)] = 0

    def set_action_mask_set_index(self, player, action_mask):
        # TODO: need a separate set_index action mask depending on card played
        for cind,pSets in enumerate(player.sets.values()):
            for pind,pSet in enumerate(pSets):
                if not pSet.isEmpty() and not pSet.isOnlyWild():
                    for card in pSet.properties:
                        action_mask["property_card"]["card"][card.ID] = 1
                    action_mask["property_card"]["set_index"][pind] = 1
                    action_mask["property_card"]["colour"][cind] = 1
                    break
                 
    def get_hand_card_from_id(self, player, ID):
        IDs = [card.ID for card in player.hand]
        index = IDs.index(ID)
        card = copy.deepcopy(player.hand[index])
        return card

    def get_property_card_from_id(self, player, colour, set_index, ID):
        pSets = player.sets[colour]
        pSet = pSets[set_index]
        IDs = [card.ID for card in pSet.properties]
        index = IDs.index(ID)
        card = copy.deepcopy(pSet.properties[index])
        return card




    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """


    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass