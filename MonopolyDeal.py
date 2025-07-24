import functools

import gymnasium as gym
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from deck import *
from player import *
from card import *
from mappings import *

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
        self.num_players = 2                        # Number of players including agent
        self.num_opponents = self.num_players - 1

        self.max_hand_size = 13                     # Maximum number of cards in hand (start with 7, play 3 Pass Go's)
        self.max_sets_per_property = 9              # Maximum possible number of sets per property colour
        self.max_any_card = 8                       # Maximum possible number of any card (8 x Pass Go)

        self.num_unique_colours = 10                # Number of unique colours
        self.num_unique_property_cards = 18         # Number of property cards
        self.num_unique_cards = 40                  # Number of unique cards in deck

        self.num_actions = 17                       # Number of actions
        self.max_decisions = 7                      # Number of decisions (choose opponent, choose property, choose set, end-of-turn)

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
                }) for colour,max_cards in SET_LENGTH.items()
            }),
            "money": gym.spaces.Box(low=0, high=self.max_any_card, shape=(self.num_unique_cards,), dtype=np.int8),
            "opponent_property": gym.spaces.Dict({
                colour: gym.spaces.Dict({
                    "cards": gym.spaces.Box(low=-1, high=self.num_unique_property_cards, shape=(self.max_sets_per_property,max_cards,self.num_opponents), dtype=np.int8),
                    "full_set": gym.spaces.MultiBinary([self.max_sets_per_property,self.num_opponents])
                }) for colour,max_cards in SET_LENGTH.items()
            }),
            "opponent_money": gym.spaces.Box(low=0, high=self.max_any_card, shape=(self.num_unique_cards,self.num_opponents), dtype=np.int8),
            "actions_left": gym.spaces.Discrete(4),
            "discard_pile": gym.spaces.Box(low=0, high=self.max_any_card, shape=(self.num_unique_cards,), dtype=np.int8),
            "action_context": gym.spaces.Dict({
                "decision": gym.spaces.Discrete(self.max_decisions+2, start=-1),
                "action": gym.spaces.Discrete(self.num_actions+1, start=-1),
                "hand_card": gym.spaces.Discrete(self.num_unique_cards+1, start=-1),
                "target_ID": gym.spaces.Discrete(self.num_players+1, start=-1),
                "opponent_ID": gym.spaces.Discrete(self.num_opponents+1, start=-1),
                "opponent_property": gym.spaces.Dict({
                    "colour": gym.spaces.Discrete(self.num_unique_colours+1, start=-1),
                    "set_index": gym.spaces.Discrete(self.max_sets_per_property+1, start=-1),
                    "card": gym.spaces.Discrete(self.num_unique_property_cards+1, start=-1)
                }),
                "opponent_set": gym.spaces.Dict({
                    "colour": gym.spaces.Discrete(self.num_unique_colours+1, start=-1),
                    "set_index": gym.spaces.Discrete(self.max_sets_per_property+1, start=-1)
                }),
                "my_property": gym.spaces.Dict({
                    "colour": gym.spaces.Discrete(self.num_unique_colours+1, start=-1),
                    "set_index": gym.spaces.Discrete(self.max_sets_per_property+1, start=-1),
                    "card": gym.spaces.Discrete(self.num_unique_property_cards+1, start=-1)
                }),
                "my_set": gym.spaces.Dict({
                    "colour": gym.spaces.Discrete(self.num_unique_colours+1, start=-1),
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
                "opponent_ID": gym.spaces.Discrete(self.num_opponents+1),                      # Choose an opponent
                "property_card": gym.spaces.Dict({                                          # Choose a card
                    "colour": gym.spaces.Discrete(self.num_unique_colours),
                    "set_index": gym.spaces.Discrete(self.max_sets_per_property),
                    "card": gym.spaces.Discrete(self.num_unique_property_cards)
                }),
                "set": gym.spaces.Dict({                                              # Choose a set
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
        self._agent_selector = agent_selector.agent_selector(self.agents)
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
        self.observations = {agent: {"observation": None, "action_mask": None} for agent in self.agents} 
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
        decision = self.action_context["decision"]
        action_ID = self.action_context["action"]

        action_mask = self.initialise_action_mask()

        if decision == -1:
            # action chosen
            action_ID = action["action_ID"]
            self.action_context["action"] = action_ID
            
            if action_ID == 0:      
                # skip
                self.action_context["decision"] = 7
            elif action_ID == 1:    
                # move property → choose (my) property colour
                self.action_context["decision"] = 2
                self.action_context["target_ID"] = self.agents.index(agent)
                # unmask (my) properties
                self.set_action_mask_property_colour(player, action_mask)
            else:                   
                # the rest → unmask hand_card action
                self.action_context["decision"] = 0
                self.set_action_mask_hand_card(player, action_mask, action_ID)

        elif decision == 0:
            # hand card chosen
            hand_card = action["hand_card"]
            self.action_context["hand_card"] = hand_card

            if action_ID == 2 or action_ID == 16:
                # play money or just say no → end of turn
                self.action_context["decision"] = 7
            elif action_ID == 3 or action_ID == 4 or action_ID == 10 or action_ID == 11 or action_ID == 12 or action_ID == 13 or action_ID == 14 or action_ID == 15:
                # play property, wild property or any rent → choose (my) set
                self.action_context["decision"] = 5
                self.action_context["target_ID"] = self.agents.index(agent)
                # unmask (my) sets
                self.set_action_mask_set_colour(player, action_mask)
            elif action_ID == 5 or action_ID == 6 or action_ID == 7 or action_ID == 8 or action_ID == 9:
                # sly deal or forced deal or debt collector or it's my birthday or deal breaker → choose opponent
                self.action_context["decision"] = 1
                # unmask opponents
                self.set_action_mask_opponent(player, action_mask)
            
        elif decision == 1:
            # opponent chosen
            opponent_ID = action["opponent_ID"]
            self.action_context["opponent_ID"] = opponent_ID
            opponent = self.agents[opponent_ID] 

            if action_ID == 5 or action_ID == 6:
                # sly deal or forced deal → choose (opponent) property colour
                self.action_context["decision"] = 2
                self.action_context["target_ID"] = opponent_ID
                # unmask (opponent) properties
                self.set_action_mask_property_colour(self.players[opponent], action_mask)
            elif action_ID == 7 or action_ID == 8 or action_ID == 15:
                # debt collector, it's my birthday or wild rent → end of turn
                self.action_context["decision"] = 7
            elif action_ID == 9:
                # deal breaker → choose (opponent) set colour
                self.action_context["decision"] = 5
                self.action_context["target"] = opponent_ID
                # unmask (opponent) sets
                self.set_action_mask_set_colour(self.players[opponent], action_mask)

        elif decision == 2:
            # property colour chosen → choose property set index
            property_colour = action["property_card"]["colour"]
            if self.action_context["target_ID"] == self.action_context["opponent_ID"]:
                self.action_context["opponent_property"]["colour"] = property_colour
                target = self.players[self.agents[self.action_context["opponent_ID"]]]
            else:
                self.action_context["my_property"]["colour"] = property_colour
                target = player

            # next decision is always 3
            self.action_context["decision"] = 3

            # unmask my property set index
            colour = self.decode_colour(property_colour)
            self.set_action_mask_property_set_index(target, colour, action_mask)

        elif decision == 3:
            # property set index chosen → choose property card
            property_set_index = action["property_card"]["set_index"]
            if self.action_context["target_ID"] == self.action_context["opponent_ID"]:
                self.action_context["opponent_property"]["set_index"] = property_set_index
                property_colour = self.action_context["opponent_property"]["colour"]
                target = self.players[self.agents[self.action_context["opponent_ID"]]]
            else:
                self.action_context["my_property"]["set_index"] = property_set_index
                property_colour = self.action_context["my_property"]["colour"]
                target = player

            # next decision is always 4
            self.action_context["decision"] = 4

            # unmask my property set index
            colour = self.decode_colour(property_colour)
            self.set_action_mask_property_card(target, colour, property_set_index, action_mask)

        elif decision == 4:
            # property card chosen
            property_card = action["property_card"]["card"]
            if self.action_context["target_ID"] == self.action_context["opponent_ID"]:
                self.action_context["opponent_property"]["card"] = property_card
                target = self.players[self.agents[self.action_context["opponent_ID"]]]
            else:
                self.action_context["my_property"]["card"] = property_card
                target = player
            
            if action_ID == 1 or action_ID == 5 or (action_ID == 6 and self.action_context["my_property"]["colour"] != -1):
                # move property, sly deal → choose (my) set colour
                self.action_context["decision"] = 5
                self.action_context["target_ID"] = self.agents.index(agent)
                # unmask (my) sets
                self.set_action_mask_set_colour(target, action_mask)
            elif action_ID == 6 and self.action_context["my_property"]["colour"] == -1:
                # forced deal → choose (my) property colour
                self.action_context["decision"] = 2
                self.action_context["target_ID"] = self.agents.index(agent)
                # unmask (my) properties
                self.set_action_mask_property_colour(target, action_mask)

        elif decision == 5:
            # set colour chosen
            set_colour = action["set"]["colour"]
            if self.action_context["target_ID"] == self.action_context["opponent_ID"]:
                self.action_context["opponent_set"]["colour"] = set_colour
                target = self.players[self.agents[self.action_context["opponent_ID"]]]
            else:
                self.action_context["my_set"]["colour"] = set_colour
                target = player

            # next decision is always 6
            self.action_context["decision"] = 6

            # unmask set index
            colour = self.decode_colour(set_colour)
            self.set_action_mask_set_index(target, colour, action_mask)

        elif decision == 6:
            # set index just chosen
            set_index = action["set"]["set_index"]
            if self.action_context["target_ID"] == self.action_context["opponent_ID"]:
                self.action_context["opponent_set"]["set_index"] = set_index
                target = self.players[self.agents[self.action_context["opponent_ID"]]]
            else:
                self.action_context["my_set"]["set_index"] = set_index
                target = player

            if action_ID in [1,3,4,5,6,9,10,11,12,13,14]:
                # move property, play property, play wild, sly deal, forced deal, deal breaker, any rent except wild → end of turn
                self.action_context["decision"] = 7
            elif action_ID == 15:
                # wild rent → choose opponent
                self.action_context["decision"] = 1
                # unmask opponents
                self.set_action_mask_opponent(target, action_mask)

        elif decision == 7:
            # end of turn, perform actions and change state

            # render pre-moves state
            if self.actions_left[agent] == 3:
                self.render(mode='pre')

            if action_ID == 0:
                # skip
                pass
            elif action_ID == 1:
                # move property
                my_property = self.action_context["my_property"]
                my_set = self.action_context["my_set"]

                # decode colours
                p_colour = self.decode_colour(my_property["colour"])
                s_colour = self.decode_colour(my_set["colour"])

                # get property
                pCard = player.removePropertyById(p_colour,my_property["set_index"],my_property["card"])

                # add property to new set
                player.addProperty(s_colour,my_set["set_index"],pCard)
            elif action_ID == 2:
                # play money
                hand_card = self.action_context["hand_card"]

                # get money hand card
                mCard = player.removeHandCardById(hand_card)

                # add money to money pile
                player.addMoney(mCard)
            elif action_ID == 3 or action_ID == 4:
                # play property or wild property
                hand_card = self.action_context["hand_card"]
                my_set = self.action_context["my_set"]

                # decode colour
                s_colour = self.decode_colour(my_set["colour"])

                # get property hand card
                pCard =  player.removeHandCardById(hand_card)
                
                # add property to new set
                player.addProperty(s_colour,my_set["set_index"],pCard)
            elif action_ID == 5:
                # sly deal
                opponent = self.players[self.agents[self.action_context["opponent_ID"]]]
                opponent_property = self.action_context["opponent_property"]

                my_set = self.action_context["my_set"]
                my_property = self.action_context["my_property"]

                # decode colours
                p_colour = self.decode_colour(opponent_property["colour"])
                s_colour = self.decode_colour(my_set["colour"])

                # steal property
                pCard = opponent.removePropertyById(p_colour,opponent_property["set_index"],opponent_property["card"])

                # add property to new set
                player.addProperty(s_colour,my_set["set_index"],pCard)

                # remove card from hand
                hand_card = self.action_context["hand_card"]
                card = player.removeHandCardById(hand_card)

                # add to discard pile
                self.deck.discardCard(card)
            elif action_ID == 6:
                # forced deal
                opponent = self.players[self.agents[self.action_context["opponent_ID"]]]
                opponent_property = self.action_context["opponent_property"]
                my_property = self.action_context["my_property"]
                my_set = self.action_context["my_set"]

                # decode colours
                p_colour_opp = self.decode_colour(opponent_property["colour"])
                s_colour_mine = self.decode_colour(my_set["colour"])

                # steal property
                pCard = opponent.removePropertyById(p_colour_opp,opponent_property["set_index"],opponent_property["card"])

                # add property to new set
                player.addProperty(s_colour_mine,my_set["set_index"],pCard)
                
                # TODO: opponent needs to choose set to add his new property to
                # will be part of logic for rebuttals (JSN)
                # will need to pass p_colour_mine as argument

                # remove card from hand
                hand_card = self.action_context["hand_card"]
                card = player.removeHandCardById(hand_card)

                # add to discard pile
                self.deck.discardCard(card)
            elif action_ID == 7:
                # debt collector
                # TODO: opponent needs to choose cards worth $5M

                # remove card from hand
                hand_card = self.action_context["hand_card"]
                card = player.removeHandCardById(hand_card)

                # add to discard pile
                self.deck.discardCard(card)
            elif action_ID == 8:
                # it's my birthday
                # TODO: each opponent gives $2M

                # remove card from hand
                hand_card = self.action_context["hand_card"]
                card = player.removeHandCardById(hand_card)
                
                # add to discard pile
                self.deck.discardCard(card)
            elif action_ID == 9:
                # deal breaker
                opponent = self.players[self.agents[self.action_context["opponent_ID"]]]
                opponent_set = self.action_context["opponent_set"]

                # decode colours
                s_colour = self.decode_colour(opponent_set["colour"])

                # steal set
                pSet_taken = opponent.removeSet(s_colour,opponent_set["set_index"])

                # add to my sets
                for pSet in player.sets[s_colour]:
                    if pSet.isEmpty():
                        pSet = pSet_taken
                        break

                # remove card from hand
                hand_card = self.action_context["hand_card"]
                card = player.removeHandCardById(hand_card)

                # add to discard pile
                self.deck.discardCard(card)
            elif action_ID > 9 and action_ID < 15:
                # rent
                my_set = self.action_context["my_set"]
                # TODO: each opponent gives money based on rent

                # remove card from hand
                hand_card = self.action_context["hand_card"]
                card = player.removeHandCardById(hand_card)

                # add to discard pile
                self.deck.discardCard(card)
            elif action_ID == 15:
                # wild rent
                my_set = self.action_context["my_set"]
                opponent = self.players[self.agents[self.action_context["opponent_ID"]]]

                # TODO: opponent gives money based on rent

                # remove card from hand
                hand_card = self.action_context["hand_card"]
                card = player.removeHandCardById(hand_card)

                # add to discard pile
                self.deck.discardCard(card)
            elif action_ID == 16:
                # just say no

                # TODO: implement rebuttal stuff  

                # remove card from hand
                hand_card = self.action_context["hand_card"]
                card = player.removeHandCardById(hand_card)
                
                # add to discard pile
                self.deck.discardCard(card)
            # Action done
            self.actions_left[agent] -= 1

            # print action
            self.render(mode='action')

            # If round done, next agent
            if self.actions_left[agent] == 0:
                # render pose-moves state
                self.render(mode='post')
                self.actions_left[agent] = 3

                # for next player
                self.agent_selection = self._agent_selector.next()
                agent = self.agent_selection
                player = self.players[agent]

                # draw 2 cards for next player
                player.drawTwo()

            # Reset action context
            self.reset_action_context()
            # Unmask valid actions
            self.initialise_action_mask()
            self.set_action_mask_action_ID(player, action_mask)
            
        # Update action mask
        self.observations[agent]["action_mask"] = action_mask
        
    def observe(self,agent):
        """
        Observe the internal state representation to the gymnasium observation space
        """

        player = self.players[agent]

        # Observe hand
        hand = np.zeros((self.num_unique_cards), dtype=np.int8)
        for card in player.hand:
            hand[card.id] += 1

        # Observe properties
        property = {}
        for colour,size in SET_LENGTH.items():
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
        money = np.zeros((self.num_unique_cards), dtype=np.int8)
        for card in player.money:
            money[card.id] += 1

        # Observe opponent properties
        opponent_property = {}
        opponents = [a for a in self.agents if a != agent]
        for colour,size in SET_LENGTH.items():
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
        opponent_money = np.zeros((self.num_opponents,self.num_unique_cards), dtype=np.int8)
        for oind,opponent in enumerate(opponents):
            for card in self.players[opponent].money:
                opponent_money[oind,card.id] += 1
        
        # Observe actions left
        actions_left = self.actions_left[agent]

        # Observe discard pile
        discard_pile = np.zeros((self.num_unique_cards), dtype=np.int8)
        for card in self.deck.discard_pile:
            discard_pile[card.id] += 1
        
        # Observe action context
        action_context = self.action_context

        self.observations[agent]["observation"] = {
            "hand": hand,
            "property": property,
            "money": money,
            "opponent_property": opponent_property,
            "opponent_money": opponent_money,
            "actions_left": actions_left,
            "discard_pile": discard_pile,
            "action_context": action_context
        }
        return self.observations[agent]
        
    def reset_action_context(self):
        self.action_context = {
            "decision": np.int8(-1),
            "action": np.int8(-1),
            "hand_card": np.int8(-1),
            "target_ID": np.int8(-1),
            "opponent_ID": np.int8(-1),
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
            "opponent_ID": np.zeros(self.num_opponents+1, dtype=np.int8),
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
                    action_mask["action_ID"][5] = 1
                    break
        
        # forced deal, at least one opponent AND player must have at least 1 property on the board
        if player.hasForcedDeal():
            for opponent in self.players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOnePropertyOnBoard():
                    action_mask["action_ID"][6] = 1
                    break
        
        # debt collector, at least one opponent must have >0 money on the board
        if player.hasDebtCollector():
            for opponent in self.players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOneMoneyOnBoard():
                    action_mask["action_ID"][7] = 1
                    break

        # its my birthday, at least one opponent must have >0 money on the board
        if player.hasItsMyBirthday():
            for opponent in self.players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOneMoneyOnBoard():
                    action_mask["action_ID"][8] = 1
                    break

        # deal breaker, at least one opponent must have at least one set on the board
        if player.hasDealBreaker():
            for opponent in self.players.values():
                if opponent == player:
                    continue
                if opponent.hasAtLeastOneSetOnBoard():
                    action_mask["action_ID"][9] = 1
                    break
        
        # rent, at least one property of that colour AND at least one opponent with >0 money
        for opponent in self.players.values():
            if opponent == player:
                continue
            if opponent.hasAtLeastOneMoneyOnBoard():
                pColours = player.whichColoursOnBoard()
                rColours = player.whichRentColoursInHand()

                validColours = pColours.intersection(rColours)

                if "Red" in validColours:
                    action_mask["action_ID"][10] = 1
                if "Green" in validColours:
                    action_mask["action_ID"][11] = 1
                if "Pink" in validColours:
                    action_mask["action_ID"][12] = 1
                if "Black" in validColours:
                    action_mask["action_ID"][13] = 1
                if "Brown" in validColours:
                    action_mask["action_ID"][14] = 1
                if "Wild" in rColours:
                    action_mask["action_ID"][15] = 1
        
        # just say no, masked 
        action_mask["action_ID"][16] = 0

        return action_mask
        
    def set_action_mask_property_colour(self, player, action_mask):
        # set action mask based on agents properties on the board

        # get indices of properties held by player
        for cind,pSets in enumerate(player.sets.values()):
            for pind,pSet in enumerate(pSets):
                if not pSet.isEmpty() and not pSet.isOnlyWild():
                    action_mask["property_card"]["colour"][cind] = 1
                    break

    def set_action_mask_property_set_index(self, player, colour, action_mask):
        # set action mask based on agents properties on the board
        
        # get indices of properties held by player
        for pind,pSet in enumerate(player.sets[colour]):
            if not pSet.isEmpty() and not pSet.isOnlyWild():
                action_mask["property_card"]["set_index"][pind] = 1
                break

    def set_action_mask_property_card(self, player, colour, set_index, action_mask):
        # set action mask based on agents properties on the board

        # get indices of properties held by player
        
        for card in player.sets[colour][set_index].properties:
            action_mask["property_card"]["card"][card.id] = 1
        
    def set_action_mask_hand_card(self, player, action_mask, action_ID):
        # set action mask based on cards in hand and action_ID
    
        if action_ID == 2:      # money
            for card in player.hand:
                action_mask["hand_card"][card.id] = 1

        elif action_ID == 3:    # property
            for card in player.hand:
                if isinstance(card,PropertyCard):
                    action_mask["hand_card"][card.id] = 1

        elif action_ID == 4:    # wild property
            action_mask["hand_card"][17] = 1

        elif action_ID == 5:    # sly deal
            action_mask["hand_card"][23] = 1

        elif action_ID == 6:    # forced deal
            action_mask["hand_card"][21] = 1

        elif action_ID == 7:    # debt collector
            action_mask["hand_card"][22] = 1

        elif action_ID == 8:    # its my birthday
            action_mask["hand_card"][24] = 1

        elif action_ID == 9:    # deal breaker
            action_mask["hand_card"][26] = 1

        elif action_ID == 10:    # rent, red/yellow
            action_mask["hand_card"][28] = 1

        elif action_ID == 11:    # rent, green/dark blue
            action_mask["hand_card"][29] = 1

        elif action_ID == 12:    # rent, pink/orange
            action_mask["hand_card"][30] = 1

        elif action_ID == 13:    # rent, black/light green
            action_mask["hand_card"][31] = 1

        elif action_ID == 14:    # rent, brown/light blue
            action_mask["hand_card"][32] = 1

        elif action_ID == 15:    # rent, wild
            action_mask["hand_card"][33] = 1

    def set_action_mask_opponent(self, agent, action_mask):
        action_mask["opponent_ID"] = np.ones(self.num_opponents+1, dtype=np.int8)
        action_mask["opponent_ID"][self.agents.index(str(agent))] = 0

    def set_action_mask_set_colour(self, player, action_mask):
        # set action mask on set colour

        # playing a property into a set
        if self.action_context["action"] < 9:
            if self.action_context["action"] in [1]:
                # move property
                my_property = self.action_context["my_property"]

                # decode colour
                colour = self.decode_colour(my_property["colour"])

                # get property
                pCard = player.getPropertyById(colour,my_property["set_index"],my_property["card"])
            elif self.action_context["action"] in [3,4]:
                # play property or wild property
                card_ID = self.action_context["hand_card"]
                
                # get property
                pCard = player.getHandCardById(card_ID)
            elif self.action_context["action"] in [5,6]:
                # sly deal or forced deal
                opponent = self.players[self.agents[self.action_context["opponent_ID"]]]
                opponent_property = self.action_context["opponent_property"]

                # decode colour
                colour = self.decode_colour(opponent_property["colour"])

                # get property
                pCard = opponent.getPropertyById(colour,opponent_property["set_index"],opponent_property["card"])

            for cind,(colour,pSets) in enumerate(player.sets.items()):
                for pSet in pSets:
                    if pSet.canAddProperty(pCard):
                        action_mask["set"]["colour"][cind] = 1
                        break
        else:
            if self.action_context["action"] == 9:
                # deal breaker, unmask all full sets
                for cind,(colour,pSets) in enumerate(player.sets.items()):
                    for pSet in pSets:
                        if pSet.isCompleted():
                            action_mask["set"]["colour"][cind] = 1
                            break
            else:
                # rent, unmask based on rent card
                card_ID = self.action_context["hand_card"]
                # get rent card
                rCard = player.getHandCardById(card_ID)

                if rCard.isWild():
                    for cind,(colour,pSets) in enumerate(player.sets.items()):
                        for pSet in pSets:
                            if not pSet.isEmpty():
                                action_mask["set"]["colour"][cind] = 1
                                break
                else:
                    for cind,(colour,pSets) in enumerate(player.sets.items()):
                        if colour not in rCard.colours:
                            continue
                        for pSet in pSets:
                            if not pSet.isEmpty():
                                action_mask["set"]["colour"][cind] = 1
                                break

    def set_action_mask_set_index(self, player, colour, action_mask):
        # set action mask on set index

        # playing a property into a set
        if self.action_context["action"] < 9:
            if self.action_context["action"] in [1]:
                # move property
                my_property = self.action_context["my_property"]

                # decode colour
                colour = self.decode_colour(my_property["colour"])

                # get property
                pCard = player.getPropertyById(colour,my_property["set_index"],my_property["card"])
            elif self.action_context["action"] in [3,4]:
                # play property or wild property
                card_ID = self.action_context["hand_card"]
                
                # get property
                pCard = player.getHandCardById(card_ID)
            elif self.action_context["action"] in [5,6]:
                # sly deal or forced deal
                opponent = self.players[self.agents[self.action_context["opponent_ID"]]]
                opponent_property = self.action_context["opponent_property"]

                # decode colour
                colour = self.decode_colour(opponent_property["colour"])

                # get property
                pCard = opponent.getPropertyById(colour,opponent_property["set_index"],opponent_property["card"])
            
            for cind,(colour,pSets) in enumerate(player.sets.items()):
                for pind,pSet in enumerate(pSets):
                    if pSet.canAddProperty(pCard):
                        action_mask["set"]["set_index"][pind] = 1
        else:
            if self.action_context["action"] == 9:
                # deal breaker, unmask all full sets
                for cind,(colour,pSets) in enumerate(player.sets.items()):
                    for pind,pSet in enumerate(pSets):
                        if pSet.isCompleted():
                            action_mask["set"]["set_index"][pind] = 1
            else:
                # rent, unmask based on rent card
                card_ID = self.action_context["hand_card"]

                # get rent card
                rCard = player.getHandCardById(card_ID)

                if rCard.isWild():
                    for cind,(colour,pSets) in enumerate(player.sets.items()):
                        for pind,pSet in enumerate(pSets):
                            if not pSet.isEmpty():
                                action_mask["set"]["set_index"][pind] = 1
                else:
                    for cind,(colour,pSets) in enumerate(player.sets.items()):
                        if colour not in rCard.colours:
                            continue
                        for pind,pSet in enumerate(pSets):
                            if not pSet.isEmpty():
                                action_mask["set"]["set_index"][pind] = 1
                 
    def decode_colour(self, id):
        return list(colour_mapping.keys())[list(colour_mapping.values()).index(id)]

    def render(self, mode):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """

        agent = self.agent_selection
        player = self.players[self.agent_selection]

        properties_rendered = []
        for colour in player.sets:
            for pSet in player.sets[colour]:
                if not pSet.isEmpty():
                    properties_rendered.insert(0,pSet.properties)

        if mode == 'pre':
            print("-"*75 + str(agent) + "-"*75)
            print("")
            print("Deck Size: " + str(self.deck.deckSize()))
            print("Discard Size: " + str(self.deck.discardSize()))
            print("")
            self.render_list(player.hand, label="Hand")
            self.render_list(player.money, label="Money")
            print("Properties: ")
            self.rich_property_sets(player.sets)
            print("")
        elif mode == 'post':
            print("")
            print("Deck Size: " + str(self.deck.deckSize()))
            print("Discard Size: " + str(self.deck.discardSize()))
            print("")
            self.render_list(player.hand, label="Hand")
            self.render_list(player.money, label="Money")
            print("Properties: ")
            self.rich_property_sets(player.sets)
            print("")
            print("")
        elif mode == 'action':
            print("[Action] " + str(ACTION_DESCRIPTION[self.action_context["action"]]))
            print(self.action_context)

    def rich_property_sets(self, sets):
        console = Console()
        table = Table(show_lines=True)

        table.add_column("PropertySet")
        for colour in sets.keys():
            table.add_column(colour)

        num_pSets = len(next(iter(sets.values())))
        for i in range(num_pSets):
            row = [f"[{i}]"]
            for colour in sets:
                pSet = sets[colour][i]
                if pSet.isEmpty():
                    row.append(Text("--", style="dim"))
                else:
                    cell_text = Text()
                    for card in pSet.properties:
                        cell_text.append(f"{card.name}\n", style=COLOUR_STYLE_MAP[colour])  # newline for vertical stack
                    row.append(cell_text)
                    
            table.add_row(*row)

        console.print(table)

    def render_list(self, cards, label):
        console = Console()

        # Sort cards
        sorted_cards = sorted(cards, key=self.get_card_sort_key)

        # Build styled line
        line = Text(f"{label}: ")
        for card in sorted_cards:
            style = self.get_card_style(card)
            line.append(f"[{card.name}] ", style=style)

        console.print(line)

    def get_card_style(self, card):
        if isinstance(card, MoneyCard):
            return "green"
        elif isinstance(card, RentCard):
            return "white"
        elif isinstance(card, ActionCard):
            return "red"
        elif isinstance(card, PropertyCard):
            return COLOUR_STYLE_MAP[card.colours[0]]
        
    def get_card_sort_key(self, card):
        if isinstance(card, MoneyCard):
            return 0
        elif isinstance(card, RentCard):
            return 1
        elif isinstance(card, ActionCard):
            return 2
        elif isinstance(card, PropertyCard):
            return 3

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass