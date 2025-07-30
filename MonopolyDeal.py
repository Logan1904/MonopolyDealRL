import functools

import gymnasium as gym
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from Deck import *
from Player import *
from Card import *
from Render import *
from ActionMask import *
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
        
        self.possible_agents = ["player_" + str(r) for r in range(NUM_PLAYERS)]

        # a mapping between agent name and ID
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        self.renderer = Render()

    # Observation space should be defined here.
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Define observation space
        """
        
        return gym.spaces.Dict({
            "hand": gym.spaces.Box(low=0, high=MAX_ANY_CARD, shape=(NUM_UNIQUE_CARDS,), dtype=np.int8),
            "property": gym.spaces.Dict({
                colour: gym.spaces.Dict({
                    "cards": gym.spaces.Box(low=-1, high=NUM_UNIQUE_PROPERTY_CARDS, shape=(MAX_SETS_PER_PROPERTY,max_cards), dtype=np.int8),
                    "full_set": gym.spaces.MultiBinary(MAX_SETS_PER_PROPERTY)
                }) for colour,max_cards in SET_LENGTH.items()
            }),
            "money": gym.spaces.Box(low=0, high=MAX_ANY_CARD, shape=(NUM_UNIQUE_CARDS,), dtype=np.int8),
            "opponent_property": gym.spaces.Dict({
                colour: gym.spaces.Dict({
                    "cards": gym.spaces.Box(low=-1, high=NUM_UNIQUE_PROPERTY_CARDS, shape=(MAX_SETS_PER_PROPERTY,max_cards,NUM_OPPONENTS), dtype=np.int8),
                    "full_set": gym.spaces.MultiBinary([MAX_SETS_PER_PROPERTY,NUM_OPPONENTS])
                }) for colour,max_cards in SET_LENGTH.items()
            }),
            "opponent_money": gym.spaces.Box(low=0, high=MAX_ANY_CARD, shape=(NUM_UNIQUE_CARDS,NUM_OPPONENTS), dtype=np.int8),
            "actions_left": gym.spaces.Discrete(4),
            "discard_pile": gym.spaces.Box(low=0, high=MAX_ANY_CARD, shape=(NUM_UNIQUE_CARDS,), dtype=np.int8),
            "action_context": gym.spaces.Dict({
                "decision": gym.spaces.Discrete(MAX_DECISIONS+2, start=-1),
                "action": gym.spaces.Discrete(NUM_ACTIONS+1, start=-1),
                "hand_card": gym.spaces.Discrete(NUM_UNIQUE_CARDS+1, start=-1),
                "target_ID": gym.spaces.Discrete(NUM_PLAYERS+1, start=-1),
                "opponent_ID": gym.spaces.Discrete(NUM_OPPONENTS+1, start=-1),
                "opponent_property": gym.spaces.Dict({
                    "colour": gym.spaces.Discrete(NUM_UNIQUE_COLOURS+1, start=-1),
                    "set_index": gym.spaces.Discrete(MAX_SETS_PER_PROPERTY+1, start=-1),
                    "card": gym.spaces.Discrete(NUM_UNIQUE_PROPERTY_CARDS+1, start=-1)
                }),
                "opponent_set": gym.spaces.Dict({
                    "colour": gym.spaces.Discrete(NUM_UNIQUE_COLOURS+1, start=-1),
                    "set_index": gym.spaces.Discrete(MAX_SETS_PER_PROPERTY+1, start=-1)
                }),
                "my_property": gym.spaces.Dict({
                    "colour": gym.spaces.Discrete(NUM_UNIQUE_COLOURS+1, start=-1),
                    "set_index": gym.spaces.Discrete(MAX_SETS_PER_PROPERTY+1, start=-1),
                    "card": gym.spaces.Discrete(NUM_UNIQUE_PROPERTY_CARDS+1, start=-1)
                }),
                "my_set": gym.spaces.Dict({
                    "colour": gym.spaces.Discrete(NUM_UNIQUE_COLOURS+1, start=-1),
                    "set_index": gym.spaces.Discrete(MAX_SETS_PER_PROPERTY+1, start=-1)
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
                "action_ID": gym.spaces.Discrete(NUM_ACTIONS),                         # Choose an action
                "hand_card": gym.spaces.Discrete(NUM_UNIQUE_CARDS),                    # Choose a card from hand
                "opponent_ID": gym.spaces.Discrete(NUM_OPPONENTS+1),                      # Choose an opponent
                "property_card": gym.spaces.Dict({                                          # Choose a card
                    "colour": gym.spaces.Discrete(NUM_UNIQUE_COLOURS),
                    "set_index": gym.spaces.Discrete(MAX_SETS_PER_PROPERTY),
                    "card": gym.spaces.Discrete(NUM_UNIQUE_PROPERTY_CARDS)
                }),
                "set": gym.spaces.Dict({                                              # Choose a set
                    "colour": gym.spaces.Discrete(NUM_UNIQUE_COLOURS),
                    "set_index": gym.spaces.Discrete(MAX_SETS_PER_PROPERTY)
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
        self.action_context = self.reset_action_context()

        # initialise observation dictionary
        self.observations = {agent: {"observation": None, "action_mask": None} for agent in self.agents} 
        self.observations = {
            agent: {
                "observation": self.observe(agent),
                "action_mask": ActionMask().action_mask
            } for agent in self.agents
        }

        # draw 2 cards for first agent
        player = self.players[self.agent_selection]
        player.drawTwo()

        # set action mask for first agent
        action_mask = ActionMask()
        action_mask.set_action_ID(self._get_internal_state())
        self.observations[self.agent_selection]["action_mask"] = action_mask.action_mask
        
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

        action_mask = ActionMask()

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
                action_mask.set_property_colour(self._get_internal_state(), target_opponent=False)
            else:                   
                # the rest → unmask hand_card action
                self.action_context["decision"] = 0
                action_mask.set_hand_card(self._get_internal_state())

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
                action_mask.set_set_colour(self._get_internal_state(), target_opponent=False)
            elif action_ID == 5 or action_ID == 6 or action_ID == 7 or action_ID == 8 or action_ID == 9:
                # sly deal or forced deal or debt collector or it's my birthday or deal breaker → choose opponent
                self.action_context["decision"] = 1
                # unmask opponents
                action_mask.set_opponent(self._get_internal_state())
            
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
                action_mask.set_property_colour(self._get_internal_state(), target_opponent=True)
            elif action_ID == 7 or action_ID == 8 or action_ID == 15:
                # debt collector, it's my birthday or wild rent → end of turn
                self.action_context["decision"] = 7
            elif action_ID == 9:
                # deal breaker → choose (opponent) set colour
                self.action_context["decision"] = 5
                self.action_context["target"] = opponent_ID
                # unmask (opponent) sets
                action_mask.set_set_colour(self._get_internal_state(), target_opponent=True)

        elif decision == 2:
            # property colour chosen → choose property set index
            property_colour = action["property_card"]["colour"]
            if self.action_context["target_ID"] == self.action_context["opponent_ID"]:
                self.action_context["opponent_property"]["colour"] = property_colour
                target_opponent = True
            else:
                self.action_context["my_property"]["colour"] = property_colour
                target_opponent = False

            # next decision is always 3
            self.action_context["decision"] = 3

            # unmask my property set index
            action_mask.set_property_set_index(self._get_internal_state(), target_opponent)

        elif decision == 3:
            # property set index chosen → choose property card
            property_set_index = action["property_card"]["set_index"]
            if self.action_context["target_ID"] == self.action_context["opponent_ID"]:
                self.action_context["opponent_property"]["set_index"] = property_set_index
                target_opponent = True
            else:
                self.action_context["my_property"]["set_index"] = property_set_index
                target_opponent = False

            # next decision is always 4
            self.action_context["decision"] = 4

            # unmask my property set index
            action_mask.set_property_card(self._get_internal_state(), target_opponent)

        elif decision == 4:
            # property card chosen
            property_card = action["property_card"]["card"]
            if self.action_context["target_ID"] == self.action_context["opponent_ID"]:
                self.action_context["opponent_property"]["card"] = property_card
            else:
                self.action_context["my_property"]["card"] = property_card
            
            if action_ID == 1 or action_ID == 5 or (action_ID == 6 and self.action_context["my_property"]["colour"] != -1):
                # move property, sly deal → choose (my) set colour
                self.action_context["decision"] = 5
                self.action_context["target_ID"] = self.agents.index(agent)
                # unmask (my) sets
                action_mask.set_set_colour(self._get_internal_state(), target_opponent=False)
            elif action_ID == 6 and self.action_context["my_property"]["colour"] == -1:
                # forced deal → choose (my) property colour
                self.action_context["decision"] = 2
                self.action_context["target_ID"] = self.agents.index(agent)
                # unmask (my) properties
                action_mask.set_property_colour(self._get_internal_state(), target_opponent=False)

        elif decision == 5:
            # set colour chosen
            set_colour = action["set"]["colour"]
            if self.action_context["target_ID"] == self.action_context["opponent_ID"]:
                self.action_context["opponent_set"]["colour"] = set_colour
                target_opponent = True
            else:
                self.action_context["my_set"]["colour"] = set_colour
                target_opponent = False

            # next decision is always 6
            self.action_context["decision"] = 6

            # unmask set index
            colour = decode_colour(set_colour)
            action_mask.set_set_index(self._get_internal_state(), target_opponent)

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
                action_mask.set_opponent(self._get_internal_state())

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
                p_colour = decode_colour(my_property["colour"])
                s_colour = decode_colour(my_set["colour"])

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
                s_colour = decode_colour(my_set["colour"])

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
                p_colour = decode_colour(opponent_property["colour"])
                s_colour = decode_colour(my_set["colour"])

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
                p_colour_opp = decode_colour(opponent_property["colour"])
                s_colour_mine = decode_colour(my_set["colour"])

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
                s_colour = decode_colour(opponent_set["colour"])

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
            self.action_context = self.reset_action_context()
            # Unmask valid actions
            action_mask = ActionMask()
            action_mask.set_action_ID(self._get_internal_state())
            
        # Update action mask
        self.observations[agent]["action_mask"] = action_mask.action_mask
        
    def observe(self,agent):
        """
        Observe the internal state representation to the gymnasium observation space
        """

        player = self.players[agent]

        # Observe hand
        hand = np.zeros((NUM_UNIQUE_CARDS), dtype=np.int8)
        for card in player.hand:
            hand[card.id] += 1

        # Observe properties
        property = {}
        for colour,size in SET_LENGTH.items():
            property[colour] = {}
            cards = np.full((MAX_SETS_PER_PROPERTY,size), -1, dtype=np.int8)
            full_set = np.zeros((MAX_SETS_PER_PROPERTY), dtype=np.int8)

            for pind,pSet in enumerate(player.sets[colour]):
                for cind,card in enumerate(pSet.properties):
                    cards[pind,cind] = card.id
                full_set[pind] = pSet.isCompleted()

            property[colour]["cards"] = cards
            property[colour]["full_set"] = full_set
        
        # Observe money
        money = np.zeros((NUM_UNIQUE_CARDS), dtype=np.int8)
        for card in player.money:
            money[card.id] += 1

        # Observe opponent properties
        opponent_property = {}
        opponents = [a for a in self.agents if a != agent]
        for colour,size in SET_LENGTH.items():
            opponent_property[colour] = {}
            cards = np.full((NUM_OPPONENTS,MAX_SETS_PER_PROPERTY,size), -1, dtype=np.int8)
            full_set = np.zeros((NUM_OPPONENTS,MAX_SETS_PER_PROPERTY), dtype=np.int8)
            
            for oind,opponent in enumerate(opponents):
                for pind,pSet in enumerate(self.players[opponent].sets[colour]):
                    for cind,card in enumerate(pSet.properties):
                        cards[oind,pind,cind] = card.id
                    full_set[oind,pind] = pSet.isCompleted()
                
            opponent_property[colour]["cards"] = cards
            opponent_property[colour]["full_set"] = full_set
        
        # Observe opponent money
        opponent_money = np.zeros((NUM_OPPONENTS,NUM_UNIQUE_CARDS), dtype=np.int8)
        for oind,opponent in enumerate(opponents):
            for card in self.players[opponent].money:
                opponent_money[oind,card.id] += 1
        
        # Observe actions left
        actions_left = self.actions_left[agent]

        # Observe discard pile
        discard_pile = np.zeros((NUM_UNIQUE_CARDS), dtype=np.int8)
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
        action_context = {
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

        return action_context
                        
    def render(self, mode):
        self.renderer.render(mode, self._get_internal_state())
    
    def _get_internal_state(self):
        return self.players, self.agents, self.agent_selection, self.deck, self.action_context