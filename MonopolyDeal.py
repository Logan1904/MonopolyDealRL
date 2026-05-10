import copy
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

# Action IDs that may be cancelled by the defender via Just Say No.
AGGRESSIVE_ACTIONS = frozenset({5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})

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

        # Cross-player action in flight (e.g. rent, JSN, forced-deal placement).
        # None during normal attacker turns. While set, agent_selection has been
        # temporarily overridden to a defender; the attacker resumes once
        # _advance_or_return_to_attacker() drains the defender list.
        self.pending = None

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
                self.action_context["target_ID"] = opponent_ID
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

            # render pre action state
            if self.actions_left[agent] == 3:
                self.render(mode='pre')

            # Render the action description NOW, before resolving. The resolve
            # for offensive actions may yield control to a defender (which
            # wipes action_context), and finalize is what would otherwise own
            # this render.
            self.render(mode='action')

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
            elif action_ID in AGGRESSIVE_ACTIONS:
                # Aggressive actions (sly/forced deal, debt, birthday, deal
                # breaker, rent, wild rent) all share the same shape: discard
                # the action card up front (consumed regardless of JSN
                # outcome), then either yield to the defender for a JSN check
                # or apply the effect directly.
                hand_card = self.action_context["hand_card"]
                self.deck.discardCard(player.removeHandCardById(hand_card))

                defender = self._jsn_defender_for(action_ID)
                if defender is not None and self.players[defender].hasJustSayNo():
                    action_mask = self._start_jsn_check(agent, defender, action_ID)
                else:
                    result = self._apply_aggressive_effect(action_ID)
                    if result is not None:
                        action_mask = result

            # If the action triggered a defender phase (rent, JSN, forced-deal
            # placement, etc.), control has been yielded to the defender via
            # _yield_to_defender() and pending is set. Skip finalize; it will
            # run once the defender drain completes.
            if self.pending is None:
                action_mask = self._finalize_attacker_action()

        elif decision == 8:
            # discard card just chosen

            # remove card from hand
            hand_card = action["hand_card"]
            self.action_context["hand_card"] = hand_card

            self.render(mode='discard')

            card = player.removeHandCardById(hand_card)

            # add to discard pile
            self.deck.discardCard(card)

            if len(player.hand) > 7:
                action_mask.initialise_action_mask()
                action_mask.set_hand_card_discard(self._get_internal_state())
            else:
                self.action_context["decision"] = 9                

        elif decision == 9:
            # render post action state
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

        elif decision == DECISION_DEFENDER_PAY:
            # Defender chose a money card to hand over. Transfer it, decrement
            # what they still owe, and either continue paying, advance to the
            # next defender, or return control to the attacker.
            card_id = action["hand_card"]
            defender = player  # agent_selection has been overridden to defender
            attacker_player = self.players[self.pending["attacker"]]

            paid_card = None
            for c in defender.money:
                if c.id == card_id:
                    paid_card = c
                    break

            defender.removeMoney(paid_card)
            attacker_player.addMoney(paid_card)

            self.pending["remaining"] -= paid_card.value

            if self.pending["remaining"] <= 0 or not defender.money:
                action_mask = self._advance_or_return_to_attacker()
            else:
                # Keep paying — refresh the defender's mask.
                action_mask = ActionMask()
                action_mask.set_defender_phase(self._get_internal_state(), self.pending)

        elif decision == DECISION_DEFENDER_FORCED_DEAL_PLACE_COLOUR:
            # Defender picked the colour bucket for the incoming property.
            set_colour = action["set"]["colour"]
            self.action_context["my_set"]["colour"] = set_colour
            self.action_context["decision"] = DECISION_DEFENDER_FORCED_DEAL_PLACE_INDEX
            action_mask = ActionMask()
            action_mask.set_defender_phase(self._get_internal_state(), self.pending)

        elif decision == DECISION_DEFENDER_FORCED_DEAL_PLACE_INDEX:
            # Defender picked the set_index. Place the card and hand control
            # back to the attacker.
            set_index = action["set"]["set_index"]
            colour = decode_colour(self.action_context["my_set"]["colour"])
            defender = player
            defender.addProperty(colour, set_index, self.pending["card"])
            action_mask = self._advance_or_return_to_attacker()

        elif decision == DECISION_DEFENDER_JSN:
            # Current chooser (defender on the first iteration, then alternates
            # with the attacker) picks: action_ID == 16 → play Just Say No,
            # anything else (mask only allows 0 = "skip" otherwise) → decline.
            chooser = self.agent_selection
            chooser_player = self.players[chooser]
            chose_jsn = (action["action_ID"] == 16)

            if chose_jsn:
                self.deck.discardCard(chooser_player.removeHandCardById(JSN_CARD_ID))
                self.pending["jsn_count"] += 1
                if chooser == self.pending["attacker"]:
                    next_chooser = self.pending["defender"]
                else:
                    next_chooser = self.pending["attacker"]
                if self.players[next_chooser].hasJustSayNo():
                    action_mask = self._yield_to_defender(next_chooser, DECISION_DEFENDER_JSN)
                else:
                    # Other side has nothing to counter with — chain ends.
                    action_mask = self._end_jsn_chain()
            else:
                action_mask = self._end_jsn_chain()

        elif decision == DECISION_DEFENDER_PAY_DONE:
            # Reserved for future "I'm done paying" sentinel; not used yet.
            action_mask = self._advance_or_return_to_attacker()

        # Update action mask for whoever holds the turn now (may differ from
        # the agent we entered step() with if a defender phase was yielded to
        # or a turn just advanced).
        self.observations[self.agent_selection]["action_mask"] = action_mask.action_mask
        
    def _finalize_attacker_action(self):
        """Run the post-resolution cleanup for the attacker's just-completed action:
        decrement actions_left, reset action_context, and arm the next
        decision (skip → next-action, hand>7 → discard, else → end-of-turn).

        Called both on the normal in-line path (decision 7 with no pending) and
        when the last defender of a pending action finishes. The action-mode
        render fires earlier (in step()) before any defender yield, since
        action_context["action"] is reset by _yield_to_defender.
        """
        agent = self.agent_selection
        player = self.players[agent]

        self.actions_left[agent] -= 1

        self.action_context = self.reset_action_context()

        action_mask = ActionMask()
        action_mask.set_action_ID(self._get_internal_state())

        # Round done
        if self.actions_left[agent] == 0:
            if len(player.hand) > 7:
                # Discard
                self.action_context["decision"] = 8
                action_mask.initialise_action_mask()
                action_mask.set_hand_card_discard(self._get_internal_state())
            else:
                self.action_context["decision"] = 9

        return action_mask

    def _yield_to_defender(self, defender_agent, decision_code):
        """Hand control to a defender for one or more follow-up decisions.

        Caller must have set self.pending = {...} first. The agent_selector's
        cyclic position is intentionally not advanced — _agent_selector.next()
        is only called at decision==9 (post-turn), so as long as control
        returns to the attacker before then, the cycle stays correct.
        """
        self.agent_selection = defender_agent
        self.action_context = self.reset_action_context()
        self.action_context["decision"] = decision_code

        action_mask = ActionMask()
        action_mask.set_defender_phase(self._get_internal_state(), self.pending)
        return action_mask

    def _advance_or_return_to_attacker(self):
        """Called when a defender finishes their decision sequence.

        If more defenders remain in self.pending["defenders"] (e.g. multi-target
        It's My Birthday), yield to the next one. Otherwise clear pending,
        restore the attacker as agent_selection, and run the deferred finalize.

        For PAYMENT-shaped pending actions, also re-arms the per-defender
        amount-owed counter on each yield, and silently skips defenders who
        have nothing to pay with (Monopoly Deal rule: "give what you have").
        """
        if self.pending is None:
            # Defensive: nothing to drain; treat as attacker finalize.
            return self._finalize_attacker_action()

        remaining = self.pending.get("defenders", [])
        while remaining:
            next_defender = remaining.pop(0)

            if self.pending["type"] == "PAYMENT":
                # Re-arm amount owed for this defender. If they have no money
                # to give, skip them entirely.
                # TODO: extend to allow paying with property cards, not just bank money.
                self.pending["remaining"] = self.pending["amount"]
                if not self.players[next_defender].money:
                    continue

            first_decision = self.pending.get("defender_first_decision", DECISION_DEFENDER_JSN)
            return self._yield_to_defender(next_defender, first_decision)

        attacker = self.pending["attacker"]
        self.pending = None
        self.agent_selection = attacker
        return self._finalize_attacker_action()

    def _start_payment(self, attacker, defenders, amount):
        """Initiate a payment-shaped pending action (rent, debt collector,
        birthday). Each defender owes `amount` to the attacker, or all their
        bank money if less. Yields control to the first solvent defender.

        If no defender has anything to give, control returns to the attacker
        immediately.
        """
        self.pending = {
            "type": "PAYMENT",
            "attacker": attacker,
            "defenders": list(defenders),
            "amount": amount,
            "remaining": 0,  # set per-defender by _advance_or_return_to_attacker
            "defender_first_decision": DECISION_DEFENDER_PAY,
        }
        return self._advance_or_return_to_attacker()

    def _jsn_defender_for(self, action_ID):
        """Return the agent name who gets the JSN-or-not decision against this
        aggressive action. Two-player only for now: a single defender either
        the explicit target (sly/forced/debt/dealbreaker/wild-rent) or the
        sole opponent (birthday / coloured rent). Multi-defender JSN (e.g.
        each opponent of Birthday in a 3+ player game getting their own
        opportunity) is an extension point.
        """
        if action_ID in (5, 6, 7, 9, 15):
            return self.agents[self.action_context["opponent_ID"]]
        if action_ID in (8, 10, 11, 12, 13, 14):
            for a in self.agents:
                if a != self.agent_selection:
                    return a
        return None

    def _apply_aggressive_effect(self, action_ID):
        """Apply the actual game-state effect of an aggressive action. The
        action card has already been discarded; this only does the steal /
        swap / payment kickoff. Returns the action_mask of any defender phase
        the effect yielded into (PAYMENT / FORCED_DEAL_PLACEMENT) or None if
        the effect resolved synchronously.
        """
        attacker = self.agent_selection
        player = self.players[attacker]

        if action_ID == 5:
            # sly deal
            opp_name = self.agents[self.action_context["opponent_ID"]]
            opponent = self.players[opp_name]
            opp_prop = self.action_context["opponent_property"]
            my_set = self.action_context["my_set"]
            p_colour = decode_colour(opp_prop["colour"])
            s_colour = decode_colour(my_set["colour"])
            pCard = opponent.removePropertyById(p_colour, opp_prop["set_index"], opp_prop["card"])
            player.addProperty(s_colour, my_set["set_index"], pCard)
            return None

        if action_ID == 6:
            # forced deal — swap then yield to defender for placement
            opp_name = self.agents[self.action_context["opponent_ID"]]
            opponent = self.players[opp_name]
            opp_prop = self.action_context["opponent_property"]
            my_prop = self.action_context["my_property"]
            my_set = self.action_context["my_set"]
            p_colour_opp = decode_colour(opp_prop["colour"])
            p_colour_mine = decode_colour(my_prop["colour"])
            s_colour_mine = decode_colour(my_set["colour"])
            stolen = opponent.removePropertyById(p_colour_opp, opp_prop["set_index"], opp_prop["card"])
            player.addProperty(s_colour_mine, my_set["set_index"], stolen)
            given = player.removePropertyById(p_colour_mine, my_prop["set_index"], my_prop["card"])
            return self._start_forced_deal_placement(attacker, opp_name, given)

        if action_ID == 7:
            # debt collector
            opp_name = self.agents[self.action_context["opponent_ID"]]
            return self._start_payment(attacker, [opp_name], amount=5)

        if action_ID == 8:
            # it's my birthday
            opponents = [a for a in self.agents if a != attacker]
            return self._start_payment(attacker, opponents, amount=2)

        if action_ID == 9:
            # deal breaker
            opp_name = self.agents[self.action_context["opponent_ID"]]
            opponent = self.players[opp_name]
            opp_set = self.action_context["opponent_set"]
            s_colour = decode_colour(opp_set["colour"])
            pSet_taken = opponent.removeSetByID(s_colour, opp_set["set_index"])
            for pind, pSet in enumerate(player.sets[s_colour]):
                if pSet.isEmpty():
                    player.sets[s_colour][pind] = pSet_taken
                    break
            return None

        if 10 <= action_ID <= 14:
            # coloured rent — every opponent pays
            my_set = self.action_context["my_set"]
            s_colour = decode_colour(my_set["colour"])
            rent_amount = player.sets[s_colour][my_set["set_index"]].rentValue()
            opponents = [a for a in self.agents if a != attacker]
            return self._start_payment(attacker, opponents, amount=rent_amount)

        if action_ID == 15:
            # wild rent — single chosen opponent pays
            my_set = self.action_context["my_set"]
            s_colour = decode_colour(my_set["colour"])
            rent_amount = player.sets[s_colour][my_set["set_index"]].rentValue()
            opp_name = self.agents[self.action_context["opponent_ID"]]
            return self._start_payment(attacker, [opp_name], amount=rent_amount)

        return None

    def _start_jsn_check(self, attacker, defender, action_ID):
        """Snapshot action_context and yield to the defender to decide whether
        to play Just Say No. The attacker's action card has already been
        discarded; if the chain ends with the action proceeding we replay the
        effect from the snapshot, otherwise we just finalize.
        """
        self.pending = {
            "type": "JSN_OPPORTUNITY",
            "attacker": attacker,
            "defender": defender,
            "jsn_count": 0,
            "saved_context": copy.deepcopy(self.action_context),
            "original_action_ID": action_ID,
        }
        return self._yield_to_defender(defender, DECISION_DEFENDER_JSN)

    def _end_jsn_chain(self):
        """Called when the JSN chain ends (either side declines, or the next
        player has no JSN to play). Restores attacker context and either
        applies the effect (even count → action proceeds) or finalizes
        without effect (odd count → cancelled). The action card was already
        discarded up-front so cancellation just consumes it.
        """
        saved = self.pending["saved_context"]
        action_ID = self.pending["original_action_ID"]
        attacker = self.pending["attacker"]
        cancelled = (self.pending["jsn_count"] % 2 == 1)

        # Restore attacker turn state.
        self.agent_selection = attacker
        self.action_context = saved
        self.pending = None

        if cancelled:
            return self._finalize_attacker_action()

        result = self._apply_aggressive_effect(action_ID)
        if result is not None:
            # Effect yielded into PAYMENT / FORCED_DEAL_PLACEMENT pending.
            return result
        return self._finalize_attacker_action()

    def _start_forced_deal_placement(self, attacker, defender, card):
        """After a Forced Deal swap, the defender must place the property they
        received from the attacker into one of their own sets. Two decisions:
        colour, then set_index.
        """
        self.pending = {
            "type": "FORCED_DEAL_PLACEMENT",
            "attacker": attacker,
            "defenders": [defender],
            "card": card,
            "defender_first_decision": DECISION_DEFENDER_FORCED_DEAL_PLACE_COLOUR,
        }
        return self._advance_or_return_to_attacker()

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