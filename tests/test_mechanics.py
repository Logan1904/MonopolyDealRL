"""Mechanic-level tests for the Monopoly Deal env.

Each test sets up a deterministic env state (clears hands/money/board,
plants the specific cards needed for the scenario), drives the env via
internal helpers and env.step(), and asserts on post-state.

Run from the repo root:
    python -m unittest discover -s tests -p "test_*.py"
"""
import os
import sys
import unittest

# Make project root importable.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np
from pettingzoo.utils import agent_selector as _agent_selector_mod

from MonopolyDeal import MonopolyDeal
from PropertySet import PropertySet
from Card import PropertyCard, MoneyCard, ActionCard, RentCard
from mappings import (
    DECISION_DEFENDER_PAY,
    DECISION_DEFENDER_JSN,
    DECISION_DEFENDER_FORCED_DEAL_PLACE_COLOUR,
    DECISION_DEFENDER_FORCED_DEAL_PLACE_INDEX,
    JSN_CARD_ID,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Card factories matching cardsdb entries (id, name, value, colours).
def prop(card_id, name, value, colours):
    return PropertyCard(card_id, name, value, colours)


def money(card_id, value):
    return MoneyCard(card_id, f"{value}M", value)


def action_card(card_id, name, value, desc=""):
    return ActionCard(card_id, name, value, desc)


def rent(card_id, name, colours):
    return RentCard(card_id, name, 1, colours)


def jsn():
    return action_card(JSN_CARD_ID, "Just Say No", 4)


def make_action(action_ID=0, hand_card=0, opponent_ID=0,
                property_card=None, set_=None):
    """Construct an action dict matching the env's action space shape."""
    return {
        "action_ID": action_ID,
        "hand_card": hand_card,
        "opponent_ID": opponent_ID,
        "property_card": property_card or {"colour": 0, "set_index": 0, "card": 0},
        "set": set_ or {"colour": 0, "set_index": 0},
    }


def make_env():
    """Fresh env with deterministic agent ordering and empty boards."""
    env = MonopolyDeal(render_mode=None)
    env.reset(seed=0)

    # Force deterministic ordering — reset shuffles agents.
    env.agents = ["player_0", "player_1"]
    env._agent_selector = _agent_selector_mod.agent_selector(env.agents)
    env.agent_selection = env._agent_selector.next()

    # Clear all hands, money, and properties — tests plant exactly what they need.
    for a in env.agents:
        env.players[a].hand = []
        env.players[a].money = []
        for colour, sets in env.players[a].sets.items():
            for s in sets:
                s.properties = []
                s.hasHouse = False
                s.hasHotel = False

    # Reset action context, terminations, rewards.
    env.action_context = env.reset_action_context()
    env.terminations = {a: False for a in env.agents}
    env.truncations = {a: False for a in env.agents}
    env.rewards = {a: 0 for a in env.agents}
    env._cumulative_rewards = {a: 0 for a in env.agents}
    env.pending = None
    env.actions_left = {a: 3 for a in env.agents}

    return env


def stack_set(env, agent, colour, set_index, cards):
    """Stack a property set deterministically."""
    env.players[agent].sets[colour][set_index].properties = list(cards)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class PropertySetTests(unittest.TestCase):
    def test_clear_set_removes_all(self):
        # The old clearSet iterated while modifying and would skip half the
        # elements. Make sure the fix actually empties the set.
        s = PropertySet("Brown", 2)
        s.properties = [prop(1, "Brown", 1, ["Brown"]), prop(1, "Brown", 1, ["Brown"])]
        s.clearSet()
        self.assertEqual(s.properties, [])


class PaymentTests(unittest.TestCase):
    def _drive_payment(self, env, defender, money_card_id):
        """Have the defender hand over one money card by id."""
        action = make_action(hand_card=money_card_id)
        env.step(action)

    def test_debt_collector_transfers_5m(self):
        env = make_env()
        attacker, defender = "player_0", "player_1"
        env.agent_selection = attacker

        # Attacker has Debt Collector in hand. Defender has $5M (single 5M card).
        env.players[attacker].hand = [action_card(22, "Debt Collector", 3)]
        env.players[defender].money = [money(38, 5)]

        # Drive the action via _apply_aggressive_effect (skips JSN since defender has none).
        env.action_context["action"] = 7
        env.action_context["hand_card"] = 22
        env.action_context["opponent_ID"] = env.agents.index(defender)
        # Discard the action card up front (mirrors the decision==7 wrapper).
        env.deck.discardCard(env.players[attacker].removeHandCardById(22))

        env._apply_aggressive_effect(7)

        # Defender now in DECISION_DEFENDER_PAY phase. Pay the 5M card.
        self.assertEqual(env.agent_selection, defender)
        self.assertEqual(env.action_context["decision"], DECISION_DEFENDER_PAY)
        self._drive_payment(env, defender, money_card_id=38)

        # Drained back to attacker; pending cleared. Money transferred.
        self.assertIsNone(env.pending)
        self.assertEqual(len(env.players[defender].money), 0)
        self.assertEqual(sum(c.value for c in env.players[attacker].money), 5)

    def test_birthday_2p_collects_2m(self):
        env = make_env()
        attacker, defender = "player_0", "player_1"
        env.agent_selection = attacker

        env.players[attacker].hand = [action_card(24, "It's My Birthday", 2)]
        env.players[defender].money = [money(35, 2), money(38, 5)]

        env.action_context["action"] = 8
        env.action_context["hand_card"] = 24
        env.deck.discardCard(env.players[attacker].removeHandCardById(24))

        env._apply_aggressive_effect(8)

        # Defender pays cheapest card sufficient to cover 2 — drive with the 2M.
        self.assertEqual(env.agent_selection, defender)
        self._drive_payment(env, defender, money_card_id=35)

        self.assertIsNone(env.pending)
        # Defender had 2M+5M=7, paid 2M → keeps 5M. Attacker now has 2M.
        self.assertEqual(sum(c.value for c in env.players[defender].money), 5)
        self.assertEqual(sum(c.value for c in env.players[attacker].money), 2)

    def test_payment_skips_broke_defender(self):
        # Birthday with a defender who has zero money: control should return
        # to the attacker without an env.step call.
        env = make_env()
        attacker, defender = "player_0", "player_1"
        env.agent_selection = attacker

        env.players[attacker].hand = [action_card(24, "It's My Birthday", 2)]
        # defender.money intentionally empty

        env.action_context["action"] = 8
        env.action_context["hand_card"] = 24
        env.deck.discardCard(env.players[attacker].removeHandCardById(24))
        env._apply_aggressive_effect(8)

        # Should immediately return to attacker and finalize.
        self.assertIsNone(env.pending)
        self.assertEqual(env.agent_selection, attacker)
        self.assertEqual(env.actions_left[attacker], 2)  # one action consumed

    def test_rent_uses_rentvalue(self):
        env = make_env()
        attacker, defender = "player_0", "player_1"
        env.agent_selection = attacker

        # Attacker has 1 Brown property and the Brown/Light Blue rent card.
        # Brown rent = len(properties) = 1.
        stack_set(env, attacker, "Brown", 0, [prop(1, "Brown", 1, ["Brown"])])
        env.players[attacker].hand = [rent(32, "Brown/Light Blue Rent", ["Brown", "Light Blue"])]
        env.players[defender].money = [money(35, 2)]

        env.action_context["action"] = 14  # Brown/Light Blue rent
        env.action_context["hand_card"] = 32
        env.action_context["my_set"]["colour"] = 1  # Brown
        env.action_context["my_set"]["set_index"] = 0
        env.deck.discardCard(env.players[attacker].removeHandCardById(32))

        env._apply_aggressive_effect(14)

        self.assertEqual(env.agent_selection, defender)
        self._drive_payment(env, defender, money_card_id=35)

        # Brown rent on a 1-property set is 1; defender paid 2M (only thing they had).
        self.assertEqual(sum(c.value for c in env.players[attacker].money), 2)


class DealBreakerTests(unittest.TestCase):
    def test_steal_set_no_aliasing(self):
        env = make_env()
        attacker, defender = "player_0", "player_1"
        env.agent_selection = attacker

        # Defender has a complete Brown set.
        brown_a = prop(1, "Brown", 1, ["Brown"])
        brown_b = prop(1, "Brown", 1, ["Brown"])
        stack_set(env, defender, "Brown", 0, [brown_a, brown_b])
        env.players[attacker].hand = [action_card(26, "Deal Breaker", 5)]

        env.action_context["action"] = 9
        env.action_context["hand_card"] = 26
        env.action_context["opponent_ID"] = env.agents.index(defender)
        env.action_context["opponent_set"]["colour"] = 1  # Brown
        env.action_context["opponent_set"]["set_index"] = 0
        env.deck.discardCard(env.players[attacker].removeHandCardById(26))

        env._apply_aggressive_effect(9)

        # Defender's Brown[0] is now a fresh empty PropertySet, attacker has the
        # populated one in their first empty Brown slot.
        self.assertTrue(env.players[defender].sets["Brown"][0].isEmpty())

        attacker_brown_with_props = [
            (i, s) for i, s in enumerate(env.players[attacker].sets["Brown"])
            if s.properties
        ]
        self.assertEqual(len(attacker_brown_with_props), 1)
        idx, stolen = attacker_brown_with_props[0]
        self.assertEqual(len(stolen.properties), 2)
        self.assertTrue(stolen.isCompleted())

        # The critical assertion: opp's slot and attacker's slot must be
        # different objects. Mutating one must not affect the other.
        self.assertIsNot(stolen, env.players[defender].sets["Brown"][0])
        env.players[defender].sets["Brown"][0].properties.append(brown_a)
        self.assertEqual(len(stolen.properties), 2)  # unchanged


class ForcedDealTests(unittest.TestCase):
    def _drive_placement(self, env, defender, colour_id, set_index):
        # Pick the colour bucket.
        action = make_action(set_={"colour": colour_id, "set_index": 0})
        env.step(action)
        # Pick the set_index.
        action = make_action(set_={"colour": colour_id, "set_index": set_index})
        env.step(action)

    def test_swap_completes_on_both_sides(self):
        env = make_env()
        attacker, defender = "player_0", "player_1"
        env.agent_selection = attacker

        attacker_brown = prop(1, "Brown", 1, ["Brown"])
        defender_lg = prop(2, "Light Green", 2, ["Light Green"])
        stack_set(env, attacker, "Brown", 0, [attacker_brown])
        stack_set(env, defender, "Light Green", 0, [defender_lg])
        env.players[attacker].hand = [action_card(21, "Forced Deal", 3)]

        env.action_context["action"] = 6
        env.action_context["hand_card"] = 21
        env.action_context["opponent_ID"] = env.agents.index(defender)
        env.action_context["opponent_property"] = {"colour": 2, "set_index": 0, "card": 2}  # Light Green
        env.action_context["my_property"] = {"colour": 1, "set_index": 0, "card": 1}  # Brown
        env.action_context["my_set"] = {"colour": 2, "set_index": 0}  # place stolen LG in attacker's LG[0]
        env.deck.discardCard(env.players[attacker].removeHandCardById(21))

        env._apply_aggressive_effect(6)

        # Defender now in placement phase.
        self.assertEqual(env.agent_selection, defender)
        self.assertEqual(env.action_context["decision"], DECISION_DEFENDER_FORCED_DEAL_PLACE_COLOUR)

        # Defender places the incoming Brown into Brown[0].
        self._drive_placement(env, defender, colour_id=1, set_index=0)

        self.assertIsNone(env.pending)
        # Attacker's Brown[0] is empty (gave away the Brown), LG[0] has the stolen card.
        self.assertTrue(env.players[attacker].sets["Brown"][0].isEmpty())
        attacker_lg_ids = [c.id for c in env.players[attacker].sets["Light Green"][0].properties]
        self.assertIn(2, attacker_lg_ids)
        # Defender's LG[0] is empty, Brown[0] has the swapped-in card.
        self.assertTrue(env.players[defender].sets["Light Green"][0].isEmpty())
        defender_brown_ids = [c.id for c in env.players[defender].sets["Brown"][0].properties]
        self.assertIn(1, defender_brown_ids)


class JsnTests(unittest.TestCase):
    def _setup_sly_deal_with_jsn_defender(self, defender_jsn_count=1, attacker_jsn_count=0):
        env = make_env()
        attacker, defender = "player_0", "player_1"
        env.agent_selection = attacker

        defender_brown = prop(1, "Brown", 1, ["Brown"])
        stack_set(env, defender, "Brown", 0, [defender_brown])
        env.players[attacker].hand = [action_card(23, "Sly Deal", 3)] + [jsn() for _ in range(attacker_jsn_count)]
        env.players[defender].hand = [jsn() for _ in range(defender_jsn_count)]

        env.action_context["action"] = 5
        env.action_context["hand_card"] = 23
        env.action_context["opponent_ID"] = env.agents.index(defender)
        env.action_context["opponent_property"] = {"colour": 1, "set_index": 0, "card": 1}
        env.action_context["my_set"] = {"colour": 1, "set_index": 0}  # place into attacker's Brown[0]
        env.deck.discardCard(env.players[attacker].removeHandCardById(23))

        env._start_jsn_check(attacker, defender, 5)
        return env, attacker, defender, defender_brown

    def test_defender_jsn_cancels_action(self):
        env, attacker, defender, defender_brown = self._setup_sly_deal_with_jsn_defender(
            defender_jsn_count=1, attacker_jsn_count=0,
        )

        # Defender plays JSN → attacker has no JSN to counter → chain ends, cancelled.
        self.assertEqual(env.agent_selection, defender)
        env.step(make_action(action_ID=16))

        self.assertIsNone(env.pending)
        # Defender still has the property; attacker's Brown[0] is empty.
        self.assertEqual([c.id for c in env.players[defender].sets["Brown"][0].properties], [1])
        self.assertTrue(env.players[attacker].sets["Brown"][0].isEmpty())
        # Both JSN cards from defender consumed (the only one).
        self.assertFalse(env.players[defender].hasJustSayNo())

    def test_jsn_chain_proceeds_on_double_play(self):
        env, attacker, defender, defender_brown = self._setup_sly_deal_with_jsn_defender(
            defender_jsn_count=1, attacker_jsn_count=1,
        )

        # Defender plays JSN.
        self.assertEqual(env.agent_selection, defender)
        env.step(make_action(action_ID=16))
        # Attacker's turn for counter-JSN.
        self.assertEqual(env.agent_selection, attacker)
        env.step(make_action(action_ID=16))

        # Chain ended (defender has no JSN left), even count → action proceeds.
        self.assertIsNone(env.pending)
        self.assertTrue(env.players[defender].sets["Brown"][0].isEmpty())
        # Attacker's Brown[0] now contains the stolen Brown.
        attacker_browns = [
            c for s in env.players[attacker].sets["Brown"] for c in s.properties
        ]
        self.assertEqual(len(attacker_browns), 1)
        self.assertEqual(attacker_browns[0].id, 1)

    def test_jsn_decline_lets_action_proceed(self):
        # Defender has JSN but chooses not to play it (action_ID = 0).
        env, attacker, defender, _ = self._setup_sly_deal_with_jsn_defender(
            defender_jsn_count=1, attacker_jsn_count=0,
        )
        env.step(make_action(action_ID=0))  # decline

        self.assertIsNone(env.pending)
        # JSN card retained (declined, not consumed).
        self.assertTrue(env.players[defender].hasJustSayNo())
        # Sly Deal proceeded — defender's Brown empty, attacker has it.
        self.assertTrue(env.players[defender].sets["Brown"][0].isEmpty())


class WinTests(unittest.TestCase):
    def test_three_completed_sets_terminates(self):
        env = make_env()
        env.agent_selection = "player_0"

        b1 = prop(1, "Brown", 1, ["Brown"])
        b2 = prop(1, "Brown", 1, ["Brown"])
        lg1 = prop(2, "Light Green", 2, ["Light Green"])
        lg2 = prop(2, "Light Green", 2, ["Light Green"])
        bl1 = prop(0, "Blue", 4, ["Blue"])
        bl2 = prop(0, "Blue", 4, ["Blue"])
        stack_set(env, "player_0", "Brown", 0, [b1, b2])
        stack_set(env, "player_0", "Light Green", 0, [lg1, lg2])
        stack_set(env, "player_0", "Blue", 0, [bl1, bl2])

        self.assertTrue(env._check_win())
        self.assertTrue(env.terminations["player_0"])
        self.assertTrue(env.terminations["player_1"])
        self.assertEqual(env._cumulative_rewards["player_0"], 1.0)
        self.assertEqual(env._cumulative_rewards["player_1"], -1.0)

    def test_two_completed_sets_does_not_terminate(self):
        env = make_env()
        env.agent_selection = "player_0"
        stack_set(env, "player_0", "Brown", 0, [
            prop(1, "Brown", 1, ["Brown"]),
            prop(1, "Brown", 1, ["Brown"]),
        ])
        stack_set(env, "player_0", "Light Green", 0, [
            prop(2, "Light Green", 2, ["Light Green"]),
            prop(2, "Light Green", 2, ["Light Green"]),
        ])
        self.assertFalse(env._check_win())
        self.assertFalse(env.terminations["player_0"])

    def test_active_player_wins_on_tie(self):
        # Both players hold 3 complete sets. Active player should win.
        env = make_env()
        env.agent_selection = "player_1"
        for owner in ("player_0", "player_1"):
            stack_set(env, owner, "Brown", 0, [
                prop(1, "Brown", 1, ["Brown"]),
                prop(1, "Brown", 1, ["Brown"]),
            ])
            stack_set(env, owner, "Light Green", 0, [
                prop(2, "Light Green", 2, ["Light Green"]),
                prop(2, "Light Green", 2, ["Light Green"]),
            ])
            stack_set(env, owner, "Blue", 0, [
                prop(0, "Blue", 4, ["Blue"]),
                prop(0, "Blue", 4, ["Blue"]),
            ])

        self.assertTrue(env._check_win())
        self.assertEqual(env._cumulative_rewards["player_1"], 1.0)
        self.assertEqual(env._cumulative_rewards["player_0"], -1.0)


if __name__ == "__main__":
    unittest.main()
