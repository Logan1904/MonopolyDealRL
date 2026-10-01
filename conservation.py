"""Card conservation over a long random rollout: nothing created or destroyed."""
import io, contextlib
from collections import Counter
from MonopolyDeal import MonopolyDeal
from cardsdb import ALL_CARDS

def census(env):
    c = Counter(card.id for card in env.deck.deck)
    c += Counter(card.id for card in env.deck.discard_pile)
    for p in env.players.values():
        c += Counter(card.id for card in p.hand)
        c += Counter(card.id for card in p.money)
        for pSets in p.sets.values():
            for pSet in pSets:
                c += Counter(card.id for card in pSet.properties)
    # Forced-deal placement holds a card in flight between attacker resolve
    # and defender placement; count it or it looks like a leak.
    if getattr(env, "pending", None) and env.pending.get("card") is not None:
        c += Counter([env.pending["card"].id])
    return c

env = MonopolyDeal(render_mode=None)
env.reset(seed=7)
expected = Counter(card.id for card in ALL_CARDS)
first_bad = None
with contextlib.redirect_stdout(io.StringIO()):
    for i in range(50000):
        if census(env) != expected:
            first_bad = i
            break
        agent = env.agent_selection
        obs, r, term, trunc, info = env.last()
        action = None if (term or trunc) else env.action_space(agent).sample(obs["action_mask"])
        env.step(action)
print("conserved through 50k steps" if first_bad is None else f"LEAK at step {first_bad}")