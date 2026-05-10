"""Bounded smoke test — same as test.py but caps the number of steps so we can
detect regressions without waiting for an episode to end (win detection isn't
implemented yet, so episodes don't terminate naturally). Also seeds gym's RNG
for reproducibility."""
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
from MonopolyDeal import MonopolyDeal

SEED = int(os.environ.get("SMOKE_SEED", "42"))
np.random.seed(SEED)
env = MonopolyDeal(render_mode=None)
env.reset(seed=SEED)
# Seed each agent's action space so sampling is deterministic
for agent in env.possible_agents:
    env.action_space(agent).seed(SEED)

MAX_STEPS = 20000
steps = 0
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        action = env.action_space(agent).sample(mask)

    env.step(action)
    steps += 1
    if steps >= MAX_STEPS:
        print(f"OK: {steps} steps without crash")
        break

env.close()
