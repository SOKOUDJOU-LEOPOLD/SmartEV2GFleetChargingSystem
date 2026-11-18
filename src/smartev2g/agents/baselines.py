"""Simple non-learning policies used as reference points.

Every policy here (and the PPO wrapper in eval/evaluate.py) has the same shape:
    action = policy.act(obs, env)
so the same evaluation code can run any of them.
"""

from __future__ import annotations

import numpy as np
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible


class AFAPPolicy:
    """Charge every car as fast as possible, never sell energy back (no V2G).

    Very safe for drivers but ignores prices. Ref: [R1] (EV2Gym heuristic).
    """

    name = "AFAP"

    def __init__(self) -> None:
        self._afap = ChargeAsFastAsPossible()

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: np.ndarray, env) -> np.ndarray:
        return self._afap.get_action(env)


class RandomPolicy:
    """Pick a random number in [-1, 1] for every charger at every step.

    The "knows nothing" baseline: any learning agent must beat it.
    Uses a PRIVATE random generator so it does not disturb EV2Gym's own random
    draws (EV2Gym re-seeds the global ones; see ADR-0002).
    """

    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        # Same day seed -> same random actions, so results are repeatable.
        self.rng = np.random.default_rng([self.base_seed, seed])

    def act(self, obs: np.ndarray, env) -> np.ndarray:
        low, high = env.action_space.low, env.action_space.high
        return self.rng.uniform(low, high)
