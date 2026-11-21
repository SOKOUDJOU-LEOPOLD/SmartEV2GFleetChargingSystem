"""Simple non-learning policies used as reference points.

Every policy here (and the PPO wrapper in eval/evaluate.py) has the same shape:
    action = policy.act(obs, env)
so the same evaluation code can run any of them.
"""

from __future__ import annotations

import numpy as np
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible

from smartev2g.envs.state import steps_to_target


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


class AFAPToTargetPolicy:
    """Charge every car as fast as possible UNTIL it reaches its target, then stop.

    EV2Gym's AFAP keeps charging until the battery is full, even when the driver only
    wants 80 %, paying for energy nobody asked for. This version is the fair "safe"
    baseline when targets are below 100 % (ADR-0006). In the last step before the
    target it charges only the fraction still needed (cars may end up to ~2.5 % above
    target: EV2Gym's real current/efficiency do not scale exactly with that fraction).
    No V2G.
    """

    name = "AFAP-target"

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: np.ndarray, env) -> np.ndarray:
        base = env.unwrapped
        actions = []
        for cs in base.charging_stations:
            for ev in cs.evs_connected:
                if ev is None:
                    actions.append(0.0)
                else:  # steps still needed at full power; < 1 -> charge that fraction
                    actions.append(min(1.0, steps_to_target(ev, cs, base.timescale)))
        return np.array(actions)


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
