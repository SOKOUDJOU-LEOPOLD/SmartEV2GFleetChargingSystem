"""Who leaves early, and when: models of hidden early departures (Phase 2, ADR-0002).

A departure model looks at one car (its arrival and ANNOUNCED departure) and
returns the step at which it will REALLY leave, or None if it leaves on time.
The agent never sees this; it only sees the announced time (POMDP [R10]).

`SyntheticDepartureModel` (Deliverable 1): with probability `p_early` a car leaves
early; it then cuts a random fraction of its announced stay, uniformly between
`cut_min` and `cut_max` (e.g. 10–50 %). Example: arrives at step 20, announces
step 52 (stay 32 steps), fraction 0.375 -> leaves at step 52 − 12 = 40.
These numbers are PROVISIONAL; Phase 6 replaces them with a model fitted to
ACN-Data (requested vs actual departure times) [R2].
"""

from __future__ import annotations

import numpy as np


class SyntheticDepartureModel:
    def __init__(self, p_early: float, cut_min: float = 0.1, cut_max: float = 0.5) -> None:
        if not 0 <= p_early <= 1:
            raise ValueError("p_early must be in [0, 1]")
        if not 0 < cut_min <= cut_max < 1:
            raise ValueError("need 0 < cut_min <= cut_max < 1")
        self.p_early = p_early
        self.cut_min, self.cut_max = cut_min, cut_max

    def sample(self, arrival: int, announced: int, rng: np.random.Generator) -> int | None:
        """Hidden real departure step (arrival < real < announced), or None (on time).

        Both random numbers are always drawn, so whether car k leaves early does not
        change the random numbers used for the other cars (same days for every p_early
        given the same generator).
        """
        u_early, u_cut = rng.random(), rng.random()
        stay = announced - arrival
        if u_early >= self.p_early or stay < 2:  # a car must stay at least one step
            return None
        fraction = self.cut_min + u_cut * (self.cut_max - self.cut_min)
        cut = max(1, round(fraction * stay))
        return max(arrival + 1, announced - cut)


def make_departure_model(spec: dict | None) -> SyntheticDepartureModel | None:
    """Departure model from the `env.departures` config section (None = all leave on time)."""
    if not spec:
        return None
    return SyntheticDepartureModel(
        p_early=spec["p_early"], cut_min=spec.get("cut_min", 0.1), cut_max=spec.get("cut_max", 0.5)
    )
