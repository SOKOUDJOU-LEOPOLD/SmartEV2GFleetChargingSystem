"""What ONE charger sees, for the shared per-charger policy (ADR-0004).

Each charger gets the same site-level information plus information about its own
car only. The same small network reads this for every charger. Ref: [R18]

Layout (26 numbers, see `charger_observation_names`):
  shared: t, P_prev, price_0..price_19, n_cars
  own:    soc, t_dep, plugged
- t: current step; P_prev: total site power at the previous step (kW), so the
  charger can sense transformer stress; price_i: price i steps ahead;
  n_cars: how many cars are plugged in at the site.
- soc: its car's battery level (0..1); t_dep: steps left until its car's
  ANNOUNCED departure (the true one stays hidden, POMDP [R10]); plugged: 1 if a
  car is connected, else 0 (then soc = t_dep = 0).
Shared values follow EV2Gym's `V2G_profit_max` state function exactly.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from smartev2g.envs.make_env import PRICE_HORIZON

SHARED_NAMES = ["t", "P_prev"] + [f"price_{i}" for i in range(PRICE_HORIZON)] + ["n_cars"]
OWN_NAMES = ["soc", "t_dep", "plugged"]
N_FEATURES = len(SHARED_NAMES) + len(OWN_NAMES)


def charger_observation_names() -> list[str]:
    return SHARED_NAMES + OWN_NAMES


def _announced_departure(ev) -> int:
    return ev.time_of_departure


def charger_observations(
    env, departure_of: Callable[[object], int] = _announced_departure
) -> np.ndarray:
    """Array (n_chargers, 26): one row per charger, in the same order as the actions.

    `departure_of(ev)` gives the departure step the charger is told about: the
    announced one by default; the oracle passes the true one instead (Phase 2).
    """
    t = env.current_step
    prices = np.abs(env.charge_prices[0, t : t + PRICE_HORIZON])
    prices = np.pad(prices, (0, PRICE_HORIZON - len(prices)))
    cars = [ev for cs in env.charging_stations for ev in cs.evs_connected]
    n_cars = sum(ev is not None for ev in cars)
    shared = np.concatenate([[t, env.current_power_usage[t - 1]], prices, [n_cars]])

    rows = []
    for ev in cars:
        if ev is None:
            own = [0.0, 0.0, 0.0]
        else:
            own = [ev.get_soc(), departure_of(ev) - t, 1.0]
        rows.append(np.concatenate([shared, own]))
    return np.array(rows, dtype=np.float64)
