"""Our observation ("state function"): EV2Gym's V2G_profit_max + laxity per car.

EV2Gym's `V2G_profit_max` gives, per port, the car's battery level (SoC, %) and the
steps left until its announced departure. It does NOT give the battery SIZE, so the
agent cannot tell how long a car needs to charge: two cars at 60 % leaving in 2 h
look identical, but a 46 kWh car needs 1.7 h at 11 kW and a 77 kWh car 2.8 h.

So we add, per port:
- `steps_needed`: steps of full-power charging still needed to reach the target
  = energy missing (kWh) / (max charging power (kW) × efficiency × step length (h)).
- `laxity`: spare time = steps left until the ANNOUNCED departure − steps_needed.
  Negative = cannot reach the target even at full power from now on.
Laxity is a standard quantity in EV charging scheduling (least-laxity-first).
Both use only information the agent is allowed to know (the announced departure),
so the hidden true departure never leaks (POMDP [R10], ADR-0002).

Layout: [t, P_prev, price_0..19, then per port: soc, t_dep, steps_needed, laxity]
(20 ports -> 102 numbers). Empty port -> four zeros.
"""

from __future__ import annotations

import datetime

import numpy as np

# Number of future prices in the observation (fixed in EV2Gym's V2G_profit_max).
PRICE_HORIZON = 20
PORT_FEATURES_LAXITY = ["soc", "t_dep", "steps_needed", "laxity"]


def charge_efficiency(ev) -> float:
    """Best charging efficiency of the car (0..1), as EV2Gym uses for its AFAP maximum."""
    eff = ev.charge_efficiency
    if isinstance(eff, dict):  # efficiency per charging current, in %
        return max(eff.values()) / 100
    return float(eff)


def steps_to_target(ev, cs, timescale_min: float) -> float:
    """Steps of full-power charging needed to bring `ev` from now to its target."""
    missing_kwh = max(0.0, ev.desired_capacity - ev.current_capacity)
    max_kw = min(ev.max_ac_charge_power, cs.get_max_power())
    kwh_per_step = max_kw * charge_efficiency(ev) * timescale_min / 60
    return missing_kwh / kwh_per_step if kwh_per_step > 0 else 0.0


def departure_seen(env, ev, real_departures: bool = False) -> int:
    """Departure step the agent is shown: the ANNOUNCED one (normal agents), or the REAL
    one for PPO-Oracle (privileged information [R6]; hidden early departures, ADR-0002)."""
    if real_departures:
        plan = getattr(env, "departure_plan", {}) or {}
        early = plan.get((ev.location, ev.time_of_arrival))
        if early is not None:
            return early[1]
    return ev.time_of_departure


def V2G_profit_max_laxity(env, *args, real_departures: bool = False) -> np.ndarray:  # noqa: N802
    """EV2Gym's V2G_profit_max observation plus steps_needed and laxity per port."""
    t = env.current_step
    prices = np.abs(env.charge_prices[0, t : t + PRICE_HORIZON])
    prices = np.pad(prices, (0, PRICE_HORIZON - len(prices)))
    state = [np.array([t, env.current_power_usage[t - 1]]), prices]
    for tr in env.transformers:
        for cs in env.charging_stations:
            if cs.connected_transformer != tr.id:
                continue
            for ev in cs.evs_connected:
                if ev is None:
                    state.append(np.zeros(len(PORT_FEATURES_LAXITY)))
                    continue
                t_dep = departure_seen(env, ev, real_departures) - t
                needed = steps_to_target(ev, cs, env.timescale)
                state.append(np.array([ev.get_soc(), t_dep, needed, t_dep - needed]))
    return np.hstack(state).astype(np.float64)


# --- Day-ahead price view (after the PPO gap diagnosis, 2026-11-21) ---
# Our prices are Dutch DAY-AHEAD prices: the market closes at 12:00 CET the day
# before and publishes every price of the next day around 13:00. So a real operator
# always knows all prices until midnight, and from 13:00 also tomorrow's prices.
# EV2Gym's observation shows only the next 20 steps (5 h), shorter than a typical
# stay (~8 h), so the agent cannot see e.g. the cheap midday valley or the evening
# price when deciding in the morning.
DAYAHEAD_HORIZON = 96  # price slots in the observation: 24 h of 15-min steps
DAYAHEAD_PUBLISH_HOUR = 13  # tomorrow's prices are known from this hour on


def known_price_steps(env, t: int) -> int:
    """Last step (exclusive) whose day-ahead price is already published at step t."""
    minutes = env.timescale
    now = env.sim_starting_date + datetime.timedelta(minutes=t * minutes)
    midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time())
    if now.hour >= DAYAHEAD_PUBLISH_HOUR:
        midnight += datetime.timedelta(days=1)  # tomorrow's prices are out too
    until = int((midnight - env.sim_starting_date).total_seconds() // (60 * minutes))
    return min(until, env.simulation_length)


def V2G_profit_max_laxity_dayahead(env, *args, real_departures: bool = False) -> np.ndarray:  # noqa: N802
    """Like V2G_profit_max_laxity, but with every PUBLISHED price of the next 24 h.

    Layout: [t, P_prev, price_0..95, n_known, then per port: soc, t_dep,
    steps_needed, laxity] (20 ports -> 179 numbers). Prices not yet published (or
    after the end of the simulated day) are 0; n_known says how many are real.
    """
    t = env.current_step
    known = max(0, known_price_steps(env, t) - t)
    n = min(known, DAYAHEAD_HORIZON)
    prices = np.zeros(DAYAHEAD_HORIZON)
    prices[:n] = np.abs(env.charge_prices[0, t : t + n])
    base = V2G_profit_max_laxity(env, real_departures=real_departures)
    head = 2 + PRICE_HORIZON  # [t, P_prev, 20 prices] in the laxity layout
    return np.hstack([base[:2], prices, [n], base[head:]]).astype(np.float64)


def V2G_profit_max_laxity_dayahead_oracle(env, *args) -> np.ndarray:  # noqa: N802
    """PPO-Oracle's observation: the day-ahead one, but t_dep and laxity use each car's
    REAL departure (it "knows" who leaves early). Upper reference for the learning
    agents (Phase 4); the exact ceiling is Optimum-real (eval/optimum.py)."""
    return V2G_profit_max_laxity_dayahead(env, real_departures=True)
