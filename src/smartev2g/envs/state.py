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


def V2G_profit_max_laxity(env, *args) -> np.ndarray:  # noqa: N802 (EV2Gym naming style)
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
                t_dep = ev.time_of_departure - t
                needed = steps_to_target(ev, cs, env.timescale)
                state.append(np.array([ev.get_soc(), t_dep, needed, t_dep - needed]))
    return np.hstack(state).astype(np.float64)
