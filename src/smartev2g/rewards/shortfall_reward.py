"""Our reward (proposal §6): profit, minus squared penalties for grid overload and
for cars that leave below their target charge.

    R_t = R_econ − λ_over · Σ_transformers overload² − λ_dep · Σ_{cars leaving at t} shortfall²

- R_econ: money earned this step (EUR; + = selling energy, − = buying it). [R1]
- overload: kW beyond the transformer's rating. The rating limits power in BOTH
  directions (charging above +100 kW, or selling below −100 kW).
- shortfall = max(0, target SoC − SoC when the car leaves), as a fraction (0..1).
  It is applied to EVERY departing car, whether it leaves on time or early.

Why squared? Big misses hurt much more than small ones: 0.1 short → 0.01,
0.3 short → 0.09. So the agent prefers many small shortfalls to one big one.
The λ weights set the trade-off between profit and drivers/grid; we test several
λ values because results depend on them (Ref: [R4], [R7]).

Each term is also stored on the env (`env.reward_terms` for the step,
`env.reward_terms_episode` summed over the day) for logging
(docs/plan/METRICS_AND_LOGGING.md: "each reward term separately").

Per-charger split (ADR-0004): the same reward is also divided between the chargers
(`env.charger_rewards`, one number per charger) so that a shared per-charger policy
gets credit/blame for ITS own decisions. The parts always add up to the site reward:
- profit: what each charger itself earned/paid this step;
- departure penalty: paid by the charger whose car left;
- overload penalty: shared in proportion to how much each charger pushed in the
  overload direction (charging when above +limit, selling when below −limit);
  equally if no charger did (e.g. other loads caused it). Ref: [R18]
"""

from __future__ import annotations

from functools import partial

import numpy as np

REWARD_TERMS = ["econ", "overload_penalty", "departure_penalty"]


def soc_shortfall(ev) -> float:
    """How much battery (fraction 0..1) the driver is missing: max(0, target − actual)."""
    target = ev.desired_capacity / ev.battery_capacity
    actual = ev.current_capacity / ev.battery_capacity
    return max(0.0, target - actual)


def transformer_overload_kw(tr, step: int) -> float:
    """kW beyond the transformer rating at `step` (0 if within limits).

    We compute this ourselves: EV2Gym's `get_how_overloaded()` measures the distance
    to the UPPER limit even when the overload comes from discharging (below the
    lower limit), which overstates it (e.g. −120 kW gives 220 instead of 20).
    """
    power = tr.current_power
    return float(max(0.0, power - tr.max_power[step], tr.min_power[step] - power))


def charger_offsets(env) -> dict[int, int]:
    """charging-station id -> index of its first port in the action/observation order."""
    offsets, i = {}, 0
    for cs in env.charging_stations:
        offsets[cs.id] = i
        i += cs.n_ports
    return offsets


def charger_terms(env, step: int, lambda_dep: float, lambda_over: float) -> dict[str, np.ndarray]:
    """Split this step's reward terms between the chargers (1 port per charger).

    Returns arrays of length n_chargers whose sums equal the site-level terms.
    """
    n = len(env.charging_stations)
    profits = np.array([cs.total_profits for cs in env.charging_stations], dtype=float)
    if step == 0 or getattr(env, "_cs_profits_prev", None) is None:
        env._cs_profits_prev = np.zeros(n)  # EV2Gym resets total_profits at reset
    econ = profits - env._cs_profits_prev
    env._cs_profits_prev = profits

    departure = np.zeros(n)
    offsets = charger_offsets(env)
    for ev in env.departing_evs:
        departure[offsets[ev.location] + ev.id] += lambda_dep * soc_shortfall(ev) ** 2

    overload = np.zeros(n)
    power = np.array([cs.current_power_output for cs in env.charging_stations], dtype=float)
    for tr in env.transformers:
        over = transformer_overload_kw(tr, step)
        if over <= 0:
            continue
        mine = np.array([cs.connected_transformer == tr.id for cs in env.charging_stations])
        direction = 1.0 if tr.current_power > tr.max_power[step] else -1.0
        push = np.where(mine, np.maximum(0.0, direction * power), 0.0)
        weights = push / push.sum() if push.sum() > 0 else mine / mine.sum()
        overload += lambda_over * over**2 * weights

    return {"econ": econ, "overload_penalty": overload, "departure_penalty": departure}


def squared_shortfall(
    env, total_costs, user_satisfaction_list, *args, lambda_dep: float, lambda_over: float
) -> float:
    """Reward for the step that just finished (same call signature as EV2Gym rewards).

    EV2Gym calls the reward AFTER moving to the next step, so the finished step is
    `current_step − 1`; `env.departing_evs` are the cars that left during it.
    """
    step = env.current_step - 1
    econ = float(total_costs)
    overload_sq = sum(transformer_overload_kw(tr, step) ** 2 for tr in env.transformers)
    shortfall_sq = sum(soc_shortfall(ev) ** 2 for ev in env.departing_evs)

    terms = {
        "econ": econ,
        "overload_penalty": lambda_over * overload_sq,
        "departure_penalty": lambda_dep * shortfall_sq,
    }
    # Running sums for the day; reset at the first step of each episode.
    if step == 0 or not hasattr(env, "reward_terms_episode"):
        env.reward_terms_episode = dict.fromkeys(REWARD_TERMS, 0.0)
    for k, v in terms.items():
        env.reward_terms_episode[k] += v
    env.reward_terms = terms

    # Per-charger split for the shared per-charger policy (ADR-0004); only for real
    # EV2Gym envs with one port per charger.
    stations = getattr(env, "charging_stations", None)
    if stations and all(cs.n_ports == 1 for cs in stations):
        parts = charger_terms(env, step, lambda_dep, lambda_over)
        env.charger_reward_terms = parts
        env.charger_rewards = parts["econ"] - parts["overload_penalty"] - parts["departure_penalty"]

    return econ - terms["overload_penalty"] - terms["departure_penalty"]


def make_squared_shortfall(lambda_dep: float, lambda_over: float):
    """Reward function with fixed λ values (a plain function EV2Gym can call)."""
    if lambda_dep < 0 or lambda_over < 0:
        raise ValueError("λ weights must be >= 0")
    return partial(squared_shortfall, lambda_dep=lambda_dep, lambda_over=lambda_over)
