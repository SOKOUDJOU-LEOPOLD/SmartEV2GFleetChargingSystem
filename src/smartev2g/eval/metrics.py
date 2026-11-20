"""Turn a finished EV2Gym episode into rows for our CSV files.

- `ev_rows`: one row per car (for test_evs.csv): arrival, departure, battery
  level at arrival / departure, target, shortfall.
- `episode_row`: one row per simulated day (for train_episodes.csv,
  val_curve.csv, test_episodes.csv): EV2Gym's own statistics + our driver metrics.
See docs/plan/METRICS_AND_LOGGING.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# A car counts as "below target" if it is more than this fraction short (float rounding).
TARGET_TOLERANCE = 1e-3

# EV2Gym end-of-episode statistics we keep (ev2gym/utilities/utils.py, get_statistics).
EV2GYM_STATS = [
    "total_ev_served",
    "total_profits",  # money: + = earned (V2G selling), - = paid for charging (EUR)
    "total_energy_charged",  # kWh
    "total_energy_discharged",  # kWh sold back to the grid (V2G)
    "average_user_satisfaction",  # mean of (energy at departure / target)
    "energy_user_satisfaction",  # mean % of the max energy reachable at full speed
    "min_energy_user_satisfaction",
    "total_transformer_overload",  # kW above the transformer limit, summed over steps
    "battery_degradation",
    "total_reward",  # EV2Gym's own running sum of rewards
]


def ev_rows(env, seed: int) -> list[dict[str, Any]]:
    """One row per car that arrived during the episode (env.EVs, in arrival order)."""
    rows = []
    for i, ev in enumerate(env.EVs):
        soc = ev.current_capacity / ev.battery_capacity
        target = ev.desired_capacity / ev.battery_capacity
        rows.append(
            {
                "seed": seed,
                "ev_index": i,  # unique within the episode
                "charger": ev.location,
                "port": ev.id,  # EV2Gym's ev.id is the port number, not a unique id
                "arrival_step": ev.time_of_arrival,
                "departure_step": ev.time_of_departure,
                "stay_steps": ev.time_of_departure - ev.time_of_arrival,
                "stay_hours": (ev.time_of_departure - ev.time_of_arrival) * env.timescale / 60,
                "battery_kwh": ev.battery_capacity,
                "soc_arrival": ev.battery_capacity_at_arrival / ev.battery_capacity,
                "soc_final": soc,
                "soc_target": target,
                # Shortfall = how much battery (as a fraction) the driver is missing.
                "shortfall": max(0.0, target - soc),
                "below_target": bool(soc < target - TARGET_TOLERANCE),
            }
        )
    return rows


def episode_row(
    seed: int, episode_return: float, info: dict[str, Any], evs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summary of one simulated day.

    Args:
        seed: the day's seed.
        episode_return: sum of the rewards the agent received (its "score").
        info: EV2Gym's info dict from the LAST step (contains its statistics).
        evs: rows from `ev_rows` for this day.
    """
    row: dict[str, Any] = {"seed": seed, "return": float(episode_return)}
    for key in EV2GYM_STATS:
        if key in info:
            row[key] = float(info[key])
    shortfall = np.array([e["shortfall"] for e in evs], dtype=float)
    below = np.array([e["below_target"] for e in evs], dtype=bool)
    row.update(
        {
            "n_evs": len(evs),
            "pct_evs_met_target": float(100 * (1 - below.mean())) if len(evs) else np.nan,
            "mean_shortfall": float(shortfall.mean()) if len(evs) else np.nan,
            "max_shortfall": float(shortfall.max()) if len(evs) else np.nan,
        }
    )
    return row
