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
    plan = getattr(env.unwrapped, "departure_plan", {})  # hidden early departures, if any
    rows = []
    for i, ev in enumerate(env.EVs):
        early = plan.get((ev.location, ev.time_of_arrival))  # (announced, real) or None
        announced = early[0] if early else ev.time_of_departure
        soc = ev.current_capacity / ev.battery_capacity
        target = ev.desired_capacity / ev.battery_capacity
        rows.append(
            {
                "seed": seed,
                "ev_index": i,  # unique within the episode
                "charger": ev.location,
                "port": ev.id,  # EV2Gym's ev.id is the port number, not a unique id
                "arrival_step": ev.time_of_arrival,
                "departure_step": ev.time_of_departure,  # when it really left
                "announced_departure": announced,  # what the agent was told
                "early": early is not None,
                "early_by_steps": announced - ev.time_of_departure,
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
    early = np.array([e.get("early", False) for e in evs], dtype=bool)
    row.update(
        {
            "n_evs": len(evs),
            "pct_evs_met_target": float(100 * (1 - below.mean())) if len(evs) else np.nan,
            "mean_shortfall": float(shortfall.mean()) if len(evs) else np.nan,
            "max_shortfall": float(shortfall.max()) if len(evs) else np.nan,
            # Proposal §11: how well cars that left EARLY (unannounced) were served.
            "n_early": int(early.sum()),
            "pct_early_met_target": float(100 * (1 - below[early].mean()))
            if early.any()
            else np.nan,
            "mean_shortfall_early": float(shortfall[early].mean()) if early.any() else np.nan,
        }
    )
    return row


# Battery changes smaller than this (kWh per step) are not counted as charging or
# discharging: EV2Gym rounds every active battery UP to 0.01 kWh.
TRADE_MIN_KWH = 0.02


class TradeTracker:
    """How well a policy TIMES its energy trades (added after the optimum comparison).

    Call `before_step(env)` and `after_step()` around every env.step. Gives per day:
    - avg_buy_price / avg_sell_price: average €/kWh paid when charging / earned when
      discharging (weighted by energy). A good trader buys low and sells high;
    - trade_spread = avg_sell_price − avg_buy_price (the perfect-knowledge plan gets
      ~3.8 cents; a policy that sells as often as it buys at random prices gets ~0);
    - switches_per_car: how often a car flips between charging and discharging
      (the optimum ~4 per day; many more means jittering, not trading).
    """

    def __init__(self) -> None:
        self.bought = self.paid = self.sold = self.earned = 0.0
        self.last_sign: dict[tuple[int, int], int] = {}
        self.switches: dict[tuple[int, int], int] = {}
        self._pending: list = []

    def before_step(self, env) -> None:
        base = env.unwrapped
        t = base.current_step
        self._pending = [
            (
                ev,
                ev.current_capacity,
                abs(base.charge_prices[cs.id, t]),
                abs(base.discharge_prices[cs.id, t]),
            )
            for cs in base.charging_stations
            for ev in cs.evs_connected
            if ev is not None
        ]

    def after_step(self) -> None:
        for ev, before, buy, sell in self._pending:
            change = ev.current_capacity - before
            key = (ev.location, ev.time_of_arrival)
            self.switches.setdefault(key, 0)
            if abs(change) < TRADE_MIN_KWH:
                continue
            sign = 1 if change > 0 else -1
            if sign > 0:
                self.bought += change
                self.paid += change * buy
            else:
                self.sold -= change
                self.earned -= change * sell
            if self.last_sign.get(key, sign) != sign:
                self.switches[key] += 1
            self.last_sign[key] = sign

    def row(self) -> dict[str, float]:
        buy = self.paid / self.bought if self.bought > 0 else np.nan
        sell = self.earned / self.sold if self.sold > 0 else np.nan
        return {
            "avg_buy_price": float(buy),
            "avg_sell_price": float(sell),
            "trade_spread": float(sell - buy),
            "switches_per_car": float(np.mean(list(self.switches.values())))
            if self.switches
            else np.nan,
        }
