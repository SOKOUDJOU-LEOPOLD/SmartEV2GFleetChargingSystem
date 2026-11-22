"""Perfect-knowledge plan: the best possible day, computed with hindsight (Phase 2).

At the start of the day the planner is told EVERYTHING: all prices, every car's
arrival, battery and departure. It then solves the whole day as one optimisation
problem ("linear program") and plays the plan in the simulator. No learning involved.

Two versions (one switch, `knows`):
- "real":      uses each car's REAL departure (including hidden early ones). Nobody can
               do better on that day -> the CEILING for every agent (an exact,
               non-learning version of the Oracle baseline [R6]).
- "announced": uses the ANNOUNCED departures and trusts them, like PPO-Trusting but
               planning perfectly. Early leavers drive off before their planned top-up.
The gap between the two = the most that knowing early departures is worth (in reward).

What is optimised is exactly our reward (rewards/shortfall_reward.py):
    money − λ_over·Σ_t overload_t² − λ_dep·Σ_cars shortfall²
Squares are not linear, so each one is replaced by many straight lines that touch it
("tangent cuts": x² ≥ 2a·x − a² for many points a). Between two points the error is
at most (gap/2)², i.e. tiny. The LP is solved with HiGHS (bundled with SciPy) [R19].

Battery physics (EV2Gym [R1], models/ev.py), checked against the simulator in tests:
- money is paid/earned on the energy that enters/leaves the BATTERY, and the
  transformer sees that same energy (kW = kWh per step / step length in hours);
- the most a car can take or give per step depends on its charger current and its
  efficiency (which varies with current);
- above its "transition" level a battery charges slower (constant-voltage phase):
  per step at most α·(capacity − energy now). This only matters above ~90 % here.
Small things the plan ignores (the replay corrects them): EV2Gym rounds the battery
energy UP to 0.01 kWh every step, and the slow-down rule is slightly simplified.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

# Points where the squares are touched by straight lines (see module docstring).
SHORTFALL_CUTS = np.linspace(0.0, 1.0, 201)  # SoC shortfall 0..100 %, every 0.5 %
OVERLOAD_CUTS = np.linspace(0.0, 40.0, 161)  # kW above the rating, every 0.25 kW
AMPS_GRID = np.linspace(0.0, 1.0, 1001)  # fractions of the charger current, for the max rate


# --- EV2Gym battery formulas (same maths as ev2gym/models/ev.py _charge/_discharge) ---


def _efficiency(table, amps: float) -> float:
    """Efficiency (0..1) at this current; EV2Gym stores it per whole ampere, in %."""
    if isinstance(table, dict):
        return table.get(np.round(amps), 1) / 100
    return float(table)


def charge_energy(ev, amps: float, cs, timescale: float) -> float:
    """kWh that enter the battery this step when charging at `amps` (> 0)."""
    if amps <= 0:
        return 0.0
    voltage = cs.voltage * math.sqrt(min(cs.phases, ev.ev_phases))
    eff = _efficiency(ev.charge_efficiency, amps)
    per_hour = 60 / timescale
    pilot = eff * amps * voltage / 1000 / ev.battery_capacity / per_hour  # SoC gain asked
    top = eff * ev.max_ac_charge_power / ev.battery_capacity / per_hour  # car's own limit
    pilot = min(pilot, top)
    soc = ev.current_capacity / ev.battery_capacity
    tr = ev.transition_soc
    if tr == 1:
        new = min(1.0, soc + pilot)
    else:  # slower charging above the transition level (constant-voltage phase)
        p_tr = tr + (pilot - top) / top * (tr - 1)
        k = ev.transition_soc_multiplier
        if soc < p_tr:
            if 1 <= (p_tr - soc) / pilot:
                new = soc + pilot
            else:
                new = 1 + np.exp(k * (pilot + soc - p_tr) / (p_tr - 1)) * (p_tr - 1)
        else:
            new = 1 + np.exp(k * pilot / (p_tr - 1)) * (soc - 1)
        new = min(new, soc + min(pilot, top))
    return (new - soc) * ev.battery_capacity


def discharge_energy(ev, amps: float, cs, timescale: float) -> float:
    """kWh that leave the battery this step when discharging at `amps` (> 0)."""
    if amps <= 0:
        return 0.0
    voltage = cs.voltage * math.sqrt(min(cs.phases, ev.ev_phases))
    power = min(amps * voltage / 1000, abs(ev.max_discharge_power))
    table = ev.discharge_efficiency if isinstance(ev.charge_efficiency, dict) else None
    eff = _efficiency(table, amps) if table is not None else float(ev.discharge_efficiency)
    energy = power * eff * timescale / 60
    return max(0.0, min(energy, ev.current_capacity - ev.min_battery_capacity))


def _max_rates(ev, cs, timescale: float) -> tuple[float, float, float]:
    """(max kWh in per step at low SoC, max kWh out per step, slow-down factor α)."""
    saved = ev.current_capacity
    ev.current_capacity = ev.battery_capacity  # full: discharging not limited by the minimum
    max_out = max(
        discharge_energy(ev, a * abs(cs.max_discharge_current), cs, timescale) for a in AMPS_GRID
    )
    ev.current_capacity = 0.0  # empty: charging not slowed down
    max_in = max(charge_energy(ev, a * cs.max_charge_current, cs, timescale) for a in AMPS_GRID)
    ev.current_capacity = saved
    # Slow-down above the transition level at full current: gain ≤ α·(1 − SoC).
    if ev.transition_soc < 1:
        top = max_in / ev.battery_capacity
        alpha = 1 - math.exp(-ev.transition_soc_multiplier * top / (1 - ev.transition_soc))
    else:
        alpha = 1.0
    return max_in, max_out, alpha


# --- The day's plan (linear program) ---


def car_key(ev) -> tuple[int, int]:
    return (ev.location, ev.time_of_arrival)


def plan_day(
    env, knows: str = "real", lambda_dep: float = 500.0, lambda_over: float = 0.1
) -> dict[str, Any]:
    """Best plan for the day the env was just reset to.

    Returns {"energy": {car key: planned battery energy (kWh) after each step,
    as {step: kWh}}, "predicted": the reward terms the plan expects}.
    """
    if knows not in ("real", "announced"):
        raise ValueError("knows must be 'real' or 'announced'")
    base = env.unwrapped
    if len(base.transformers) != 1:  # our site (configs/env/) has one transformer
        raise NotImplementedError("plan_day supports one transformer")
    n_steps, dt = base.simulation_length, base.timescale / 60
    plan = getattr(base, "departure_plan", {}) or {}
    cars = []
    for ev in base.EVs_profiles:
        key = car_key(ev)
        dep = plan[key][1] if (knows == "real" and key in plan) else ev.time_of_departure
        steps = list(range(ev.time_of_arrival, min(dep, n_steps - 1) + 1))  # controlled steps
        if steps:
            cars.append((ev, key, dep, steps))

    # Variable layout: per car and step c (kWh in), d (kWh out), E (kWh after the step);
    # per car s (SoC shortfall), z (≥ s²); per step o (overload kW), q (≥ o²).
    n = 0
    idx = []
    for _, _, _, steps in cars:
        m = len(steps)
        idx.append((n, n + m, n + 2 * m))  # start of c, d, E blocks
        n += 3 * m
    s0 = n
    n += len(cars)
    z0 = n
    n += len(cars)
    o0 = n
    n += n_steps
    q0 = n
    n += n_steps

    cost = np.zeros(n)
    lo, hi = np.zeros(n), np.full(n, np.inf)
    eq_r, eq_c, eq_v, eq_b = [], [], [], []  # equality rows (battery bookkeeping)
    ub_r, ub_c, ub_v, ub_b = [], [], [], []  # "≤" rows
    row_eq = row_ub = 0
    power_terms = [[] for _ in range(n_steps)]  # (column, sign) per step for the transformer

    for j, (ev, _, dep, steps) in enumerate(cars):
        cs = base.charging_stations[ev.location]
        max_in, max_out, alpha = _max_rates(ev, cs, base.timescale)
        c0, d0, e0 = idx[j]
        e_start = ev.battery_capacity_at_arrival
        for i, t in enumerate(steps):
            c, d, e = c0 + i, d0 + i, e0 + i
            # Money: pay |charge price| per kWh in, earn discharge price per kWh out.
            cost[c] = abs(base.charge_prices[cs.id, t])
            cost[d] = -base.discharge_prices[cs.id, t]
            hi[c], hi[d] = max_in, max_out
            lo[e], hi[e] = ev.min_battery_capacity, ev.battery_capacity
            # E_t − E_{t−1} − c_t + d_t = 0 (E_{−1} = energy at arrival)
            eq_r += [row_eq] * 3
            eq_c += [e, c, d]
            eq_v += [1.0, -1.0, 1.0]
            if i > 0:
                eq_r.append(row_eq)
                eq_c.append(e - 1)
                eq_v.append(-1.0)
                eq_b.append(0.0)
            else:
                eq_b.append(e_start)
            row_eq += 1
            # Slower charging near full: c_t ≤ α·(capacity − E_{t−1})
            if ev.transition_soc < 1:
                ub_r += [row_ub]
                ub_c += [c]
                ub_v += [1.0]
                if i > 0:
                    ub_r.append(row_ub)
                    ub_c.append(e - 1)
                    ub_v.append(alpha)
                    ub_b.append(alpha * ev.battery_capacity)
                else:
                    ub_b.append(alpha * (ev.battery_capacity - e_start))
                row_ub += 1
            power_terms[t] += [(c, 1.0 / dt), (d, -1.0 / dt)]
        # Shortfall when it leaves (only cars that leave during the day are judged).
        if dep <= n_steps - 1:
            target = ev.desired_capacity / ev.battery_capacity
            # target − E_dep/cap − s ≤ 0
            ub_r += [row_ub, row_ub]
            ub_c += [e0 + len(steps) - 1, s0 + j]
            ub_v += [-1.0 / ev.battery_capacity, -1.0]
            ub_b.append(-target)
            row_ub += 1
            for a in SHORTFALL_CUTS:  # z ≥ 2a·s − a²
                ub_r += [row_ub, row_ub]
                ub_c += [s0 + j, z0 + j]
                ub_v += [2 * a, -1.0]
                ub_b.append(a * a)
                row_ub += 1
        cost[z0 + j] = lambda_dep

    tr = base.transformers[0]
    for t in range(n_steps):
        cols = power_terms[t]
        # power − o ≤ max  and  −power − o ≤ −min   (overload in both directions)
        for sign, bound in ((1.0, tr.max_power[t]), (-1.0, -tr.min_power[t])):
            ub_r += [row_ub] * (len(cols) + 1)
            ub_c += [col for col, _ in cols] + [o0 + t]
            ub_v += [sign * v for _, v in cols] + [-1.0]
            ub_b.append(bound)
            row_ub += 1
    for t in range(n_steps):
        for a in OVERLOAD_CUTS:  # q ≥ 2a·o − a²
            ub_r += [row_ub, row_ub]
            ub_c += [o0 + t, q0 + t]
            ub_v += [2 * a, -1.0]
            ub_b.append(a * a)
            row_ub += 1
        cost[q0 + t] = lambda_over

    a_eq = sparse.csr_matrix((eq_v, (eq_r, eq_c)), shape=(row_eq, n))
    a_ub = sparse.csr_matrix((ub_v, (ub_r, ub_c)), shape=(row_ub, n))
    res = linprog(
        cost,
        A_ub=a_ub,
        b_ub=np.array(ub_b),
        A_eq=a_eq,
        b_eq=np.array(eq_b),
        bounds=np.column_stack([lo, hi]),
        method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"plan_day: solver failed ({res.message})")
    x = res.x

    energy = {}
    for j, (_, key, _, steps) in enumerate(cars):
        e0 = idx[j][2]
        energy[key] = {t: float(x[e0 + i]) for i, t in enumerate(steps)}
    money = -float(cost[:s0] @ x[:s0])  # car columns only (E columns cost nothing)
    predicted = {
        "econ": money,
        "overload_penalty": lambda_over * float(np.sum(x[o0 : o0 + n_steps] ** 2)),
        "departure_penalty": lambda_dep * float(np.sum(x[s0 : s0 + len(cars)] ** 2)),
    }
    predicted["return"] = (
        predicted["econ"] - predicted["overload_penalty"] - predicted["departure_penalty"]
    )
    return {"energy": energy, "predicted": predicted}


# --- Playing the plan in the simulator (same interface as every other policy) ---


def _amps_for(energy_fn, ev, cs, timescale: float, want: float, max_amps: float) -> float:
    """Smallest current that moves `want` kWh this step (bisection; max if unreachable)."""
    if energy_fn(ev, max_amps, cs, timescale) <= want:
        return max_amps
    lo, hi = 0.0, max_amps
    for _ in range(40):
        mid = (lo + hi) / 2
        if energy_fn(ev, mid, cs, timescale) < want:
            lo = mid
        else:
            hi = mid
    return hi


class PerfectPlanPolicy:
    """Plays the perfect-knowledge plan (`knows` = "real" or "announced", see top).

    At each step it steers every battery towards the energy the plan wants after that
    step (so EV2Gym's small rounding is corrected as it goes).
    """

    def __init__(self, knows: str = "real", lambda_dep: float = 500.0, lambda_over: float = 0.1):
        self.knows = knows
        self.lambda_dep, self.lambda_over = lambda_dep, lambda_over
        self.name = "Optimum-real" if knows == "real" else "Optimum-announced"
        self.plan: dict[str, Any] | None = None
        self.predicted: list[dict[str, float]] = []  # one entry per day played

    def reset(self, seed: int) -> None:
        self.plan = None  # planned at the first step, once the day exists

    def act(self, obs: np.ndarray, env) -> np.ndarray:
        base = env.unwrapped
        if self.plan is None:
            self.plan = plan_day(base, self.knows, self.lambda_dep, self.lambda_over)
            self.predicted.append(self.plan["predicted"])
        t = base.current_step
        actions = []
        for cs in base.charging_stations:
            for ev in cs.evs_connected:
                wanted = None if ev is None else self.plan["energy"].get(car_key(ev), {}).get(t)
                if wanted is None:
                    actions.append(0.0)
                    continue
                delta = wanted - ev.current_capacity
                if delta > 1e-4:
                    amps = _amps_for(
                        charge_energy, ev, cs, base.timescale, delta, cs.max_charge_current
                    )
                    actions.append(amps / cs.max_charge_current)
                elif delta < -1e-4:
                    top = abs(cs.max_discharge_current)
                    amps = _amps_for(discharge_energy, ev, cs, base.timescale, -delta, top)
                    actions.append(-amps / top)
                else:
                    actions.append(0.0)
        return np.array(actions)
