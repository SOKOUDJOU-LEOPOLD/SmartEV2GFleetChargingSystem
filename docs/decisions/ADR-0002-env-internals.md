# ADR-0002: EV2Gym internals and how the early-departure wrapper will hook in

- **Date:** 2026-11-17
- **Status:** Accepted
- **Author(s):** Leopold Sokoudjou Gatsing

## Context
The `UnexpectedDepartureWrapper` (Phase 2) must make some cars leave earlier than announced **without** the agent ever seeing the true time. To do that safely we need to know exactly how EV2Gym 2.0.0 stores departures, removes cars, and builds the observation. Findings come from reading the installed source (`ev2gym/models/ev2gym_env.py`, `ev_charger.py`, `ev.py`, `rl_agent/state.py`, `rl_agent/reward.py`).

## Findings

### Versions and config
- `ev2gym==2.0.0` (MIT license). Our env config `configs/env/v2g_profitmax_20port.yaml` is the example `V2GProfitMax.yaml` with 20 chargers and **no `ev_specs_file` line** (that line is a path relative to EV2Gym's source repo and crashes when installed with pip; without it EV2Gym loads its bundled EV specs).

### One step of `EV2Gym.step(actions)`, in order
1. For each charging station `cs.step(...)`: apply the actions (charge/discharge) for this step, add profit, **then** remove every car with `cs.current_step >= ev.time_of_departure` (`EV.is_departing`). Removed cars are returned as `departing_evs` and their satisfaction `current_capacity / desired_capacity` is reported.
2. New cars whose `time_of_arrival == current_step + 1` are plugged in.
3. `env.current_step += 1`, reward computed by the reward function, observation built by the state function **after** all of the above.

Consequence: a car is charged/discharged during its departure step, then leaves; the next observation no longer contains it.

### Where departure information lives
- `ev.time_of_departure` (int, simulation step) — the **only** departure field. It is set when episode profiles are generated and copied (`deepcopy`) when the car plugs in.
- `env.EVs` lists every car that has arrived in the episode; `cs.evs_connected[port]` holds cars currently plugged in.
- `ev.desired_capacity` (kWh) = `config.ev.desired_capacity × battery_capacity`. With our config `desired_capacity: 1`, **every driver wants a full battery**.

### What the agent observes (`V2G_profit_max`)
`[t, P_prev, price_0..price_19, (soc_k, t_dep_k) for each of the 20 ports]` = 62 numbers, where `t_dep_k = ev.time_of_departure − current_step` (0 if the port is empty). See `observation_names()` in `src/smartev2g/envs/make_env.py`.

### Default reward (`profit_maximization`)
`reward = profit − Σ_departing 100·exp(−10·satisfaction)`. Used only for the Phase 1 sanity check; Phase 2 replaces the penalty with our squared-shortfall term.

### Random seeds
- `EV2Gym.reset(seed)` calls `np.random.seed(seed)` and `random.seed(seed)` — it re-seeds the **global** generators. `reset()` without a seed draws a new seed from the global NumPy generator, so a run is fully determined by its first seed. Verified: same seed + same actions ⇒ identical episode (`tests/test_make_env.py`).
- `tr_seed: -1` is replaced by the first seed and then kept for later resets. It only affects transformer loads/solar, which are disabled in our config, so it does not matter now.

### Typical simulated day (AFAP rule, 30 seeds; `scripts/inspect_env.py`)
| Quantity | Value |
|---|---|
| Episode | 112 steps × 15 min = 28 h, starts 05:00, random weekday |
| Cars per episode | mean 14.9 (11–20) |
| Announced stay | mean 7.4 h (3.5–13.3 h), median 31 steps |
| SoC at arrival | mean 0.74 (0.22–0.93) |
| Cars below target when charged at full speed | 0.9 % |

So drivers usually have hours of slack, which is exactly what makes V2G discharging tempting — and risky if they leave early.

## Decision: how the wrapper forces an early departure
**"Just-in-time" swap.** The wrapper keeps the hidden true departure `T_actual` in its own table (keyed by car), never in the car object while the agent can see it. Right **before** calling `env.step()` at the step where `current_step == T_actual`, it sets `ev.time_of_departure = T_actual`. EV2Gym then removes the car inside that same step with its normal logic (satisfaction, statistics, `departing_evs`), and the next observation no longer contains the car. The announced time `T_scheduled` is stored by the wrapper for logging.

Why this approach:
- The agent only ever sees `t_dep = T_scheduled − t` (the observation is built after the car has already left) ⇒ no information leak. A unit test will assert this.
- Reuses EV2Gym's own departure code, so profit, satisfaction and statistics stay consistent.
- No changes to EV2Gym's source (no fork to maintain).

Why not the alternatives:
- *Set `time_of_departure = T_actual` at arrival and rewrite the observation* — the true time would sit inside the car object for hours and every state function would need patching; easy to leak by mistake.
- *Remove the car from `evs_connected` ourselves* — would skip EV2Gym's satisfaction/statistics bookkeeping and risk inconsistent metrics.
- *Fork EV2Gym* — more code to maintain and harder for others to reproduce.

Other rules for the wrapper:
- Use a **private** RNG (`np.random.default_rng(seed)`) so sampling early departures does not change EV2Gym's own random draws (cars, prices stay identical for the same seed; `p_early = 0` must reproduce the unwrapped env exactly).
- Only sample `T_actual` with `arrival < T_actual < T_scheduled` (a car leaves at least one step after arriving).

## Consequences
- Oracle (Phase 2): a custom state function reading the wrapper's `T_actual` table instead of `ev.time_of_departure`.
- Target SoC is 100 % in the default config; the proposal's example uses 80 %. Whether to lower `desired_capacity` is a Phase 2 / λ-calibration decision (it changes how much V2G room exists).
- If EV2Gym is upgraded, re-check this ADR (the wrapper relies on `time_of_departure` being read in `cs.step`).
