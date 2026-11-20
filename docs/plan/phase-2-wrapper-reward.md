# Phase 2 — Shortfall Reward, UnexpectedDepartureWrapper, Oracle

**Goal:** build and *fully test* the core contribution: hidden early departures (POMDP) + the shared reward.

**Status:** 🟡 In progress (reward done on `leopold/feat/p2-shortfall-reward`; shared per-charger policy next, ADR-0004) · **Milestone tag:** `v0.2-wrapper`

## Branches
| Branch | Content |
|---|---|
| `leopold/feat/p2-shortfall-reward` | Reward function (+ first PPO runs, ADR-0004) |
| `leopold/feat/p2-shared-charger-policy` | One policy shared by all chargers, per-charger rewards (ADR-0004) |
| `<name>/feat/p2-departure-wrapper` | UnexpectedDepartureWrapper + synthetic departure model |
| `<name>/feat/p2-oracle-wrapper` | Oracle observation (true departure visible) |

## Tasks
### Reward (`src/smartev2g/rewards/shortfall_reward.py`)
- [x] `R = R_econ − λ_over·max(0, P_tot − P_max)² − λ_dep·Σ_{i∈D_t} max(0, SoC*_i − SoC_i)²`
- [x] Applied to *every* departing EV (scheduled or early); λ values from config (`env.reward_kwargs`)
- [x] Return each term separately in `info` (needed for logging, see METRICS_AND_LOGGING.md) — `RewardTermsInfo` wrapper; `reward_*` columns in train/val/test CSVs
- [x] Overload counts both transformer limits (EV2Gym's helper overstates discharge overload; tested)
- [x] First PPO runs with this reward (no early departures): does not beat AFAP with one site-wide agent → ADR-0004

### Shared per-charger policy (ADR-0004, `leopold/feat/p2-shared-charger-policy`)
- [ ] Per-charger observation (`envs/charger_obs.py`)
- [ ] Per-charger reward split; Σ per-charger rewards = site reward (test)
- [ ] One site → 20 SB3 environments (`envs/charger_vec_env.py`)
- [ ] Evaluation adapter; PPO run vs AFAP / Random

### Departure model (`src/smartev2g/envs/departure_models.py`)
- [ ] Interface: `sample(ev, t_now, rng) -> T_actual or None`
- [ ] `SyntheticDepartureModel(p_early, delta_distribution)` — e.g. Δ = uniform fraction of remaining stay (configurable); later Phase 6 adds `ACNDepartureModel`

### Wrapper (`src/smartev2g/envs/departure_wrapper.py`)
- [ ] On EV connect: with probability `p_early` sample hidden `T_actual < T_scheduled`
- [ ] When `t == T_actual`: force-disconnect the EV through EV2Gym internals (approach from ADR-0002)
- [ ] Observation keeps `T_scheduled` only — `T_actual` never leaks
- [ ] `info` logs: which EVs left early, by how much, their shortfall
- [ ] Own RNG seeded from env seed (reproducible)

### Oracle (`src/smartev2g/envs/oracle_wrapper.py`)
- [ ] Same hidden departures, but observation's time-to-departure uses `T_actual`

### Tests (`tests/`)
- [ ] `p_early = 0` ⇒ trajectories identical to unwrapped env (same seed)
- [ ] `T_actual` never appears in observation (non-oracle)
- [ ] Early EV disconnects exactly at `T_actual`
- [x] Penalty matches hand example: target 0.8, actual 0.5 ⇒ 0.3² = 0.09 (× λ)
- [ ] Empirical early fraction ≈ `p_early` over many EVs
- [ ] Oracle observation shows `T_actual`

## Exit criteria
- All tests pass; a short PPO run trains on the wrapped env without errors.
