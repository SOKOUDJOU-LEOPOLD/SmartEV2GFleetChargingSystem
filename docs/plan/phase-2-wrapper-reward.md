# Phase 2 — Shortfall Reward, UnexpectedDepartureWrapper, Oracle

**Goal:** build and *fully test* the core contribution: hidden early departures (POMDP) + the shared reward.

**Status:** 🟡 In progress (reward done on `leopold/feat/p2-shortfall-reward`; shared per-charger policy tried (ADR-0004); central agent with per-charger credit next (ADR-0005)) · **Milestone tag:** `v0.2-wrapper`

## Modelling assumptions (state these in the paper's method section)
1. **Fleet setting.** The site serves a company **fleet**: the company owns the cars, pays for their electricity and keeps all V2G revenue. Drivers are employees who need enough charge when they leave; their interest enters the reward only through the departure (shortfall) penalty.
2. **Prices.** Buying and selling energy use the same real Dutch day-ahead wholesale price for that hour (EV2Gym, `Netherlands_day-ahead-2015-2024.csv`, €/MWh → €/kWh; `discharge_price_factor: 1`). No network fees, taxes or retail mark-ups.
3. **No driver tariff or V2G compensation.** Drivers neither pay per kWh nor receive money for V2G use of their battery. *Extension:* for privately owned cars, add a driver tariff and/or a compensation per kWh discharged to the reward (makes V2G less attractive to the operator).
4. **Profit = the operator's net energy cost/revenue for the whole site per day**; penalty terms are "points", not money; λ is the exchange rate between them.
5. **Transformer** limit ±100 kW, no other loads or solar on it (config); overload is penalised (λ_over), not prevented.

## Open decisions (decide before the departure-wrapper experiments)
- [ ] **Charging target per driver.** Today every simulated driver wants 100 % (`desired_capacity: 1` in the EV2Gym config is one global value), which leaves no battery room for V2G: even a perfect-knowledge plan beats AFAP by only ~3.4 €/day (7 %) on the test days. Options: (a) keep 100 %; (b) 80 % for everyone (the proposal's example); (c) **each driver gets their own target when they plug in** (e.g. drawn from 60–100 %, later from ACN-Data user requests in Phase 6) — closest to reality, set by the same wrapper as hidden departures. Laxity already uses each car's own target.
- [ ] **λ values.** No study gives a "true" λ_dep: sweep several values and report profit-vs-shortfall trade-off curves (Literature Review, refinement 1), choose the range so PPO-Trusting satisfies most drivers without early departures (Phase 4); optionally anchor λ in euros via the cost of getting the missing energy elsewhere (e.g. public charging price, to be looked up).
- [ ] **Perfect-knowledge reference.** Add the per-day optimum with perfect knowledge of prices and departures (linear/quadratic program) as an upper bound, so results can be reported as "% of the gap between AFAP and the best possible closed".

## Branches
| Branch | Content |
|---|---|
| `leopold/feat/p2-shortfall-reward` | Reward function (+ first PPO runs, ADR-0004) |
| `leopold/feat/p2-shared-charger-policy` | One policy shared by all chargers, per-charger rewards (ADR-0004), seed fix |
| `leopold/feat/p2-central-credit-ppo` | Central agent with per-charger credit (ADR-0005) |
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
- [x] Per-charger observation (`envs/charger_obs.py`)
- [x] Per-charger reward split; Σ per-charger rewards = site reward (test)
- [x] One site → 20 SB3 environments (`envs/charger_env.py`)
- [x] Evaluation adapter; PPO run vs AFAP / Random → drivers well served, but uncoordinated chargers overload the transformer → ADR-0005

### Central agent with per-charger credit (ADR-0005, `leopold/feat/p2-central-credit-ppo`)
- [ ] `CreditPPO`: 20-output critic, per-charger GAE and PPO loss (`agents/credit_ppo.py`)
- [ ] Site observation + vector rewards env; `agent_design: central_credit`
- [ ] Tests: per-charger GAE sums to site GAE; log-prob decomposition; end-to-end run
- [ ] λ_over = 0.1, fixed reward scale; PPO run vs AFAP / Random

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
