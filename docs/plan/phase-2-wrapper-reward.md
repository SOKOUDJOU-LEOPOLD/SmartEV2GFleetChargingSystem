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
- [ ] **Charging target per driver.** Today every simulated driver wants 100 % (`desired_capacity: 1` in the EV2Gym config is one global value), which leaves no battery room for V2G: even a perfect-knowledge plan beats AFAP by only ~3.4 €/day (7 %) on the test days *(corrected later: ≈ 14 €/day, see ADR-0006)*. Options: (a) keep 100 %; (b) 80 % for everyone (the proposal's example); (c) **each driver gets their own target when they plug in** (e.g. drawn from 60–100 %, later from ACN-Data user requests in Phase 6) — closest to reality, set by the same wrapper as hidden departures. Laxity already uses each car's own target.
- [x] **First λ_dep calibration (80 % target, PPO-Trusting, 1 seed, 3M steps; EXPERIMENT_LOG):**

  | λ_dep | Money €/day | Mean SoC at departure | Cars within 5 % of target | within 10 % |
  |---|---|---|---|---|
  | AFAP-target | −15.1 | ≥ 0.80 | 100 % | 100 % |
  | 50 | +23.9 | 0.67 | 17 % | 38 % |
  | 200 | −0.7 | 0.77 | 65 % | 96 % |
  | **500** | **−6.3** | **0.79** | **87 %** | **99 %** |
  | 1000 | −10.1 | 0.82 | 92 % | 100 % |

  **Decided (Leopold, 2026-11-21):** λ_dep ∈ {200, 500, 1000} for the main grid, **500 as development default** (λ = 1000 = "drivers first" setting in the paper) (most drivers within 5 % of target, saves ~9 €/day vs AFAP-target). Confirm over ≥ 5 seeds in Phase 4.
- [ ] **λ values (remaining).** No study gives a "true" λ_dep: sweep several values and report profit-vs-shortfall trade-off curves (Literature Review, refinement 1), choose the range so PPO-Trusting satisfies most drivers without early departures (Phase 4); optionally anchor λ in euros via the cost of getting the missing energy elsewhere (e.g. public charging price, to be looked up).
- [x] **Perfect-knowledge reference.** Add the per-day optimum with perfect knowledge of prices and departures (linear/quadratic program) as an upper bound, so results can be reported as "% of the gap between AFAP and the best possible closed".
  **Done:** `eval/optimum.py` (`plan_day`, `PerfectPlanPolicy`; LP with HiGHS, squares as tangent cuts, EV2Gym battery maths checked against the simulator) + `scripts/evaluate_optimum.py`. Two versions: Optimum-real (knows real departures = ceiling) and Optimum-announced (trusts announced ones). λ 500, test p 0.1: AFAP-target −15.1, Optimum-announced +2.4, Optimum-real +11.0; PPO agents ≈ −15. Optional later: Optimum-risk (knows the risk, not which car) = the best possible hedger.
- [x] **Why PPO captures so little (diagnosis, EXPERIMENT_LOG 2026-11-21).** PPO moves as much energy as the optimum but does not time it (buy/sell spread 0.2 c vs 5.3 c) and flips charge↔discharge ~20×/car/day, mostly while holding cars near 80 %. Validation plateaus at ~850k steps; updates were too large (approx_kl 0.14–0.27).
  - Calmer PPO updates (setting B: lr 1e-4, n_epochs 5, batch 2048; `target_kl` now in the config, off by default): healthy training, +1–2 points.
  - Day-ahead price view (`V2G_profit_max_laxity_dayahead`: all published prices of the next 24 h, tomorrow's from 13:00): 3× better spread, fewer flips, more cars served, same score.
  - Trade-timing metrics in every evaluation: `avg_buy_price`, `avg_sell_price`, `trade_spread`, `switches_per_car`.
- [x] **Imitation warm start (`agents/imitation.py`, `configs/experiments/p2_imitation_ppo.yaml`).** Teach PPO by example first: clone Optimum-announced (uses only what the agent sees: day-ahead prices + announced departures) on ~2,000 training days, pre-train the critic on the same days, then continue with PPO (setting B, day-ahead view). Same warm start for PPO-Trusting and PPO-RiskAware.
  **Results (3 seeds, EXPERIMENT_LOG 2026-11-22):** PPO now trades (spread 3.9–4.5 c vs 1.4 c, ~4 switches/car, positive money). Trusting test p 0: −9.7 (was −12.9). RiskAware hedges: early leavers reaching 80 % 59–61 % vs 51–53 %, sells 7–8 kWh/day less; on seeds 0–1 it beats Trusting at p 0.1 / 0.2 (−13.1 / −16.3 vs −13.8 / −18.9), but RiskAware seed 2 failed (−20 to −28), so the score comparison is open. Next: ≥ 5 seeds and a look at seed instability.
- [x] **5-seed check with warm start (6M steps, 30 validation days; EXPERIMENT_LOG 2026-11-22).** RiskAware − Trusting: −0.8 / +0.7 / +1.9 / +3.2 at test p 0 / 0.05 / 0.1 / 0.2 (wins on 1 / 2 / 3 / 4 of 5 seeds): the predicted pattern (small cost without early departures, growing benefit with more), with clear hedging (early leavers at 80 %: ~56 % vs ~45 %), but not yet statistically significant (p 0.17 at test p 0.2). RiskAware beats AFAP-target in all four worlds.

## Branches
| Branch | Content |
|---|---|
| `leopold/feat/p2-shortfall-reward` | Reward function (+ first PPO runs, ADR-0004) |
| `leopold/feat/p2-shared-charger-policy` | One policy shared by all chargers, per-charger rewards (ADR-0004), seed fix |
| `leopold/feat/p2-central-credit-ppo` | Central agent with per-charger credit (ADR-0005), laxity observation, main-agent choice |
| `leopold/feat/p2-target-80` | 80 % charging target, AFAP-to-target baseline |
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
- [x] `CreditPPO`: 20-output critic, per-charger GAE and PPO loss (`agents/credit_ppo.py`)
- [x] Site observation + vector rewards env; `agent_design: central_credit`
- [x] Tests: per-charger GAE sums to site GAE; log-prob decomposition; end-to-end run
- [x] λ_over = 0.1, fixed reward scale; PPO run vs AFAP / Random → no better than the site agent
- [x] **Laxity observation** (`V2G_profit_max_laxity`: steps still needed + spare time per car): site agent −33.4, credit −35.6, AFAP −46.0 → **main agent = site agent + laxity** (`configs/experiments/p2_site_laxity_ppo.yaml`, ADR-0005 Outcome)

### Departure model (`src/smartev2g/envs/departure_models.py`)
- [x] Interface: `sample(arrival, announced, rng) -> real departure step or None` (decided once per car right after reset)
- [x] `SyntheticDepartureModel(p_early, cut_min=0.1, cut_max=0.5)` — an early car cuts a uniform 10–50 % of its announced stay; both random numbers always drawn, so early sets are nested across p_early (same days in every world). Phase 6 adds `ACNDepartureModel`

### Hidden early departures (built into `ProjectEV2Gym`, `envs/make_env.py`, not a separate wrapper file)
- [x] After reset: with probability `p_early` each car gets a hidden real departure `T_actual < T_scheduled` (`env.departure_plan`)
- [x] At `t == T_actual`: just-in-time swap of `ev.time_of_departure`, so EV2Gym removes the car itself (ADR-0002); `max_energy_AFAP` refreshed
- [x] Observation keeps `T_scheduled` only — `T_actual` never leaks
- [x] Logs: per car `announced_departure`, `early`, `early_by_steps`, shortfall; per day `n_early`, `pct_early_met_target`, `mean_shortfall_early` (eval) and `day_seed`, `n_early` (training)
- [x] Own RNG `default_rng([day_seed, 20261121])` (reproducible, does not disturb EV2Gym's draws)
- [x] Train/test grid: `scripts/evaluate_model.py --p-early 0 0.05 0.1 0.2`; baselines `--p-early`
- [x] First results (λ_dep 500, 3 seeds, EXPERIMENT_LOG): PPO-RiskAware (train p 0.1 or 0.2) vs PPO-Trusting differ by less than the seed spread — no clear hedging yet.
- [x] λ_dep 1000 check (3 seeds): PPO-RiskAware (train p 0.2) hedges — sells less, pays ~1.8 €/day, early leavers reaching 80 % 39 % vs 31 %, better score in every risky world (gap ≈ 1 std; needs ≥ 5 seeds in Phase 4). Both PPO agents still score below AFAP-target (only ~half the cars reach 80 %) → perfect-knowledge optimum next

### Oracle (`src/smartev2g/envs/oracle_wrapper.py`)
- [ ] Same hidden departures, but observation's time-to-departure uses `T_actual`

### Tests (`tests/`)
- [x] `p_early = 0` ⇒ trajectories identical to unwrapped env (same seed)
- [x] `T_actual` never appears in observation (non-oracle)
- [x] Early EV disconnects exactly at `T_actual`
- [x] Penalty matches hand example: target 0.8, actual 0.5 ⇒ 0.3² = 0.09 (× λ)
- [x] Empirical early fraction ≈ `p_early` over many EVs
- [ ] Oracle observation shows `T_actual`

## Exit criteria
- All tests pass; a short PPO run trains on the wrapped env without errors.
