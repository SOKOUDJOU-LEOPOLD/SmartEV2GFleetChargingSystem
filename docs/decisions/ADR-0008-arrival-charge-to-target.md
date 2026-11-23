# ADR-0008: Cars arrive "target − the driver's energy" below the target

- **Date:** 2026-11-22
- **Status:** Accepted (Leopold, 2026-11-22)
- **Author(s):** Leopold Sokoudjou Gatsing

## Context
EV2Gym does not draw a car's arrival charge directly. It draws **how much energy the driver takes** (kWh) from real Dutch charging sessions (ElaadNL, `mean-demand-per-arrival.csv`, by arrival time and site type; workplace: ~11–15 kWh on average, ±50 % spread). It then sets

  arrival = full battery − that energy  (`utilities/utils.py`, `spawn_single_EV`)

which assumes every driver charges to **100 %**. ADR-0006 moved every driver's target to **80 %** but kept EV2Gym's arrival levels, so 20 % of each battery was silently removed from the driver's energy need. In the Phase 4 grid (release `v0.3-p4-light-charging`, `reports/p4-main-grid/`):
- cars arrived at **74.7 %** on average, only ~5 points below the target;
- **37 %** of cars arrived already above 80 % (no charging needed at all);
- early leavers could therefore miss only a few points, which makes hedging look less important than it is.

No charging dataset records a car's arrival charge or target (a charger cannot see inside the battery); ElaadNL and ACN-Data record the **energy** that flowed (ACN also the energy drivers *asked for*). The energy is the measured part; the target is our modelling choice (ADR-0006).

## Decision
1. **arrival = target − the energy the driver takes**, with the energy exactly as EV2Gym drew it from the data (`battery − EV2Gym's arrival`), i.e. every car's arrival moves down by (1 − target) × battery = 20 points at an 80 % target.
2. Never below EV2Gym's minimum battery level (`min_battery_capacity`, 5 kWh); on the 50 test days this affects 1 car of 722.
3. **How:** `env.arrival_rule: target` (`ProjectEV2Gym.shift_arrival`, `envs/make_env.py`), applied after `reset()` together with the target, so cars, arrival/departure times and battery sizes stay identical; only the arrival level changes (tested, `tests/test_arrival_rule.py`). Default `arrival_rule: ev2gym` keeps the old behaviour, so earlier configs reproduce exactly.
4. New frozen config `configs/experiments/p4_arrival.yaml` = `p4_main.yaml` + this one line (ADR-0007: a changed setup gets a new file + ADR). The Phase 4 grid on `p4_main.yaml` stays in the paper as the **"light charging need"** case.

## Why (and why not the alternatives)
- **Shift the arrival (chosen):** keeps every number that comes from data (energy, arrival time, stay, battery size); only our own choice (80 %) moves. One sentence in the paper: "each driver adds the energy measured in the ElaadNL data, up to an 80 % target".
- **Target back to 100 %:** rejected in ADR-0006 (unrealistic for daily charging, little V2G room).
- **Draw arrival levels ourselves (e.g. 30–50 %):** no data behind the numbers; a reviewer could say the world was tuned to favour hedging.
- **Per-driver targets (e.g. 70–90 %):** ElaadNL has no requests; planned with ACN-Data `kWhRequested` in Phase 6.

## Consequences
- Cars arrive at ~55 % on average (test days: 54.7 %, 4 % below 30 %); every car needs charging; early leavers can miss real energy, so hedging is tested properly.
- Cars must charge ~25 points in their stay; with chargers of up to 22 kW (32 A, 400 V, 3 phases; many cars accept less) and stays of several hours this is reachable for almost all cars (AFAP-target check below), but there is less time for V2G trading (the optimum earns less).
- Teacher data and optimum / AFAP-target references are recomputed (separate cache key and result folders `optimum-<split>-lamdep<λ>-arrivaltarget`).
- Focused re-run (`configs/sweeps/p4_arrival_grid.yaml`, `p4_arrival_oracle.yaml`): λ {500, 1000} × {Trusting, RiskAware p0.2} × 10 seeds + PPO-Oracle λ 500 p0.2 × 10 seeds.
