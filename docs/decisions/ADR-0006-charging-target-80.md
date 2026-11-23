# ADR-0006: Drivers want 80 % (not 100 %) when they leave

- **Date:** 2026-11-20
- **Status:** Accepted (Leopold, 2026-11-20)
- **Author(s):** Leopold Sokoudjou Gatsing

## Context
EV2Gym's example config makes every simulated driver want a **full** battery (`desired_capacity: 1`). That is unrealistic for our setting (daily workplace/fleet AC charging): manufacturers commonly recommend daily charging limits of about 80–90 % for most (NMC) batteries, and 100 % mainly for LFP batteries or before long trips. *(Primary sources to be added to the report; the strongest evidence will be real driver requests in ACN-Data, Phase 6.)* The proposal's own example also uses a target of 0.80.

It also leaves almost no room for the trade-off we study: with a 100 % target, even a perfect-knowledge plan (all prices known, V2G allowed, every car full) beats AFAP by only ~3.4 €/day (7 %) on the test days (Phase 2 plan, "Open decisions").

*Correction (2026-11-21):* that ~3.4 €/day came from a rough one-car-at-a-time estimate that wrongly charged money for charging losses (EV2Gym bills only the energy that enters or leaves the battery). The official planner (`eval/optimum.py`, tested against the simulator) gives −32.1 vs AFAP −46.0 €/day at 100 % (≈ 14 €/day better, λ_dep 500, 50 test days). The decision below still stands: the main reason for 80 % is realism, and 80 % leaves more room for V2G.

## Decision
1. **Every driver's target is 80 % of their battery** (`env.charging_target: 0.8` in experiment configs) for Deliverable 1. 100 % is run as a **sensitivity check** (`--set env.charging_target=1.0`).
2. **How:** the EV2Gym config keeps `desired_capacity: 1`; `ProjectEV2Gym` (`envs/make_env.py`; first called `TargetEV2Gym`) sets each car's target to 0.8 × battery right after `reset()` builds the day's car list, before any car arrives.
3. **New baseline "AFAP-target"** (`agents/baselines.py`): charge every car at full speed **until it reaches its target**, then stop; no V2G. EV2Gym's AFAP keeps charging to 100 %, paying for energy nobody asked for, so it is not a fair "safe" baseline when targets are below 100 %. Both are reported.

## Why (and why not the alternatives)
- **Set targets after reset (chosen):** EV2Gym also uses `desired_capacity` when it *creates* cars — a car that would arrive above the target gets a new, random arrival level (`utilities/utils.py`, `spawn_single_EV`). Changing the config would change the simulated days themselves; our way keeps identical cars and days, so 80 % vs 100 % differ **only** in the target (tested).
- **`desired_capacity: 0.8` in the EV2Gym config:** changes the arrival distribution (see above), mixing two effects.
- **Per-driver targets drawn from a distribution now:** more realistic but needs a justified distribution; planned for Phase 6 with ACN-Data user requests (same hook: set each profile's target after reset).
- **Keep 100 %:** unrealistic and leaves little V2G room.

## Consequences
- Cars arriving above 80 % (common: mean arrival level ≈ 0.74) need no charging and give real V2G room down to 80 %.
  *Update (2026-11-22, ADR-0008):* this came from keeping EV2Gym's arrival rule (full − the driver's energy) with an 80 % target, which removed 20 % of each battery from the driver's real energy need. ADR-0008 sets arrival = 80 % − the driver's energy (mean arrival ≈ 0.55); results with the old rule are the "light charging need" case (release v0.3-p4-light-charging).
- AFAP-target is a much stronger baseline: on 10 validation days it costs −13.4 €/day vs −38.3 €/day for AFAP, with no penalty. PPO must beat AFAP-target.
- AFAP-target can end up to ~2.5 % above target (EV2Gym's real current/efficiency do not scale exactly with a partial action) — about 1 kWh, harmless for a baseline.
- Earlier results (Phases 1–2, before this ADR) used 100 %; they stay valid as the 100 % case.
