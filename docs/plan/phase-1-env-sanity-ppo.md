# Phase 1 — EV2Gym Exploration & Sanity PPO

**Goal:** prove the training pipeline works on the *unmodified* EV2Gym before we change anything ("simplest possible first result", proposal §12).

**Status:** 🟡 In progress (env setup merged; V2G car-model fix on `leopold/fix/p1-v2g-ev-specs`; baselines + sanity PPO next) · **Milestone tag:** `v0.1-pipeline`

## Branches
| Branch | Content |
|---|---|
| `leopold/feat/p1-env-setup` | Env config, seeded env factory, exploration notes |
| `<name>/exp/p1-sanity-ppo` | AFAP + random baselines, first PPO run, CPU vs GPU timing |

## Tasks
- [x] Pin `ev2gym` version; record it in ADR-0002
- [x] `configs/env/v2g_profitmax_20port.yaml` — based on EV2Gym's `example_config_files/V2GProfitMax.yaml` (25 chargers, 1 transformer, 15-min steps, 112 steps/episode) → set 20 chargers. keep `ev_specs_file` but write it as `pkg://ev2gym/data/ev_specs_v2g_enabled2024.json` (resolved by `make_env`). ⚠️ Do **not** remove it: EV2Gym would fall back to car models that cannot discharge, silently disabling V2G (bug found and fixed on `leopold/fix/p1-v2g-ev-specs`, see ADR-0002)
- [x] `src/smartev2g/envs/make_env.py` — seeded env factory (one function everyone uses)
- [x] Inspect and document: observation vector (`V2G_profit_max` state function), action space, episode length, timestep size
- [x] Document EV2Gym internals we will need for the wrapper: where departure time is stored (`ev.time_of_departure`), desired capacity, when/how EVs are removed, where per-EV results are recorded → `docs/decisions/ADR-0002-env-internals.md`
- [ ] Evaluate AFAP heuristic and a random policy on fixed seeds
- [ ] Train PPO with EV2Gym's default reward, 1 seed; check learning curve rises and beats random
- [ ] **CPU vs GPU timing test**: same short training with `device="cpu"` vs `device="cuda"`; record steps/sec in EXPERIMENT_LOG + ADR; choose the faster (expected: CPU — EV2Gym runs on CPU and the PPO network is tiny)
- [x] Unit tests: env builds, same seed ⇒ same trajectory

## Exit criteria
- PPO trains end-to-end, learning curve improves, beats random.
- Env internals documented well enough to design the wrapper.
