# Phase 1 — EV2Gym Exploration & Sanity PPO

**Goal:** prove the training pipeline works on the *unmodified* EV2Gym before we change anything ("simplest possible first result", proposal §12).

**Status:** ✅ Done (pending merge of `leopold/exp/p1-sanity-ppo`) · **Milestone tag:** `v0.1-pipeline`

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
- [x] Evaluate AFAP heuristic and a random policy on fixed seeds (`scripts/evaluate_baselines.py`; fixed days in `configs/eval/scenarios.yaml`: validation seeds 1,000,000–1,000,009, test 2,000,000–2,000,049)
- [x] Train PPO with EV2Gym's default reward, 1 seed; check learning curve rises and beats random → **did not pass** (curve falls then slowly recovers; ≈ random). Cause: the default reward (see ADR-0002). γ variants 0.99/0.999/1.0 (3M steps) did not fix it.
- [x] **Pipeline check** with a reward whose optimum is known (`energy_delivered`, best = AFAP), `configs/experiments/p1_pipeline_check.yaml`: PPO reaches 213.4 kWh/day vs AFAP 213.5 (99.98 %) and random −19.3 → **passed**
- [x] **CPU vs GPU timing test**: CPU with 1 PyTorch thread is fastest (3,709 steps/s vs 2,807 on an H100) → ADR-0003
- [x] Unit tests: env builds, same seed ⇒ same trajectory

## Exit criteria
- PPO trains end-to-end, learning curve improves, beats random. ✅ via the pipeline check (EV2Gym's default reward is unsuitable, documented).
- Env internals documented well enough to design the wrapper. ✅ ADR-0002
