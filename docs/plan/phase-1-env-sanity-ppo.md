# Phase 1 — EV2Gym Exploration & Sanity PPO

**Goal:** prove the training pipeline works on the *unmodified* EV2Gym before we change anything ("simplest possible first result", proposal §12).

**Status:** ⚪ Not started · **Milestone tag:** `v0.1-pipeline`

## Branches
| Branch | Content |
|---|---|
| `<name>/feat/p1-env-setup` | Env config, seeded env factory, exploration notes |
| `<name>/exp/p1-sanity-ppo` | AFAP + random baselines, first PPO run, CPU vs GPU timing |

## Tasks
- [ ] Pin `ev2gym` version; record it in ADR-0002
- [ ] `configs/env/v2g_profitmax_20port.yaml` — based on EV2Gym's `V2GProfitPlusLoads.yaml` example: 20 chargers, one transformer, V2G enabled
- [ ] `src/smartev2g/envs/make_env.py` — seeded env factory (one function everyone uses)
- [ ] Inspect and document: observation vector (`V2G_profit_max` state function), action space, episode length, timestep size
- [ ] Document EV2Gym internals we will need for the wrapper: where departure time is stored (`ev.time_of_departure`), desired capacity, when/how EVs are removed, where per-EV results are recorded → `docs/decisions/ADR-0002-env-internals.md`
- [ ] Evaluate AFAP heuristic and a random policy on fixed seeds
- [ ] Train PPO with EV2Gym's default reward, 1 seed; check learning curve rises and beats random
- [ ] **CPU vs GPU timing test**: same short training with `device="cpu"` vs `device="cuda"`; record steps/sec in EXPERIMENT_LOG + ADR; choose the faster (expected: CPU — EV2Gym runs on CPU and the PPO network is tiny)
- [ ] Unit tests: env builds, same seed ⇒ same trajectory

## Exit criteria
- PPO trains end-to-end, learning curve improves, beats random.
- Env internals documented well enough to design the wrapper.
