# Phase 4 — Main Experiments

**Goal:** run the full experiment grid of Deliverable 1.

**Status:** ⚪ Not started

## Branches
| Branch | Content |
|---|---|
| `<name>/exp/p4-lambda-calibration` | Pick the λ_departure range |
| `<name>/exp/p4-main-grid` | Configs + launch of the full grid |

## Tasks
- [ ] **λ calibration:** choose 3–4 values of λ_departure so PPO-Trusting satisfies most drivers when departures are accurate (lit. review §1.1, refinement 1)
- [ ] **Training grid** (configs in `configs/experiments/`):
  - PPO-Trusting: train `p_early = 0`
  - PPO-RiskAware (ours): train `p_early ∈ {0.05, 0.10, 0.20}`
  - PPO-Oracle: train `p_early ∈ {0.05, 0.10, 0.20}`
  - × each λ_departure × seeds `{0..4}` (≥ 5)
- [ ] **Test grid:** every trained agent + AFAP on test `p_early ∈ {0, 0.05, 0.10, 0.20}`, fixed held-out scenario seeds
- [ ] Every run logged in `EXPERIMENT_LOG.md` (config, seed, commit, hardware, wall-clock) and in wandb
- [ ] Sanity check: no failed / NaN runs; re-run any that crashed

## Exit criteria
- All runs complete, evaluated, logged; aggregated `summary.csv` produced.
