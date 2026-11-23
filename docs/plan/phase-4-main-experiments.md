# Phase 4 — Main Experiments

**Goal:** run the full experiment grid of Deliverable 1.

**Status:** 🟢 Main grid done (2026-11-22, 140 runs, results in [`reports/p4-main-grid/`](../../reports/p4-main-grid/README.md)). ADR-0008 (arrival charge consistent with the 80 % target) accepted; focused re-run in progress (`configs/sweeps/p4_arrival_*.yaml`). Every run uses the frozen setup of ADR-0007 (`configs/experiments/p4_main.yaml`, PPO with imitation warm start) and is launched with `scripts/sweep.py` (a sweep file in `configs/sweeps/`); figures and tables with `scripts/make_figures.py`.

## Branches
| Branch | Content |
|---|---|
| `<name>/exp/p4-lambda-calibration` | Pick the λ_departure range |
| `<name>/exp/p4-main-grid` | Configs + launch of the full grid |

## Tasks
- [x] **λ calibration:** λ_dep {200, 500, 1000} from the Phase 2 calibration; choose 3–4 values of λ_departure so PPO-Trusting satisfies most drivers when departures are accurate (lit. review §1.1, refinement 1)
- [x] **Training grid** (configs in `configs/experiments/`):
  - PPO-Trusting: train `p_early = 0`
  - PPO-RiskAware (ours): train `p_early ∈ {0.05, 0.10, 0.20}`
  - PPO-Oracle: train `p_early ∈ {0.05, 0.10, 0.20}` (observation shows the REAL departure; moved here from Phase 2 — needs a state function reading `env.departure_plan`)
  - × each λ_departure × seeds `{0..4}` (≥ 5). Done: λ {200, 500, 1000} × {Trusting, RiskAware p 0.05 / 0.1 / 0.2} × seeds 0–9 (120 runs) + PPO-Oracle p 0.1 / 0.2 at λ 500 × seeds 0–9 (20 runs; `configs/sweeps/`)
- [x] **Test grid:** every trained agent + AFAP on test `p_early ∈ {0, 0.05, 0.10, 0.20}`, fixed held-out scenario seeds
- [x] Every run logged in `EXPERIMENT_LOG.md` (config, seed, commit, hardware, wall-clock) and in wandb
- [x] Sanity check: no failed / NaN runs; re-run any that crashed (none failed)
- [x] **ADR-0008:** arrival charge = 80 % − the driver's energy from the data (today cars arrive at 74.7 %, only ~5 points below target), then a focused re-run: λ 500 / 1000 × {Trusting, RiskAware p0.2} × 10 seeds + PPO-Oracle λ 500 (config `p4_arrival.yaml`, `arrival_rule: target`)
- [ ] Focused re-run done, evaluated and reported (`reports/p4-arrival/`)

## Exit criteria
- All runs complete, evaluated, logged; aggregated `summary.csv` produced.
