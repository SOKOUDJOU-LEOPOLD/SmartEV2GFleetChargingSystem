# Phase 3 — Training, Logging & Evaluation Infrastructure

**Goal:** one command trains an agent and saves *every* metric we may need for the paper (local CSV + wandb). One command evaluates and plots.

**Status:** ⚪ Not started

## Branches
| Branch | Content |
|---|---|
| `<name>/feat/p3-logging` | Run info, CSV logger, wandb integration, callbacks |
| `<name>/feat/p3-train-eval` | `train.py`, `evaluate.py`, `sweep.py`, metrics, hedging analysis |

## Tasks
- [ ] wandb: shared team/entity + project `smartev2g` (needs user's `wandb login` and team name; Yeshas invited)
- [ ] `src/smartev2g/utils/run_info.py` — git commit, library versions, CPU/GPU model, start/end time, wall-clock
- [ ] `src/smartev2g/utils/logging.py` + SB3 callbacks: `WandbCallback` (sync TensorBoard), `EvalCallback` (validation + best model), custom `MetricsCallback` (per-episode domain metrics) — writes every file in [METRICS_AND_LOGGING.md](METRICS_AND_LOGGING.md)
- [ ] Config switch `logging.wandb: online | offline | disabled`
- [ ] `scripts/train.py --config <yaml> --seed <int>` — SB3 PPO, `VecNormalize`, checkpoints (`best` + `final` only), TensorBoard + CSV logger
- [ ] `scripts/evaluate.py` — fixed held-out test scenarios; writes `test_episodes.csv`, `test_evs.csv`, `test_trajectories.csv`
- [ ] `src/smartev2g/eval/metrics.py` — all proposal §11 metrics
- [ ] `src/smartev2g/eval/hedging.py` — SoC margin & discharge vs. time-to-announced-departure
- [ ] `scripts/sweep.py` — launches a grid of runs as detached parallel processes (`setsid`), resumable
- [ ] `scripts/make_figures.py` — all figures generated from CSVs
- [ ] Small PPO hyperparameter check on PPO-Trusting (n_steps, learning rate, ent_coef); freeze choice in ADR
- [ ] Test: a tiny run produces every file and every column listed in the spec

## Exit criteria
- One PPO-Trusting run trained, evaluated and plotted automatically; visible in wandb.
