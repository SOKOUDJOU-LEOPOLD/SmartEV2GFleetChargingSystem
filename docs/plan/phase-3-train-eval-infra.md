# Phase 3 — Training, Logging & Evaluation Infrastructure

**Goal:** one command trains an agent and saves *every* metric we may need for the paper (local CSV + wandb). One command evaluates and plots.

**Status:** ✅ Done — infrastructure built during Phases 1–2 plus `feat/p3-hedging-trajectories` (trajectories, hedging analysis) and `feat/p3-sweep-figures` (sweep, figures, full-output test, ADR-0007).

## Branches
| Branch | Content |
|---|---|
| `<name>/feat/p3-logging` | Run info, CSV logger, wandb integration, callbacks |
| `<name>/feat/p3-train-eval` | `train.py`, `evaluate.py`, `sweep.py`, metrics, hedging analysis |

## Tasks
- [x] wandb: entity `leopoldgatsing-texas-a-m-university`, project `smartev2g` (CoreWeave server)
- [x] `src/smartev2g/utils/run_info.py` — git commit, library versions, CPU/GPU model, start/end time, wall-clock
- [x] `src/smartev2g/utils/logging.py` + SB3 callbacks (done as our own `EpisodeMetricsCallback`, `ValidationCallback` + SB3 `CheckpointCallback`, TensorBoard + CSV logger): `WandbCallback` (sync TensorBoard), `EvalCallback` (validation + best model), custom `MetricsCallback` (per-episode domain metrics) — writes every file in [METRICS_AND_LOGGING.md](METRICS_AND_LOGGING.md)
- [x] Config switch `logging.wandb: online | offline | disabled`
- [x] `scripts/train.py --config <yaml> --seed <int>` — SB3 PPO, `VecNormalize`, checkpoints (`best` + `final` only), TensorBoard + CSV logger
- [x] Evaluation on fixed held-out test scenarios — done as `scripts/evaluate_model.py` (train/test grid; `--trajectories` writes `<split>_p<p>_trajectories.csv.gz`), `evaluate_baselines.py`, `evaluate_optimum.py`
- [x] `src/smartev2g/eval/metrics.py` — all proposal §11 metrics + trade timing (`TradeTracker`) + per-step trajectories (`TrajectoryRecorder`)
- [x] `src/smartev2g/eval/hedging.py` + `scripts/analyze_hedging.py` — battery margin & energy sold over the announced stay (common clock), risk-window summary, figure
- [x] `scripts/sweep.py` — launches a grid of runs as detached parallel processes (`setsid`), resumable (skips finished jobs, resumes from checkpoints, evaluates after training, `status.csv`, max 2 attempts per job)
- [x] `scripts/make_figures.py` — all figures generated from CSVs (train/test grid with AFAP-target + optimum lines, learning curves, hedging; `summary.csv` with mean/std/95 % CI, `paired.csv` with paired seed differences + t-test); shared style in `eval/plots.py`
- [x] Small PPO hyperparameter check on PPO-Trusting (n_steps, learning rate, ent_coef); freeze choice in ADR — check done (PPO gap diagnosis, setting B); frozen in **ADR-0007** (`configs/experiments/p4_main.yaml`)
- [x] Test: a tiny run produces every file and every column listed in the spec (`tests/test_outputs.py`)

## Exit criteria
- One PPO-Trusting run trained, evaluated and plotted automatically; visible in wandb.
