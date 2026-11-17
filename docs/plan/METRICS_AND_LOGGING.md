# Metrics & Logging Specification

> **Rule: if it might go in the paper, log it.**
> Every training / evaluation script must produce everything below. A new metric is added to the logger — never computed only inside a notebook.

## Where things are saved

1. **Local files** in `results/<run_id>/` — CSV (easy to plot) + TensorBoard. **These are the source of truth**: all paper figures are regenerated from them by `scripts/make_figures.py`, never edited by hand.
2. **Weights & Biases (wandb)** — shared online dashboard for both teammates (live curves, filtering, comparing runs). If wandb is down, nothing is lost because of (1).

`run_id` format: `<agent>_lam<λ>_ptrain<p>_s<seed>_<YYYYmmdd-HHMMSS>`, e.g. `ppo-riskaware_lam10_ptrain0.10_s3_20261201-142530`.

## What is saved

| When | What | File | Used for |
|---|---|---|---|
| Training, every PPO update | mean episode return, episode length, **policy loss, value loss, entropy loss, total loss**, approx KL, clip fraction, explained variance, learning rate, steps/sec, total timesteps | `train_progress.csv` | learning & loss curves, training stability |
| Training, every episode | profit/cost, V2G energy sold, transformer overload, # departures, # early departures, mean & max shortfall, % EVs meeting target, **each reward term separately** (economic / overload / departure penalty) | `train_episodes.csv` | shows *what* the agent learns over time, not just reward |
| Every N steps during training | same metrics on fixed **validation** scenarios + best-model checkpoint | `val_curve.csv` | over-fitting check; picks the checkpoint we test |
| Final test, per episode | all proposal §11 metrics, per test `p_early` | `test_episodes.csv` | main results tables, train/test grid |
| Final test, per EV | arrival, announced vs. true departure, early yes/no, target vs. final SoC, shortfall, energy charged / discharged | `test_evs.csv` | shortfall distributions, % meeting target |
| Final test, per time step (sampled episodes) | per-charger SoC, action, price, total power, time-to-announced-departure | `test_trajectories.csv` | **hedging plots** |
| Once per run | resolved config, seed, git commit, library versions, CPU/GPU model, start/end time, wall-clock | `run_info.json` + one line in [`EXPERIMENT_LOG.md`](../experiments/EXPERIMENT_LOG.md) | reproducibility; compute section of the paper |
| Aggregation over runs | mean ± 95% CI over seeds | `reports/<deliverable>/summary.csv` (**committed**) | final tables & figures |

## Proposal §11 evaluation metrics (must all appear in `test_episodes.csv`)
- % of unexpected departures meeting target SoC
- average SoC shortfall at unexpected departures (and at all departures)
- total charging cost / V2G profit
- total V2G energy discharged
- transformer overload (total and count of violating steps)
- average user satisfaction (EV2Gym metric)
- cumulative RL return

## wandb conventions
- Project: `smartev2g`, under the shared team entity.
- `wandb.init(config=<resolved config>, group=<experiment name>, tags=[agent, f"lam{λ}", f"ptrain{p}", f"seed{s}"], name=<run_id>)`.
- SB3 integration `wandb.integration.sb3.WandbCallback` + `sync_tensorboard=True` → all training losses.
- Our `MetricsCallback` logs per-episode domain metrics and validation metrics.
- Final test tables uploaded as `wandb.Table`; `best` model and `run_info.json` uploaded as wandb Artifacts.
- API key only via `wandb login` or `WANDB_API_KEY` in `.env` (gitignored). Never in code or configs.
- Config switch `logging.wandb: online | offline | disabled` (offline → `wandb sync` later; disabled for tests/CI).

## Storage hygiene
- Keep only `best` and `final` checkpoints per run (the shared disk is nearly full).
- CSV logs are small — always keep them.
- `results/`, `wandb/`, checkpoints are gitignored. Only `reports/` summaries and final figures are committed.
