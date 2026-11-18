"""Evaluate the non-learning baselines (AFAP, Random) on the fixed test days.

Gives the reference numbers PPO must beat (Phase 1 sanity check).

Usage:
    python scripts/evaluate_baselines.py --config configs/experiments/p1_sanity_ppo.yaml

Saved in results/<run_id>/: test_episodes.csv, test_evs.csv, run_info.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from smartev2g.agents.baselines import AFAPPolicy, RandomPolicy
from smartev2g.agents.train import env_kwargs, load_config
from smartev2g.eval.evaluate import evaluate_policy, scenario_seeds
from smartev2g.utils.run_info import RESULTS_DIR, RunInfo, make_run_dir, make_run_id

# Columns printed in the console summary (all columns are saved in the CSV).
SUMMARY_COLUMNS = [
    "return",
    "total_profits",
    "total_energy_charged",
    "total_energy_discharged",
    "average_user_satisfaction",
    "pct_evs_met_target",
    "mean_shortfall",
    "total_transformer_overload",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="experiment config (configs/experiments/)")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--random-seed", type=int, default=0, help="seed of the random policy")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args()

    cfg = load_config(args.config)
    seeds = scenario_seeds(args.split, cfg["scenarios"])
    run_dir = make_run_dir(make_run_id(f"baselines-{args.split}"), root=args.results_dir)
    info = RunInfo(run_dir, "scripts/evaluate_baselines.py", args.config, seeds, vars(args))
    info.data["env_config"] = load_config(cfg["env"]["config"])

    episodes, cars = [], []
    for policy in [AFAPPolicy(), RandomPolicy(seed=args.random_seed)]:
        ep, ev = evaluate_policy(policy, seeds, env_kwargs(cfg))
        episodes.append(ep)
        cars.append(ev)
    episodes_df = pd.concat(episodes, ignore_index=True)
    pd.concat(cars, ignore_index=True).to_csv(run_dir / f"{args.split}_evs.csv", index=False)
    episodes_df.to_csv(run_dir / f"{args.split}_episodes.csv", index=False)

    summary = episodes_df.groupby("policy")[SUMMARY_COLUMNS].agg(["mean", "std"])
    info.finish(
        results={
            pol: {col: float(v) for col, v in row.xs("mean", level=1).items()}
            for pol, row in summary.iterrows()
        }
    )
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(f"{args.split} days: {len(seeds)} (seeds {seeds[0]}..{seeds[-1]})")
        print(summary.xs("mean", axis=1, level=1).round(3).T)
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
