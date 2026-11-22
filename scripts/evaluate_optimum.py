"""Perfect-knowledge reference: the best possible score on each test day (Phase 2).

Plays three non-learning policies on the SAME fixed test days, once per world (rate of
hidden early departures), with the reward of the given experiment config (λ values,
80 % target):
- AFAP-target         safe rule (charge to the target as fast as possible), the floor;
- Optimum-announced   perfect plan that TRUSTS the announced departures;
- Optimum-real        perfect plan that KNOWS the real departures = the ceiling.
(eval/optimum.py explains the plan.) Optimum-real − Optimum-announced = the most that
knowing early departures is worth; where an agent lands between them shows how much
of that it captures.

Usage:
    python scripts/evaluate_optimum.py --config configs/experiments/p2_site_laxity_ppo.yaml \
        --p-early 0 0.05 0.1 0.2 [--set env.reward_kwargs.lambda_dep=1000]

Saved in results/<run_id>/: <split>_p<p>_episodes.csv, <split>_p<p>_evs.csv,
<split>_summary.csv (one row per policy and world) and run_info.json; also wandb.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from smartev2g.agents.baselines import AFAPToTargetPolicy
from smartev2g.agents.train import apply_settings, env_kwargs, load_config
from smartev2g.eval.evaluate import evaluate_policy, scenario_seeds, world_config
from smartev2g.eval.optimum import PerfectPlanPolicy
from smartev2g.utils.logging import init_wandb
from smartev2g.utils.run_info import RESULTS_DIR, RunInfo, make_run_dir, make_run_id

SUMMARY_COLUMNS = [
    "return",
    "predicted_return",
    "total_profits",
    "reward_overload_penalty",
    "reward_departure_penalty",
    "pct_evs_met_target",
    "mean_shortfall",
    "n_early",
    "pct_early_met_target",
    "mean_shortfall_early",
    "total_energy_charged",
    "total_energy_discharged",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="experiment config (env, reward, λ)")
    p.add_argument("--p-early", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args()

    cfg = load_config(args.config)
    settings = {}
    for item in args.set:  # same syntax as scripts/train.py
        key, sep, value = item.partition("=")
        if not sep:
            p.error(f"--set expects KEY=VALUE, got {item!r}")
        settings[key] = yaml.safe_load(value)
    apply_settings(cfg, settings)
    lam = cfg["env"]["reward_kwargs"]
    seeds = scenario_seeds(args.split, cfg["scenarios"])
    tag = f"lamdep{lam['lambda_dep']:g}"
    run_dir = make_run_dir(make_run_id(f"optimum-{args.split}-{tag}"), root=args.results_dir)
    info = RunInfo(run_dir, "scripts/evaluate_optimum.py", args.config, seeds, vars(args))
    info.data["resolved_config"] = cfg

    rows, tables = [], {}
    for p_early in args.p_early:
        kwargs = env_kwargs(world_config(cfg, p_early))
        episodes, cars = [], []
        policies = [
            AFAPToTargetPolicy(),
            PerfectPlanPolicy("announced", lam["lambda_dep"], lam["lambda_over"]),
            PerfectPlanPolicy("real", lam["lambda_dep"], lam["lambda_over"]),
        ]
        for policy in policies:
            ep, ev = evaluate_policy(policy, seeds, kwargs)
            # What the plan expected (same days, same order); the simulator decides `return`.
            predicted = getattr(policy, "predicted", None)
            ep["predicted_return"] = [d["return"] for d in predicted] if predicted else float("nan")
            episodes.append(ep)
            cars.append(ev)
        name = f"{args.split}_p{p_early:g}"
        episodes_df = pd.concat(episodes, ignore_index=True)
        episodes_df.to_csv(run_dir / f"{name}_episodes.csv", index=False)
        pd.concat(cars, ignore_index=True).to_csv(run_dir / f"{name}_evs.csv", index=False)
        tables[name] = episodes_df
        for policy_name, ep in episodes_df.groupby("policy", sort=False):
            rows.append(
                {
                    "policy": policy_name,
                    "test_p_early": p_early,
                    "lambda_dep": lam["lambda_dep"],
                    **ep[SUMMARY_COLUMNS].mean().to_dict(),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / f"{args.split}_summary.csv", index=False)
    info.finish(results={"summary": summary.to_dict(orient="records")})  # local files first

    run = init_wandb(
        {**cfg, "agent": "Optimum"},
        run_id=run_dir.name,
        run_dir=run_dir,
        group=cfg["name"],
        tags=["optimum", "baseline", tag],
    )
    import wandb

    run.log({"optimum/summary": wandb.Table(dataframe=summary)})
    run.finish()

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary.round(3).to_string(index=False))
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
