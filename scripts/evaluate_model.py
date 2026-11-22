"""Test a trained agent in other "worlds": different rates of hidden early departures.

This is the train/test grid (proposal §10, Literature Review refinement 3): every
agent is tested on the SAME fixed test days, once per early-departure rate, so we can
separate what an agent learned from where it was tested.

Usage:
    python scripts/evaluate_model.py --run results/<run_id> --p-early 0 0.05 0.1 0.2

Saved in results/<run_id>/eval/: <split>_p<p>_episodes.csv, <split>_p<p>_evs.csv,
summary.csv (one row per world) and run_info.json; also logged to wandb.
(--checkpoint final / pretrained: saved in eval_final/ / eval_pretrained/ instead.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from smartev2g.agents.train import env_kwargs, load_config, load_trained_policy
from smartev2g.eval.evaluate import evaluate_policy, scenario_seeds, world_config
from smartev2g.utils.logging import init_wandb
from smartev2g.utils.run_info import RunInfo

# Per-world numbers printed and logged (all columns are in the CSV files).
SUMMARY_COLUMNS = [
    "return",
    "total_profits",
    "reward_departure_penalty",
    "pct_evs_met_target",
    "mean_shortfall",
    "n_early",
    "pct_early_met_target",
    "mean_shortfall_early",
    "total_energy_discharged",
    "total_transformer_overload",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, required=True, help="results/<run_id> of a trained agent")
    p.add_argument("--p-early", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--checkpoint", choices=["best", "final", "pretrained"], default=None)
    args = p.parse_args()

    cfg = load_config(args.run / "config_resolved.yaml")
    seeds = scenario_seeds(args.split, cfg["scenarios"])
    # Default (best) -> eval/; another checkpoint gets its own folder (eval_final/, ...).
    out = args.run / ("eval" if args.checkpoint in (None, "best") else f"eval_{args.checkpoint}")
    out.mkdir(exist_ok=True)
    info = RunInfo(
        out,
        "scripts/evaluate_model.py",
        args.run / "config_resolved.yaml",
        seeds,
        {k: str(v) for k, v in vars(args).items()},
    )
    policy, checkpoint = load_trained_policy(cfg, args.run, args.checkpoint)

    rows, tables = [], {}
    for p_early in args.p_early:
        world = world_config(cfg, p_early)
        episodes, cars = evaluate_policy(policy, seeds, env_kwargs(world))
        tag = f"{args.split}_p{p_early:g}"
        episodes.to_csv(out / f"{tag}_episodes.csv", index=False)
        cars.to_csv(out / f"{tag}_evs.csv", index=False)
        tables[tag] = episodes
        means = episodes[SUMMARY_COLUMNS].mean()
        rows.append(
            {
                "agent": cfg["agent"],
                "train_p_early": (cfg["env"].get("departures") or {}).get("p_early", 0.0),
                "test_p_early": p_early,
                "checkpoint": checkpoint,
                **means.to_dict(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / f"{args.split}_summary.csv", index=False)
    info.finish(results={"summary": summary.to_dict(orient="records")})  # local files first

    run = init_wandb(
        cfg,
        run_id=f"{args.run.name}_{out.name}",
        run_dir=out,
        group=cfg["name"],
        tags=[cfg["agent"], "cross-eval"],
    )
    import wandb

    run.log({"cross_eval/summary": wandb.Table(dataframe=summary)})
    run.finish()

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary.round(3).T)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
