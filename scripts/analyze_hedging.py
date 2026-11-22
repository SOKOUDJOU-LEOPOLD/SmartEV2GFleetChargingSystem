"""Hedging analysis: how agents manage a car's battery over its stay (Deliverable 1, item 7).

Compares groups of trained agents (e.g. PPO-Trusting vs PPO-RiskAware, several seeds
each) on the per-step trajectories saved by `evaluate_model.py --trajectories`.

Usage:
    python scripts/analyze_hedging.py \
        --group "PPO-Trusting=results/*_p2-imitation-ppo-6M-val30_s?" \
        --group "PPO-RiskAware=results/*_p2-imitation-ppo-riskaware-p0.1-6M-val30_s?" \
        --p-early 0.1 0.2

Saved in results/<run_id>/ (per test world p):
- profile_p<p>.csv      per agent, seed and stay-fraction bin: margin, kWh sold, share selling
- risk_window_p<p>.csv  per agent and seed: behaviour in the risk window (stay 0.5–0.9)
- hedging_p<p>.png      two panels: battery margin and energy sold over the stay
Figures are always regenerated from these CSV files (CODING_RULES: no hand-made plots).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from smartev2g.eval.hedging import (
    RISK_WINDOW,
    load_trajectories,
    over_seeds,
    profile,
    risk_window_summary,
)
from smartev2g.eval.plots import plot_hedging
from smartev2g.utils.run_info import RESULTS_DIR, RunInfo, make_run_dir, make_run_id


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--group", action="append", required=True, metavar="LABEL=GLOB")
    p.add_argument("--p-early", type=float, nargs="+", default=[0.1])
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args()

    groups = {}
    for item in args.group:
        label, sep, pattern = item.partition("=")
        runs = sorted(Path(r) for r in glob.glob(pattern) if Path(r).is_dir())
        if not sep or not runs:
            p.error(f"--group {item!r}: expected LABEL=GLOB matching run folders")
        groups[label] = runs
    agents = list(groups)
    out = make_run_dir(make_run_id("hedging"), root=args.results_dir)
    first = next(iter(groups.values()))[0]
    info = RunInfo(
        out,
        "scripts/analyze_hedging.py",
        first / "config_resolved.yaml",
        seeds=[],
        args={"groups": {k: [r.name for r in v] for k, v in groups.items()}, **vars(args)},
    )

    results = {}
    for p_early in args.p_early:
        profs, windows = [], []
        for agent, runs in groups.items():
            traj = load_trajectories(runs, p_early, args.split)
            profs.append(profile(traj).assign(agent=agent))
            windows.append(risk_window_summary(traj).assign(agent=agent))
        prof = pd.concat(profs, ignore_index=True)
        window = pd.concat(windows, ignore_index=True)
        prof.to_csv(out / f"profile_p{p_early:g}.csv", index=False)
        window.to_csv(out / f"risk_window_p{p_early:g}.csv", index=False)
        plot_hedging(prof, agents, p_early, out / f"hedging_p{p_early:g}.png")
        summary = over_seeds(window, by=["agent"]).round(4)
        results[f"p{p_early:g}"] = {
            agent: {f"{c}_{s}": float(v) for (c, s), v in row.items()}
            for agent, row in summary.iterrows()
        }
        print(f"Risk window (stay {RISK_WINDOW[0]}–{RISK_WINDOW[1]}), test p_early {p_early:g}:")
        print(summary.to_string())
    info.finish(results=results)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
