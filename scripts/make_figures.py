"""Make every results figure and summary table from the saved CSV files (CODING_RULES §3b).

Usage:
    python scripts/make_figures.py \
        --group "PPO-Trusting=results/*_p2-imitation-ppo-6M-val30_s?" \
        --group "PPO-RiskAware=results/*_p2-imitation-ppo-riskaware-p0.1-6M-val30_s?" \
        --optimum "results/*_optimum-test-lamdep500" --out reports/p2-warmstart

Each group = one agent, several seeds (runs evaluated with scripts/evaluate_model.py).
Writes to --out (default results/<run_id>_figures/):
- summary.csv   per agent and test world: mean, std, 95 % CI and n over seeds
- paired.csv    per test world, every agent − the FIRST agent, paired by seed: mean
                difference, its 95 % CI, wins, paired t-test p-value, and the
                Holm-corrected p-value over all rows (many comparisons at once)
- drivers.csv   battery level when cars leave (per agent and test world, mean ± 95 % CI
                over seeds): all cars, early leavers, worst 10 % of early leavers,
                share of early leavers below 70 % (+ AFAP-target / optimum with --optimum)
- grid.png      train/test grid (+ AFAP-target and optimum lines with --optimum)
- learning.png  validation score during training
- hedging_p<p>.png  for every test world with saved trajectories (--trajectories)
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from smartev2g.eval.hedging import load_trajectories, profile
from smartev2g.eval.plots import ci95, plot_grid, plot_hedging, plot_learning
from smartev2g.utils.run_info import RESULTS_DIR, RunInfo, make_run_dir, make_run_id

METRICS = [
    "return",
    "total_profits",
    "reward_departure_penalty",
    "pct_evs_met_target",
    "mean_shortfall",
    "pct_early_met_target",
    "mean_shortfall_early",
    "total_energy_discharged",
]


def seed_of(run: Path) -> int:
    match = re.search(r"_s(\d+)$", run.name)
    if not match:
        raise ValueError(f"cannot read the seed from {run.name}")
    return int(match.group(1))


def load_groups(items: list[str]) -> dict[str, list[Path]]:
    groups = {}
    for item in items:
        label, sep, pattern = item.partition("=")
        runs = sorted(Path(r) for r in glob.glob(pattern) if Path(r).is_dir())
        if not sep or not runs:
            raise SystemExit(f"--group {item!r}: expected LABEL=GLOB matching run folders")
        groups[label] = runs
    return groups


def per_run_results(groups: dict[str, list[Path]]) -> pd.DataFrame:
    """One row per agent, seed and test world (from each run's eval/test_summary.csv)."""
    rows = []
    for agent, runs in groups.items():
        for run in runs:
            path = run / "eval" / "test_summary.csv"
            if not path.exists():
                raise SystemExit(f"{path} missing: run scripts/evaluate_model.py first")
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                row = {"agent": agent, "seed": seed_of(run), "run": run.name}
                row["test_p_early"] = r["test_p_early"]
                row.update({m: r[m] for m in METRICS if m in r})
                rows.append(row)
    return pd.DataFrame(rows)


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (agent, p), g in per_run.groupby(["agent", "test_p_early"], sort=False):
        row = {"agent": agent, "test_p_early": p, "n_seeds": len(g)}
        for m in METRICS:
            if m in g:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std(ddof=1)
                row[f"{m}_ci95"] = ci95(g[m])
        out.append(row)
    return pd.DataFrame(out)


def holm(pvalues: pd.Series) -> pd.Series:
    """Holm correction: when we test many differences at once, a few small p-values
    appear by luck; Holm raises each p-value so the chance of ANY false "significant"
    stays below 5 %. Ref: Holm 1979 [R22]."""
    order = pvalues.sort_values().index
    m, running, out = len(pvalues), 0.0, pd.Series(np.nan, index=pvalues.index)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        out[idx] = running
    return out


def paired(per_run: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    """Every agent − the first agent on the same seed, per test world (shared seeds)."""
    base = agents[0]
    wide = per_run.pivot_table(index=["test_p_early", "seed"], columns="agent", values="return")
    rows = []
    for other in agents[1:]:
        for p, g in wide[[base, other]].dropna().groupby(level="test_p_early"):
            diff = g[other] - g[base]
            test = st.ttest_1samp(diff, 0) if len(diff) > 1 else None
            rows.append(
                {
                    "test_p_early": p,
                    "comparison": f"{other} − {base}",
                    "n_pairs": len(diff),
                    "diff_mean": diff.mean(),
                    "diff_ci95": ci95(diff),
                    "wins": int((diff > 0).sum()),
                    "t_test_p": float(test.pvalue) if test is not None else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["t_test_p_holm"] = holm(out["t_test_p"])
    return out


LOW_LEVEL = 0.7  # "left with less than 70 %": a driver who clearly lost range


def car_stats(evs: pd.DataFrame) -> dict:
    """Battery level when cars leave, from one test_p*_evs.csv (one agent, one seed)."""
    early = evs[evs["early"].astype(bool)]
    row = {"soc_final_all": evs["soc_final"].mean(), "pct_all_below_target":
           100 * evs["below_target"].astype(bool).mean()}  # fmt: skip
    if len(early):
        row.update(
            soc_final_early=early["soc_final"].mean(),
            soc_final_early_p10=early["soc_final"].quantile(0.1),
            pct_early_below_70=100 * (early["soc_final"] < LOW_LEVEL).mean(),
        )
    return row


def drivers(groups: dict[str, list[Path]], ref_dir: Path | None) -> pd.DataFrame:
    """Driver protection per agent and test world: mean and 95 % CI over seeds."""
    rows = []
    for agent, runs in groups.items():
        for run in runs:
            for f in sorted((run / "eval").glob("test_p*_evs.csv")):
                p = float(re.search(r"test_p([\d.]+)_evs", f.name).group(1))
                rows.append({"agent": agent, "test_p_early": p, **car_stats(pd.read_csv(f))})
    if ref_dir is not None:  # reference policies: one deterministic set of test days
        for f in sorted(ref_dir.glob("test_p*_evs.csv")):
            p = float(re.search(r"test_p([\d.]+)_evs", f.name).group(1))
            for policy, evs in pd.read_csv(f).groupby("policy", sort=False):
                rows.append({"agent": policy, "test_p_early": p, **car_stats(evs)})
    df = pd.DataFrame(rows)
    out = []
    for (agent, p), g in df.groupby(["agent", "test_p_early"], sort=False):
        row = {"agent": agent, "test_p_early": p, "n_seeds": len(g)}
        for col in df.columns.drop(["agent", "test_p_early"]):
            if g[col].notna().any():
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_ci95"] = ci95(g[col])
        out.append(row)
    return pd.DataFrame(out)


def reference_dir(pattern: str | None) -> Path | None:
    """Latest scripts/evaluate_optimum.py results folder matching the glob."""
    if not pattern:
        return None
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise SystemExit(f"--optimum {pattern!r} matches nothing")
    return Path(dirs[-1])


def references(ref_dir: Path | None) -> pd.DataFrame | None:
    """AFAP-target and optimum lines for the grid figure."""
    if ref_dir is None:
        return None
    ref = pd.read_csv(ref_dir / "test_summary.csv")
    return ref[["policy", "test_p_early", "return"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--group", action="append", required=True, metavar="LABEL=GLOB")
    p.add_argument("--optimum", help="glob of an evaluate_optimum.py results folder")
    p.add_argument("--out", type=Path, help="output folder (default results/<id>_figures)")
    args = p.parse_args()

    groups = load_groups(args.group)
    agents = list(groups)
    out = args.out or make_run_dir(make_run_id("figures"), root=RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    first = next(iter(groups.values()))[0]
    info = RunInfo(
        out,
        "scripts/make_figures.py",
        first / "config_resolved.yaml",
        seeds=[],
        args={"groups": {k: [r.name for r in v] for k, v in groups.items()}, **vars(args)},
    )

    per_run = per_run_results(groups)
    summary = summarise(per_run)
    summary.to_csv(out / "summary.csv", index=False)
    results = {"summary": summary.round(4).to_dict(orient="records")}
    ref_dir = reference_dir(args.optimum)
    if len(agents) >= 2:
        pairs = paired(per_run, agents)
        pairs.to_csv(out / "paired.csv", index=False)
        results["paired"] = pairs.round(4).to_dict(orient="records")
        print(pairs.drop(columns="n_pairs").round(3).to_string(index=False))
    protection = drivers(groups, ref_dir)
    protection.to_csv(out / "drivers.csv", index=False)
    results["drivers"] = protection.round(4).to_dict(orient="records")
    plot_grid(summary, agents, out / "grid.png", references(ref_dir))

    curves = []
    for agent, runs in groups.items():
        for run in runs:
            v = pd.read_csv(run / "val_curve.csv")[["timesteps", "return"]]
            curves.append(v.assign(agent=agent, run=run.name))
    plot_learning(pd.concat(curves, ignore_index=True), agents, out / "learning.png")

    worlds = sorted(
        {float(m.group(1)) for f in first.glob("eval/test_p*_trajectories.csv.gz")
         if (m := re.search(r"test_p([\d.]+)_traj", f.name))}
    )  # fmt: skip
    for world in worlds:
        prof = pd.concat(
            [profile(load_trajectories(runs, world)).assign(agent=a) for a, runs in groups.items()]
        )
        plot_hedging(prof, agents, world, out / f"hedging_p{world:g}.png")

    info.finish(results=results)
    if "reports" in out.resolve().parts:  # committed to git: leave out the machine's name
        path = out / "run_info.json"
        data = json.loads(path.read_text())
        data.get("hardware", {}).pop("host", None)
        path.write_text(json.dumps(data, indent=2))
    print(summary[["agent", "test_p_early", "n_seeds", "return_mean", "return_ci95"]].round(2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
