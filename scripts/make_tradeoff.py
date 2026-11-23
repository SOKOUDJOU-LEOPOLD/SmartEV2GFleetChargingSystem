"""Money-vs-safety curve: what does each agent pay per day, and how many early leavers
drive off with less than 70 %, at every penalty weight λ_dep? (Phase 5, CODING_RULES §3b)

Usage:
    python scripts/make_tradeoff.py \
        --agent "PPO-Trusting=results/*_p4-arrival-lam{lam}-trusting_s?" \
        --agent "RiskAware p0.2=results/*_p4-arrival-lam{lam}-riskaware-p0.2_s?" \
        --lambdas 200 500 1000 2000 \
        --optimum "results/*_optimum-test-lamdep{lam}-arrivaltarget" \
        --test-p 0.2 --out reports/p5-tradeoff

"{lam}" in a pattern is replaced by each λ. Each λ is one point of an agent's curve.
Raising λ is the "simple" way to protect drivers better; the question is whether
training with early departures (RiskAware) buys more protection per euro.
Writes to --out:
- tradeoff.csv   one row per agent and λ: money per day and % of early leavers below
                 70 % (mean, 95 % CI, n over seeds) + AFAP-target / optimum references
- dominance.csv  every (RiskAware point, Trusting point) pair: difference in money and
                 in % below 70 %, Welch t-test p for each, Holm-corrected over all tests,
                 and whether the second agent is better on BOTH (significant after Holm)
- tradeoff.png   the curves
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import pandas as pd
from scipy import stats as st

from smartev2g.eval.plots import ci95, plot_tradeoff
from smartev2g.utils.run_info import RunInfo

_spec = importlib.util.spec_from_file_location("mf", Path(__file__).with_name("make_figures.py"))
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

ALPHA = 0.05  # significance level after Holm correction. Ref: [R22]


def per_seed(runs: list[Path], test_p: float) -> pd.DataFrame:
    """Money per day and % of early leavers below 70 %, one row per run (= seed)."""
    rows = []
    for run in runs:
        summary = pd.read_csv(run / "eval" / "test_summary.csv")
        money = summary.loc[summary["test_p_early"] == test_p, "total_profits"]
        evs = pd.read_csv(run / "eval" / f"test_p{test_p:g}_evs.csv")
        rows.append({"seed": mf.seed_of(run), "money": float(money.iloc[0]),
                     "below70": mf.car_stats(evs)["pct_early_below_70"]})  # fmt: skip
    return pd.DataFrame(rows)


def reference_points(pattern: str, lambdas: list[float], test_p: float) -> pd.DataFrame:
    """AFAP-target and optimum (one deterministic run each) at every λ."""
    rows = []
    for lam in lambdas:
        ref_dir = mf.reference_dir(pattern.replace("{lam}", f"{lam:g}"))
        summary = pd.read_csv(ref_dir / "test_summary.csv")
        evs = pd.read_csv(ref_dir / f"test_p{test_p:g}_evs.csv")
        for policy, cars in evs.groupby("policy", sort=False):
            money = summary.loc[(summary["policy"] == policy)
                                & (summary["test_p_early"] == test_p), "total_profits"]  # fmt: skip
            rows.append({"policy": policy, "lambda_dep": lam, "money": float(money.iloc[0]),
                         "below70": mf.car_stats(cars)["pct_early_below_70"]})  # fmt: skip
    return pd.DataFrame(rows)


def summarise(seeds: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (agent, lam), g in seeds.groupby(["agent", "lambda_dep"], sort=False):
        row = {"agent": agent, "lambda_dep": lam, "n_seeds": len(g)}
        for m in ("money", "below70"):
            row[f"{m}_mean"], row[f"{m}_ci95"] = g[m].mean(), ci95(g[m])
        out.append(row)
    return pd.DataFrame(out)


def dominance(seeds: pd.DataFrame, base: str, other: str) -> pd.DataFrame:
    """Is `other` at λ_a better on BOTH money and safety than `base` at λ_b?

    Runs at different λ are separate trainings, so seeds are not paired: Welch t-test
    (unequal variances). Holm correction over every test in the table. Ref: [R23], [R22]"""
    rows = []
    for lam_o, o in seeds[seeds["agent"] == other].groupby("lambda_dep"):
        for lam_b, b in seeds[seeds["agent"] == base].groupby("lambda_dep"):
            row = {"other": other, "lambda_other": lam_o, "base": base, "lambda_base": lam_b}
            for m in ("money", "below70"):
                row[f"{m}_diff"] = o[m].mean() - b[m].mean()  # money: + better; below70: − better
                row[f"{m}_p"] = st.ttest_ind(o[m], b[m], equal_var=False).pvalue
            rows.append(row)
    df = pd.DataFrame(rows)
    holm = mf.holm(pd.concat([df["money_p"], df["below70_p"]], ignore_index=True))
    df["money_p_holm"] = holm.iloc[: len(df)].to_numpy()
    df["below70_p_holm"] = holm.iloc[len(df) :].to_numpy()
    df["better_on_both"] = (
        (df["money_diff"] > 0) & (df["below70_diff"] < 0)
        & (df["money_p_holm"] < ALPHA) & (df["below70_p_holm"] < ALPHA)
    )  # fmt: skip
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--agent", action="append", required=True, metavar="LABEL=GLOB_WITH_{lam}")
    p.add_argument("--lambdas", nargs="+", type=float, required=True)
    p.add_argument("--optimum", help="glob of evaluate_optimum.py folders, with {lam}")
    p.add_argument("--test-p", type=float, default=0.2, help="test world (share leaving early)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seeds, used, first = [], {}, None
    for item in args.agent:
        label, _, pattern = item.partition("=")
        for lam in args.lambdas:
            runs = sorted(Path(r) for r in glob.glob(pattern.replace("{lam}", f"{lam:g}")))
            if not runs:
                raise SystemExit(f"--agent {item!r}: nothing matches at λ {lam:g}")
            first = first or runs[0]
            used[f"{label} λ{lam:g}"] = [r.name for r in runs]
            seeds.append(per_seed(runs, args.test_p).assign(agent=label, lambda_dep=lam))
    seeds = pd.concat(seeds, ignore_index=True)
    agents = list(dict.fromkeys(seeds["agent"]))
    info = RunInfo(args.out, "scripts/make_tradeoff.py", first / "config_resolved.yaml",
                   seeds=[], args={"runs": used, **vars(args)})  # fmt: skip

    points = summarise(seeds)
    refs = reference_points(args.optimum, args.lambdas, args.test_p) if args.optimum else None
    table = points if refs is None else pd.concat(
        [points, refs.rename(columns={"policy": "agent", "money": "money_mean",
                                      "below70": "below70_mean"}).assign(n_seeds=1)],
        ignore_index=True)  # fmt: skip
    table.to_csv(args.out / "tradeoff.csv", index=False)
    results = {"tradeoff": table.round(4).to_dict(orient="records")}
    if len(agents) >= 2:
        dom = dominance(seeds, agents[0], agents[1])
        dom.to_csv(args.out / "dominance.csv", index=False)
        results["dominance"] = dom.round(4).to_dict(orient="records")
        print(dom.round(4).to_string(index=False))
    plot_tradeoff(points, agents, args.test_p, args.out / "tradeoff.png", refs)

    info.finish(results=results)
    if "reports" in args.out.resolve().parts:  # committed to git: leave out the machine's name
        path = args.out / "run_info.json"
        data = json.loads(path.read_text())
        data.get("hardware", {}).pop("host", None)
        path.write_text(json.dumps(data, indent=2))
    print(table.round(2).to_string(index=False))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
