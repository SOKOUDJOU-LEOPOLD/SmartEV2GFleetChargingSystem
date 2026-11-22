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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from smartev2g.eval.hedging import (  # noqa: E402
    RISK_WINDOW,
    load_trajectories,
    over_seeds,
    profile,
    risk_window_summary,
)
from smartev2g.utils.run_info import RESULTS_DIR, RunInfo, make_run_dir, make_run_id  # noqa: E402

# Chart style (light, for the paper): categorical slots 1–2 of the validated palette,
# recessive grid and axes, text in text colours (never the series colour).
SERIES_COLORS = ["#2a78d6", "#eb6834"]
TEXT, TEXT_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def _direct_labels(ax, ends: list[list]) -> None:
    """Agent names at the right end of each line, pushed apart if they would overlap."""
    ends.sort()
    low, high = ax.get_ylim()
    gap = 0.07 * (high - low)  # minimum vertical distance between two labels
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, agent, x in ends:
        ax.annotate(agent, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=TEXT)  # fmt: skip


def plot(prof: pd.DataFrame, agents: list[str], p_early: float, path: Path) -> None:
    """Two panels sharing the stay axis; mean over seeds with a ±1 std band."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    panels = [
        ("margin", "Battery level above target (SoC fraction)", 1.0),
        ("sold_kwh", "Energy sold per car and step (kWh)", 1.0),
    ]
    for ax, (col, title, scale) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        lo, hi = RISK_WINDOW
        ax.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0)
        ax.text(
            (lo + hi) / 2, 0.02, "early leavers\ndrive off here",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8, color=TEXT_2,
        )  # fmt: skip
        if col == "margin":
            ax.axhline(0, color=MUTED, lw=1)
        ends = []  # (y at the right end, agent): direct labels, nudged apart below
        for agent, color in zip(agents, SERIES_COLORS, strict=False):
            stats = (
                prof[prof["agent"] == agent].groupby("stay_bin")[col].agg(["mean", "std"])
            ) * scale
            x = stats.index.to_numpy()
            ax.fill_between(
                x, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color=color,
                alpha=0.15, lw=0,
            )  # fmt: skip
            ax.plot(x, stats["mean"], color=color, lw=2, marker="o", ms=4, label=agent)
            ends.append([float(stats["mean"].iloc[-1]), agent, float(x[-1])])
        _direct_labels(ax, ends)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("Share of the announced stay elapsed (0 = arrival, 1 = announced departure)",
                      fontsize=8, color=TEXT_2)  # fmt: skip
        ax.set_xlim(0, 1.18)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=TEXT_2, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT)
    fig.suptitle(
        f"How agents manage a car's battery over its stay (test world: {p_early:.0%} early "
        "departures; mean over seeds, band = ±1 std)",
        fontsize=10, color=TEXT, x=0.01, ha="left",
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


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
        plot(prof, agents, p_early, out / f"hedging_p{p_early:g}.png")
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
