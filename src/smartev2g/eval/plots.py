"""All paper figures, drawn from the CSV results (never by hand; CODING_RULES §3b).

One visual style for every figure (light, for the paper):
- agents get the validated categorical palette in FIXED order (slot 1 = first agent,
  slot 2 = second, ...), never cycled; at most 8 agents per figure;
- reference policies (AFAP-target, optimum) are grey lines with their name written
  next to them, so they never borrow an agent's colour;
- recessive grid and axes; text in text colours; one measure per panel (no second
  y-axis); a legend, plus names at the line ends when there are at most 4 agents
  (with more, the names would crowd: the legend moves under the plot instead).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from smartev2g.eval.hedging import RISK_WINDOW  # noqa: E402

# Validated categorical palette (light mode), fixed order.
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
TEXT, TEXT_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
REFERENCE_STYLES = ["--", ":", "-."]  # grey reference lines, told apart by dash + label
DIRECT_LABEL_MAX = 4  # more agents than this: legend under the plot, no names at line ends


def ci95(values: pd.Series) -> float:
    """Half-width of the 95 % confidence interval of the mean over seeds (t-distribution)."""
    n = values.count()
    if n < 2:
        return float("nan")
    return float(st.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def _style(ax, title: str, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=TEXT_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def _direct_labels(ax, ends: list[list]) -> None:
    """Names at the right end of each line, pushed apart if they would overlap."""
    ends.sort()
    low, high = ax.get_ylim()
    gap = 0.06 * (high - low)  # minimum vertical distance between two labels
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    overflow = ends[-1][0] - (high - gap / 2) if ends else 0
    if overflow > 0:  # keep every label inside the plot: shift the stack down
        for e in ends:
            e[0] -= overflow
    for y, name, x in ends:
        ax.annotate(name, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=TEXT, annotation_clip=False)  # fmt: skip


def _check_agents(agents: list[str]) -> None:
    if len(agents) > len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} agents per figure (fold or facet)")


def _legend(fig, ax, agents: list[str], loc: str) -> float:
    """Legend inside the plot (few agents) or one row under the figure (many agents).

    Returns the share of the figure height to keep free at the bottom for it."""
    if len(agents) <= DIRECT_LABEL_MAX:
        ax.legend(frameon=False, fontsize=8, loc=loc, labelcolor=TEXT)
        return 0.0
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=TEXT,
               loc="lower center", ncol=min(len(agents), 4))  # fmt: skip
    return 0.06 * ((len(agents) - 1) // 4 + 1) + 0.03


def _save(fig, path: Path, title: str, bottom: float = 0.0) -> None:
    fig.suptitle(title, fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_hedging(prof: pd.DataFrame, agents: list[str], p_early: float, path: Path) -> None:
    """Battery margin and energy sold over the announced stay (eval/hedging.profile)."""
    _check_agents(agents)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    panels = [
        ("margin", "Battery level above target (SoC fraction)"),
        ("sold_kwh", "Energy sold per car and step (kWh)"),
    ]
    for ax, (col, title) in zip(axes, panels, strict=True):
        lo, hi = RISK_WINDOW
        ax.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0)
        ax.text((lo + hi) / 2, 0.02, "early leavers\ndrive off here",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color=TEXT_2)  # fmt: skip
        if col == "margin":
            ax.axhline(0, color=MUTED, lw=1)
        ends = []
        for agent, color in zip(agents, SERIES_COLORS, strict=False):
            s = prof[prof["agent"] == agent].groupby("stay_bin")[col].agg(["mean", "std"])
            x = s.index.to_numpy()
            ax.fill_between(x, s["mean"] - s["std"], s["mean"] + s["std"], color=color,
                            alpha=0.15, lw=0)  # fmt: skip
            ax.plot(x, s["mean"], color=color, lw=2, marker="o", ms=4, label=agent)
            if len(agents) <= DIRECT_LABEL_MAX:
                ends.append([float(s["mean"].iloc[-1]), agent, float(x[-1])])
        _style(ax, title, "Share of the announced stay elapsed (0 = arrival, 1 = announced "
                          "departure)")  # fmt: skip
        ax.set_xlim(0, 1.18)
        _direct_labels(ax, ends)
    bottom = _legend(fig, axes[0], agents, "upper left")
    title = (
        f"How agents manage a car's battery over its stay (test world: {p_early:.0%} early "
        "departures; mean over seeds, band = ±1 std)"
    )
    _save(fig, path, title, bottom)


def plot_grid(summary: pd.DataFrame, agents: list[str], path: Path,
              references: pd.DataFrame | None = None, metric: str = "return",
              ylabel: str = "Score per day (higher is better)") -> None:  # fmt: skip
    """Train/test grid: each agent's test score in every world (mean ± 95 % CI over seeds).

    summary: one row per agent and test_p_early with columns <metric>_mean, <metric>_ci95.
    references: optional rows (policy, test_p_early, <metric>) for grey reference lines.
    """
    _check_agents(agents)
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ends = []
    if references is not None:
        for (name, ref), dash in zip(references.groupby("policy", sort=False), REFERENCE_STYLES,
                                     strict=False):  # fmt: skip
            ref = ref.sort_values("test_p_early")
            x = 100 * ref["test_p_early"].to_numpy()
            ax.plot(x, ref[metric], color=MUTED, lw=1.5, ls=dash)
            ends.append([float(ref[metric].iloc[-1]), name, float(x[-1])])
    for agent, color in zip(agents, SERIES_COLORS, strict=False):
        s = summary[summary["agent"] == agent].sort_values("test_p_early")
        x = 100 * s["test_p_early"].to_numpy()
        mean, half = s[f"{metric}_mean"].to_numpy(), s[f"{metric}_ci95"].to_numpy()
        ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.15, lw=0)
        ax.plot(x, mean, color=color, lw=2, marker="o", ms=5, label=agent)
        if len(agents) <= DIRECT_LABEL_MAX:
            ends.append([float(mean[-1]), agent, float(x[-1])])
    _style(ax, ylabel, "Drivers leaving early without warning in the test world (%)")
    ax.set_xlim(left=-0.5, right=ax.get_xlim()[1] * 1.3)
    _direct_labels(ax, ends)
    bottom = _legend(fig, ax, agents, "lower left")
    _save(fig, path, "Train/test grid: every agent tested in every world "
                     "(mean over seeds, band = 95 % CI)", bottom)  # fmt: skip


def plot_learning(
    curves: pd.DataFrame, agents: list[str], path: Path, zoom_from: float = 2e6
) -> None:
    """Validation score during training (mean ± std over seeds): whole run + zoom.

    curves: rows (agent, run, timesteps, return) from each run's val_curve.csv. The
    first steps (copied agent fixing its mistakes) are far below the rest, so the
    right panel zooms in on training after `zoom_from` steps, where agents differ.
    """
    _check_agents(agents)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    for ax, start, title in (
        (axes[0], 0, "Validation score per day: whole training"),
        (axes[1], zoom_from, f"Zoom: after {zoom_from / 1e6:g}M steps"),
    ):
        ends = []
        for agent, color in zip(agents, SERIES_COLORS, strict=False):
            c = curves[(curves["agent"] == agent) & (curves["timesteps"] >= start)]
            s = c.groupby("timesteps")["return"].agg(["mean", "std"])
            x = s.index.to_numpy() / 1e6
            ax.fill_between(x, s["mean"] - s["std"], s["mean"] + s["std"], color=color,
                            alpha=0.15, lw=0)  # fmt: skip
            ax.plot(x, s["mean"], color=color, lw=2, label=agent)
            if len(agents) <= DIRECT_LABEL_MAX:
                ends.append([float(s["mean"].iloc[-1]), agent, float(x[-1])])
        _style(ax, title, "Simulator steps (millions)")
        left, right = ax.get_xlim()
        ax.set_xlim(left, right + 0.3 * (right - left))
        _direct_labels(ax, ends)
    bottom = _legend(fig, axes[0], agents, "lower right")
    _save(
        fig,
        path,
        "Learning curves (mean over seeds, band = ±1 std). Each agent is validated in its "
        "OWN training world: compare agents on the train/test grid, not here.",
        bottom,
    )


def plot_tradeoff(points: pd.DataFrame, agents: list[str], p_early: float, path: Path,
                  refs: pd.DataFrame | None = None) -> None:  # fmt: skip
    """Money-vs-safety curve: one line per agent through its λ_dep points.

    x = money per day (right = cheaper), y = % of early leavers below 70 % (down = safer),
    so the best corner is bottom-right. Error bars: 95 % CI over seeds. Grey references:
    AFAP-target (one point, it ignores λ) and the optimum planners at each λ."""
    _check_agents(agents)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, agent in enumerate(agents):
        g = points[points["agent"] == agent].sort_values("lambda_dep")
        color = SERIES_COLORS[i]
        ax.errorbar(g["money_mean"], g["below70_mean"], xerr=g["money_ci95"],
                    yerr=g["below70_ci95"], color=color, lw=2, marker="o", ms=7,
                    elinewidth=1, capsize=3, label=agent, zorder=3)  # fmt: skip
        for _, r in g.iterrows():  # λ next to each point: which penalty bought this point
            ax.annotate(f"λ {r['lambda_dep']:g}", (r["money_mean"], r["below70_mean"]),
                        xytext=(8, -12), textcoords="offset points",
                        fontsize=7, color=TEXT_2)  # fmt: skip
    if refs is not None:
        for style, policy in zip(
            REFERENCE_STYLES[:2], ("Optimum-real", "Optimum-announced"), strict=True
        ):
            g = refs[refs["policy"] == policy].sort_values("lambda_dep")
            if len(g):
                ax.plot(g["money"], g["below70"], ls=style, color=MUTED, marker="s", ms=5,
                        lw=1.2, label=f"{policy} (perfect planner)")  # fmt: skip
        afap = refs[refs["policy"] == "AFAP-target"]
        if len(afap):
            ax.plot(afap["money"].mean(), afap["below70"].mean(), ls="none", marker="D", ms=7,
                    color=TEXT_2, label="AFAP-target (rule)")  # fmt: skip
    ax.annotate("better", xy=(0.97, 0.3), xytext=(0.84, 0.45), xycoords="axes fraction",
                fontsize=8, color=TEXT_2,
                arrowprops={"arrowstyle": "->", "color": TEXT_2})  # fmt: skip
    ax.set_ylabel("Early leavers driving off below 70 % (%)", fontsize=8, color=TEXT_2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    _style(ax, f"Test world: {p_early:.0%} of drivers leave early without warning",
           "Money per day (€, higher = cheaper charging)")  # fmt: skip
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT)
    _save(fig, path, "Money vs driver protection across penalty weights λ_dep "
                     "(mean over seeds, bars = 95 % CI)")  # fmt: skip
