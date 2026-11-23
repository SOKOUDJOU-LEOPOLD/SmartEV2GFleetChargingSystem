"""Tests for scripts/make_figures.py summaries and eval/plots.py figures."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from smartev2g.envs.make_env import REPO_ROOT
from smartev2g.eval import plots

spec = importlib.util.spec_from_file_location("mf", REPO_ROOT / "scripts" / "make_figures.py")
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)


def per_run():
    rows = []
    for seed, (a, b) in enumerate([(-10, -8), (-12, -9), (-11, -12)]):
        for p, shift in ((0.0, 0), (0.2, -5)):
            rows.append({"agent": "A", "seed": seed, "test_p_early": p, "return": a + shift})
            rows.append({"agent": "B", "seed": seed, "test_p_early": p, "return": b + shift})
    return pd.DataFrame(rows)


def test_summary_and_paired_difference():
    df = per_run()
    summary = mf.summarise(df)
    row = summary[(summary["agent"] == "A") & (summary["test_p_early"] == 0.0)].iloc[0]
    assert row["return_mean"] == pytest.approx(-11.0)
    assert row["n_seeds"] == 3
    half = plots.ci95(pd.Series([-10.0, -12.0, -11.0]))
    assert row["return_ci95"] == pytest.approx(half)
    pairs = mf.paired(df, ["A", "B"])
    first = pairs[pairs["test_p_early"] == 0.0].iloc[0]
    assert first["diff_mean"] == pytest.approx(np.mean([2, 3, -1]))
    assert first["wins"] == 2 and first["n_pairs"] == 3


def test_figures_are_written(tmp_path):
    df = per_run()
    summary = mf.summarise(df)
    refs = pd.DataFrame(
        {"policy": ["AFAP-target"] * 2, "test_p_early": [0.0, 0.2], "return": [-15.0, -15.0]}
    )
    plots.plot_grid(summary, ["A", "B"], tmp_path / "grid.png", refs)
    curves = pd.DataFrame(
        {
            "agent": ["A"] * 4,
            "run": ["r"] * 4,
            "timesteps": [1e6, 2e6, 3e6, 4e6],
            "return": [-50, -20, -10, -8],
        }
    )
    plots.plot_learning(curves, ["A"], tmp_path / "learning.png")
    assert (tmp_path / "grid.png").stat().st_size > 10_000
    assert (tmp_path / "learning.png").stat().st_size > 10_000
    with pytest.raises(ValueError):
        plots.plot_grid(summary, [f"x{i}" for i in range(9)], tmp_path / "too_many.png")


def test_every_agent_is_paired_with_the_first_and_holm_corrected():
    df = per_run()
    extra = df[df["agent"] == "B"].assign(agent="C")
    extra["return"] += 1
    pairs = mf.paired(pd.concat([df, extra]), ["A", "B", "C"])
    assert set(pairs["comparison"]) == {"B − A", "C − A"}
    assert len(pairs) == 4  # 2 comparisons × 2 test worlds
    assert (pairs["t_test_p_holm"] >= pairs["t_test_p"]).all()
    assert (pairs["t_test_p_holm"] <= 1).all()


def test_holm_by_hand():
    p = pd.Series([0.01, 0.04, 0.03])
    # sorted 0.01, 0.03, 0.04 -> ×3, ×2, ×1 = 0.03, 0.06, 0.04 -> never decreasing: 0.06
    assert mf.holm(p).tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_car_stats_battery_level_when_leaving():
    evs = pd.DataFrame(
        {
            "soc_final": [0.80, 0.60, 0.75, 0.90],
            "early": [False, True, True, False],
            "below_target": [False, True, True, False],
        }
    )
    row = mf.car_stats(evs)
    assert row["soc_final_all"] == pytest.approx(0.7625)
    assert row["soc_final_early"] == pytest.approx(0.675)
    assert row["pct_early_below_70"] == pytest.approx(50.0)
    assert row["pct_all_below_target"] == pytest.approx(50.0)


def test_many_agents_get_a_legend_under_the_plot(tmp_path):
    df = per_run()
    many = pd.concat([df.assign(agent=f"agent {i}") for i in range(6)])
    plots.plot_grid(mf.summarise(many), [f"agent {i}" for i in range(6)], tmp_path / "g.png")
    assert (tmp_path / "g.png").stat().st_size > 10_000


# --- money-vs-safety curve (scripts/make_tradeoff.py) ---
tspec = importlib.util.spec_from_file_location("mt", REPO_ROOT / "scripts" / "make_tradeoff.py")
mt = importlib.util.module_from_spec(tspec)
tspec.loader.exec_module(mt)


def tradeoff_seeds():
    """Agent B at λ 1 is cheaper AND safer than A at λ 2 (clear gap, small spread)."""
    rows = []
    for seed in range(6):
        noise = 0.1 * seed
        rows += [
            {
                "agent": "A",
                "lambda_dep": 1.0,
                "seed": seed,
                "money": -10 + noise,
                "below70": 30 + noise,
            },
            {
                "agent": "A",
                "lambda_dep": 2.0,
                "seed": seed,
                "money": -20 + noise,
                "below70": 20 + noise,
            },
            {
                "agent": "B",
                "lambda_dep": 1.0,
                "seed": seed,
                "money": -12 + noise,
                "below70": 10 + noise,
            },
            {
                "agent": "B",
                "lambda_dep": 2.0,
                "seed": seed,
                "money": -22 + noise,
                "below70": 5 + noise,
            },
        ]
    return pd.DataFrame(rows)


def test_tradeoff_summary_and_dominance():
    seeds = tradeoff_seeds()
    points = mt.summarise(seeds)
    row = points[(points["agent"] == "B") & (points["lambda_dep"] == 1.0)].iloc[0]
    assert row["money_mean"] == pytest.approx(-12 + 0.25) and row["n_seeds"] == 6
    dom = mt.dominance(seeds, "A", "B")
    assert len(dom) == 4  # every (B point, A point) pair
    both = dom[dom["better_on_both"]]
    # only B at λ 1 vs A at λ 2 is better on both money (+8) and safety (−10)
    assert list(zip(both["lambda_other"], both["lambda_base"], strict=True)) == [(1.0, 2.0)]
    assert (dom["money_p_holm"] >= dom["money_p"]).all()


def test_tradeoff_figure_is_written(tmp_path):
    points = mt.summarise(tradeoff_seeds())
    refs = pd.DataFrame([
        {"policy": "AFAP-target", "lambda_dep": 1.0, "money": -30, "below70": 0.5},
        {"policy": "Optimum-real", "lambda_dep": 1.0, "money": -5, "below70": 0.5},
        {"policy": "Optimum-real", "lambda_dep": 2.0, "money": -6, "below70": 0.5},
    ])  # fmt: skip
    plots.plot_tradeoff(points, ["A", "B"], 0.2, tmp_path / "t.png", refs)
    assert (tmp_path / "t.png").stat().st_size > 0
