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
