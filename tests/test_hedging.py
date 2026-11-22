"""Tests for per-step trajectories (eval/metrics.py) and the hedging analysis (eval/hedging.py)."""

import numpy as np
import pandas as pd
import pytest

from smartev2g.agents.baselines import AFAPToTargetPolicy, RandomPolicy
from smartev2g.eval import hedging
from smartev2g.eval.evaluate import evaluate_policy
from smartev2g.eval.metrics import TrajectoryRecorder

DAY = 2_000_000
KW = {
    "reward_function": "squared_shortfall",
    "reward_kwargs": {"lambda_dep": 500.0, "lambda_over": 0.1},
    "charging_target": 0.8,
    "departures": {"p_early": 0.5},
}


def test_trajectories_agree_with_the_car_table():
    recorder = TrajectoryRecorder()
    episodes, cars = evaluate_policy(RandomPolicy(seed=0), [DAY], KW, recorder)
    traj = pd.DataFrame(recorder.rows)
    # One row per car per step it was plugged in (arrival .. real departure, inclusive).
    steps = (cars["departure_step"] - cars["arrival_step"] + 1).sum()
    assert len(traj) == steps
    # Energy per car adds up from arrival level to final level.
    for (_, arrival), g in traj.groupby(["charger", "arrival_step"]):
        car = cars[(cars["charger"] == g["charger"].iloc[0]) & (cars["arrival_step"] == arrival)]
        soc = g["soc_before"].iloc[0] + g["energy_kwh"].sum() / g["battery_kwh"].iloc[0]
        assert soc == pytest.approx(car["soc_final"].iloc[0], abs=1e-6)
    # Early leavers: the agent's clock (announced) runs past their real departure.
    early = traj[traj["early"]]
    assert len(early) and (early["real_departure"] < early["announced_departure"]).all()
    assert (traj.groupby(["charger", "arrival_step"])["stay_fraction"].min() == 0).all()
    assert (early["stay_fraction"] < 1).all()  # they never reach their announced time


def test_risk_window_and_profile():
    """AFAP-target fills cars at once: in the risk window they sit at the target."""
    recorder = TrajectoryRecorder()
    evaluate_policy(AFAPToTargetPolicy(), [DAY, DAY + 1], KW, recorder)
    traj = pd.DataFrame(recorder.rows).assign(run="afap")
    window = hedging.risk_window_summary(traj)
    assert window["sold_kwh"].iloc[0] == 0  # never sells
    assert window["pct_at_target"].iloc[0] > 80
    prof = hedging.profile(traj)
    assert np.allclose(sorted(prof["stay_bin"].unique()), np.arange(0.05, 1.0, 0.1))
    # Margin grows over the stay (cars are being charged up to the target).
    assert prof["margin"].iloc[-1] > prof["margin"].iloc[0]


def test_over_seeds_mean_and_std():
    per_run = pd.DataFrame({"run": ["a", "b"], "agent": ["x", "x"], "margin": [0.1, 0.3]})
    out = hedging.over_seeds(per_run, by=["agent"])
    assert out.loc["x", ("margin", "mean")] == pytest.approx(0.2)
    assert out.loc["x", ("margin", "std")] == pytest.approx(np.std([0.1, 0.3], ddof=1))
