"""Tests for the per-driver charging target (ADR-0006) and the AFAP-to-target baseline."""

import numpy as np
import pytest

from smartev2g.agents.baselines import AFAPPolicy, AFAPToTargetPolicy
from smartev2g.envs.make_env import make_env
from smartev2g.eval.evaluate import evaluate_policy

DAYS = [1_000_000, 1_000_001, 1_000_002]
REWARD = {
    "reward_function": "squared_shortfall",
    "reward_kwargs": {"lambda_dep": 50.0, "lambda_over": 0.1},
}


def test_target_changes_only_the_target_not_the_cars():
    """Same day with target 1.0 and 0.8: identical cars (arrival, departure, battery,
    arrival level); only desired_capacity differs."""
    full, eighty = make_env(seed=DAYS[0]), make_env(seed=DAYS[0], charging_target=0.8)
    full.reset(seed=DAYS[0])
    eighty.reset(seed=DAYS[0])
    assert len(full.EVs_profiles) == len(eighty.EVs_profiles) > 0
    for a, b in zip(full.EVs_profiles, eighty.EVs_profiles, strict=True):
        assert (a.time_of_arrival, a.time_of_departure, a.location) == (
            b.time_of_arrival,
            b.time_of_departure,
            b.location,
        )
        assert a.battery_capacity == b.battery_capacity
        assert a.battery_capacity_at_arrival == b.battery_capacity_at_arrival
        assert a.desired_capacity == pytest.approx(a.battery_capacity)
        assert b.desired_capacity == pytest.approx(0.8 * b.battery_capacity)


def test_target_holds_on_every_day_of_a_reused_env():
    """reset() of later days (as in evaluation) must apply the target again."""
    _, cars = evaluate_policy(AFAPPolicy(), DAYS, {"charging_target": 0.8})
    np.testing.assert_allclose(cars["soc_target"], 0.8)


def test_afap_to_target_stops_at_the_target():
    """Cars that can reach 80 % end at ~80 % (not 100 %), none ends below target if it
    had time, and cars arriving above 80 % are not charged at all."""
    kw = {**REWARD, "charging_target": 0.8}
    episodes, cars = evaluate_policy(AFAPToTargetPolicy(), DAYS, kw)
    charged = cars[cars["soc_arrival"] < 0.8]
    above = cars[cars["soc_arrival"] >= 0.8]
    # The last partial step overshoots slightly: EV2Gym's real current/efficiency do not
    # scale exactly with the action fraction (<= ~2.5 % of the battery, ~1 kWh).
    assert (charged["soc_final"] < 0.825).all() and (charged["soc_final"] > 0.79).mean() > 0.95
    np.testing.assert_allclose(above["soc_final"], above["soc_arrival"])
    assert (episodes["total_energy_discharged"] == 0).all()
    afap, _ = evaluate_policy(AFAPPolicy(), DAYS, kw)
    assert (episodes["total_profits"] > afap["total_profits"]).all()  # buys less energy


def test_invalid_target_rejected():
    with pytest.raises(ValueError):
        make_env(seed=0, charging_target=1.5)
