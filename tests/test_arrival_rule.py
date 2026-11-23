"""Tests for the arrival charge rule (ADR-0008): arrival = target − the driver's energy."""

import numpy as np
import pytest

from smartev2g.agents.baselines import AFAPToTargetPolicy
from smartev2g.envs.make_env import make_env
from smartev2g.eval.evaluate import evaluate_policy

DAYS = [1_000_000, 1_000_001, 1_000_002]


def profiles(**kwargs):
    env = make_env(seed=DAYS[0], charging_target=0.8, **kwargs)
    env.reset(seed=DAYS[0])
    return env.EVs_profiles


def test_each_driver_keeps_the_energy_of_the_data():
    """Same cars; with "target" every driver needs exactly the kWh EV2Gym drew from the
    data (battery − EV2Gym's arrival), counted up to 80 % instead of 100 %."""
    old, new = profiles(), profiles(arrival_rule="target")
    assert len(old) == len(new) > 0
    for a, b in zip(old, new, strict=True):
        assert (a.time_of_arrival, a.time_of_departure, a.location, a.battery_capacity) == (
            b.time_of_arrival,
            b.time_of_departure,
            b.location,
            b.battery_capacity,
        )
        data_energy = a.battery_capacity - a.battery_capacity_at_arrival
        shifted = 0.8 * a.battery_capacity - data_energy
        if shifted >= a.min_battery_capacity:
            assert b.desired_capacity - b.battery_capacity_at_arrival == pytest.approx(data_energy)
        else:  # floor: EV2Gym's minimum battery level
            assert b.battery_capacity_at_arrival == pytest.approx(a.min_battery_capacity)
        assert b.current_capacity == b.battery_capacity_at_arrival


def test_default_rule_is_unchanged():
    old, same = profiles(), profiles(arrival_rule="ev2gym")
    for a, b in zip(old, same, strict=True):
        assert a.battery_capacity_at_arrival == b.battery_capacity_at_arrival


def test_cars_really_arrive_lower_in_the_simulation():
    """Arriving cars are copies of the profiles: their logged arrival level must follow
    the new rule on every day of a reused env (no car above target any more)."""
    kw = {"charging_target": 0.8}
    _, old = evaluate_policy(AFAPToTargetPolicy(), DAYS, kw)
    _, new = evaluate_policy(AFAPToTargetPolicy(), DAYS, {**kw, "arrival_rule": "target"})
    assert len(old) == len(new)
    assert (new["soc_arrival"] < 0.8).all()
    shift = (old["soc_arrival"] - new["soc_arrival"]).to_numpy()
    floored = new["soc_arrival"].to_numpy() * new["battery_kwh"].to_numpy() <= 5 + 1e-9
    np.testing.assert_allclose(shift[~floored], 0.2, atol=1e-9)


def test_target_rule_needs_a_target():
    with pytest.raises(ValueError):
        make_env(seed=0, arrival_rule="target")
    with pytest.raises(ValueError):
        make_env(seed=0, charging_target=0.8, arrival_rule="nonsense")
