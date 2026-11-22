"""Tests for the perfect-knowledge plan (eval/optimum.py)."""

import copy

import numpy as np
import pytest

from smartev2g.agents.baselines import AFAPToTargetPolicy
from smartev2g.envs.make_env import make_env
from smartev2g.eval.evaluate import evaluate_policy
from smartev2g.eval.optimum import (
    PerfectPlanPolicy,
    charge_energy,
    discharge_energy,
    plan_day,
)

DAY = 2_000_000
REWARD = {
    "reward_function": "squared_shortfall",
    "reward_kwargs": {"lambda_dep": 500.0, "lambda_over": 0.1},
    "charging_target": 0.8,
}


def test_battery_formulas_match_ev2gym():
    """The plan's battery maths must be EV2Gym's own (models/ev.py)."""
    env = make_env(seed=DAY, **REWARD)
    env.reset(seed=DAY)
    rng = np.random.default_rng(0)
    for ev0 in env.EVs_profiles:
        cs = env.charging_stations[ev0.location]
        for _ in range(100):
            ev = copy.deepcopy(ev0)
            ev.reset()
            ev.current_capacity = rng.uniform(ev.min_battery_capacity, ev.battery_capacity)
            amps = rng.uniform(-32, 32)
            if amps > 0:
                expected = charge_energy(ev, amps, cs, env.timescale)
            else:
                expected = -discharge_energy(ev, -amps, cs, env.timescale)
            energy, _ = ev.step(amps, cs.voltage, phases=cs.phases, type=cs.charger_type)
            assert energy == pytest.approx(expected, abs=1e-9)


def test_replay_matches_plan_and_beats_afap_target():
    policy = PerfectPlanPolicy("real")
    episodes, _ = evaluate_policy(policy, [DAY, DAY + 1], REWARD)
    afap, _ = evaluate_policy(AFAPToTargetPolicy(), [DAY, DAY + 1], REWARD)
    predicted = np.array([p["return"] for p in policy.predicted])
    # The simulator gives about what the plan expected (EV2Gym rounds energy up
    # by ≤ 0.01 kWh per step, so the replay can be slightly better).
    np.testing.assert_allclose(episodes["return"], predicted, atol=1.0)
    assert (episodes["return"].to_numpy() > afap["return"].to_numpy()).all()
    assert episodes["reward_overload_penalty"].max() < 0.5  # grid respected (almost)


def test_without_early_departures_both_versions_plan_the_same():
    env = make_env(seed=DAY, departures={"p_early": 0.0}, **REWARD)
    env.reset(seed=DAY)
    real, announced = plan_day(env, "real"), plan_day(env, "announced")
    assert real["predicted"]["return"] == pytest.approx(announced["predicted"]["return"])


def test_knowing_real_departures_is_worth_something():
    """In a world with many early leavers, trusting the announced times costs reward."""
    kw = {**REWARD, "departures": {"p_early": 0.5}}
    days = [DAY, DAY + 1]
    real, _ = evaluate_policy(PerfectPlanPolicy("real"), days, kw)
    trusting, cars = evaluate_policy(PerfectPlanPolicy("announced"), days, kw)
    assert real["return"].sum() > trusting["return"].sum()
    early = cars[cars["early"]]
    assert len(early) > 0
    # Trusting plans top early leavers up too late: they leave shorter.
    _, real_cars = evaluate_policy(PerfectPlanPolicy("real"), days, kw)
    assert early["shortfall"].mean() > real_cars[real_cars["early"]]["shortfall"].mean()


def test_invalid_version_rejected():
    env = make_env(seed=DAY, **REWARD)
    env.reset(seed=DAY)
    with pytest.raises(ValueError):
        plan_day(env, "psychic")


def test_trade_metrics_separate_timing_from_jitter():
    """eval/metrics.py TradeTracker: the optimum buys low / sells high with few
    switches; AFAP-target never sells and never switches."""
    days = [DAY, DAY + 1]
    best, _ = evaluate_policy(PerfectPlanPolicy("real"), days, REWARD)
    safe, _ = evaluate_policy(AFAPToTargetPolicy(), days, REWARD)
    assert (best["trade_spread"] > 0.01).all()
    assert (best["switches_per_car"] < 8).all()
    assert (safe["switches_per_car"] == 0).all()
    assert safe["avg_sell_price"].isna().all()
