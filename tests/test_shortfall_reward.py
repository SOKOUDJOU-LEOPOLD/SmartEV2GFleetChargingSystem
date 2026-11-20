"""Tests for our squared-shortfall reward (Phase 2, docs/plan/phase-2-*.md)."""

from types import SimpleNamespace

import numpy as np
import pytest

from smartev2g.agents.baselines import AFAPPolicy, RandomPolicy
from smartev2g.envs.make_env import get_reward_function, make_env
from smartev2g.eval.evaluate import evaluate_policy
from smartev2g.rewards.shortfall_reward import (
    make_squared_shortfall,
    soc_shortfall,
    transformer_overload_kw,
)

LAMBDAS = {"lambda_dep": 10.0, "lambda_over": 0.5}


def fake_ev(target_soc: float, soc: float, battery_kwh: float = 50.0):
    return SimpleNamespace(
        battery_capacity=battery_kwh,
        desired_capacity=target_soc * battery_kwh,
        current_capacity=soc * battery_kwh,
    )


def fake_transformer(power_kw: float, limit_kw: float = 100.0):
    return SimpleNamespace(
        current_power=power_kw, max_power=np.full(5, limit_kw), min_power=np.full(5, -limit_kw)
    )


def fake_env(departing=(), transformers=(), step_just_finished=2):
    return SimpleNamespace(
        current_step=step_just_finished + 1,
        departing_evs=list(departing),
        transformers=list(transformers),
    )


def test_hand_example_from_plan():
    """Target 0.8, leaves at 0.5 -> shortfall 0.3 -> penalty 0.3² = 0.09 (× λ_dep)."""
    reward = make_squared_shortfall(lambda_dep=10.0, lambda_over=0.0)
    env = fake_env(departing=[fake_ev(0.8, 0.5)])
    assert reward(env, 1.5, []) == pytest.approx(1.5 - 10.0 * 0.09)
    assert env.reward_terms["departure_penalty"] == pytest.approx(0.9)


def test_no_penalty_when_target_reached_or_exceeded():
    assert soc_shortfall(fake_ev(0.8, 0.8)) == 0.0
    assert soc_shortfall(fake_ev(0.8, 0.95)) == 0.0


def test_squared_penalty_hurts_big_misses_more():
    """Two cars 0.1 short cost less than one car 0.2 short (0.02 < 0.04)."""
    reward = make_squared_shortfall(lambda_dep=1.0, lambda_over=0.0)
    two_small = reward(fake_env([fake_ev(1, 0.9), fake_ev(1, 0.9)]), 0.0, [])
    one_big = reward(fake_env([fake_ev(1, 0.8)]), 0.0, [])
    assert two_small == pytest.approx(-0.02)
    assert one_big == pytest.approx(-0.04)


def test_overload_counts_both_directions():
    """Rating ±100 kW: +120 kW and −120 kW are both 20 kW over (EV2Gym's own helper says 220)."""
    assert transformer_overload_kw(fake_transformer(120), step=2) == pytest.approx(20)
    assert transformer_overload_kw(fake_transformer(-120), step=2) == pytest.approx(20)
    assert transformer_overload_kw(fake_transformer(80), step=2) == 0.0
    reward = make_squared_shortfall(lambda_dep=0.0, lambda_over=0.5)
    assert reward(fake_env(transformers=[fake_transformer(110)]), 0.0, []) == pytest.approx(-50)


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        make_squared_shortfall(lambda_dep=-1, lambda_over=0)
    with pytest.raises(ValueError):  # EV2Gym's reward takes no settings
        get_reward_function("profit_maximization", {"lambda_dep": 1})
    with pytest.raises(KeyError):
        get_reward_function("no_such_reward")


def test_terms_add_up_on_a_real_episode():
    """On a real day: every step's reward = econ − penalties, and the day's sums match
    EV2Gym's own profit statistic and the episode return."""
    env = make_env(seed=1_000_000, reward_function="squared_shortfall", reward_kwargs=LAMBDAS)
    env.reset(seed=1_000_000)
    rng = np.random.default_rng(0)
    total, done, info = 0.0, False, {}
    while not done:
        _, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 20))
        t = env.reward_terms
        assert reward == pytest.approx(t["econ"] - t["overload_penalty"] - t["departure_penalty"])
        total += reward
        done = terminated or truncated
    ep = env.reward_terms_episode
    assert ep["econ"] - ep["overload_penalty"] - ep["departure_penalty"] == pytest.approx(total)
    assert ep["econ"] == pytest.approx(info["total_profits"])


def test_departure_penalty_matches_per_car_shortfalls():
    """The day's departure penalty = λ_dep × Σ shortfall² over cars that left during the day."""
    kwargs = {"reward_function": "squared_shortfall", "reward_kwargs": LAMBDAS}
    episodes, cars = evaluate_policy(RandomPolicy(seed=0), [1_000_000], kwargs)
    left = cars[cars["departure_step"] < 112]  # cars still plugged in at the end never "leave"
    expected = LAMBDAS["lambda_dep"] * (left["shortfall"] ** 2).sum()
    assert episodes["reward_departure_penalty"].iloc[0] == pytest.approx(expected)
    assert expected > 0  # random drains cars


def test_afap_gets_almost_no_departure_penalty():
    kwargs = {"reward_function": "squared_shortfall", "reward_kwargs": LAMBDAS}
    episodes, _ = evaluate_policy(AFAPPolicy(), [1_000_000, 1_000_001], kwargs)
    assert (episodes["reward_departure_penalty"] < 0.01).all()
