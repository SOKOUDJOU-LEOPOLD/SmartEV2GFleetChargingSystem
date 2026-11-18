"""Tests for baselines, fixed evaluation days and the evaluation loop (Phase 1)."""

import numpy as np

from smartev2g.agents.baselines import AFAPPolicy, RandomPolicy
from smartev2g.envs.make_env import make_env
from smartev2g.eval.evaluate import evaluate_policy, run_episode, scenario_seeds


def test_scenario_seeds_never_overlap_training():
    """Training days come from seeds < 1,000,000 (EV2Gym's own draw); eval days never do."""
    val, test = scenario_seeds("validation"), scenario_seeds("test")
    assert min(val) >= 1_000_000 and min(test) >= 1_000_000
    assert not set(val) & set(test)
    assert len(val) == 10 and len(test) == 50


def test_reused_env_gives_same_day_as_fresh_env():
    """evaluate_policy reuses one env for all days; that must not change results."""
    seed = 1_000_003
    reused = make_env(seed=5)
    run_episode(RandomPolicy(1), reused, 77)  # play another day first
    ep_reused, evs_reused = run_episode(RandomPolicy(1), reused, seed)
    ep_fresh, evs_fresh = run_episode(RandomPolicy(1), make_env(seed=seed), seed)
    assert ep_reused == ep_fresh
    assert evs_reused == evs_fresh


def test_afap_never_discharges_and_satisfies_drivers():
    episodes, cars = evaluate_policy(AFAPPolicy(), [1_000_000, 1_000_001])
    assert (episodes["total_energy_discharged"] == 0).all()
    assert episodes["pct_evs_met_target"].min() > 90
    assert len(cars) == episodes["n_evs"].sum()


def test_random_policy_is_repeatable_and_sells_energy():
    a, _ = evaluate_policy(RandomPolicy(seed=0), [1_000_000])
    b, _ = evaluate_policy(RandomPolicy(seed=0), [1_000_000])
    assert a.equals(b)
    assert a["total_energy_discharged"].iloc[0] > 0  # V2G works


def test_shortfall_matches_definition():
    _, cars = evaluate_policy(RandomPolicy(seed=0), [1_000_000])
    expected = np.maximum(0, cars["soc_target"] - cars["soc_final"])
    np.testing.assert_allclose(cars["shortfall"], expected)
