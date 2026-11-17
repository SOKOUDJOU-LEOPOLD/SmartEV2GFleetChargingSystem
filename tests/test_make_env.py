"""Tests for the standard environment factory (Phase 1)."""

import numpy as np
import pytest

from smartev2g.envs.make_env import make_env, observation_names

N_PORTS = 20  # our site: 20 chargers x 1 port
EPISODE_STEPS = 112  # 112 steps x 15 min = 28 h


@pytest.fixture(scope="module")
def env():
    return make_env(seed=0)


def run_episode(seed: int, action_seed: int = 123) -> tuple[np.ndarray, float]:
    """Play one full episode with fixed random actions; return all observations + return."""
    e = make_env(seed=seed)
    obs, _ = e.reset(seed=seed)
    rng = np.random.default_rng(action_seed)  # private RNG: same actions every run
    observations, total = [obs], 0.0
    done = False
    while not done:
        action = rng.uniform(-1, 1, size=e.action_space.shape)
        obs, reward, terminated, truncated, _ = e.step(action)
        observations.append(obs)
        total += reward
        done = terminated or truncated
    return np.array(observations), total


def test_spaces(env):
    """20 ports -> 20 actions; observation = 2 + 20 prices + 2 per port."""
    assert env.number_of_ports == N_PORTS
    assert env.action_space.shape == (N_PORTS,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2 + 20 + 2 * N_PORTS,)
    assert len(observation_names(env)) == obs.shape[0]


def test_episode_length():
    obs_seq, _ = run_episode(seed=1)
    assert len(obs_seq) == EPISODE_STEPS + 1  # reset obs + one per step


def test_same_seed_same_episode():
    """Reproducibility: same seed + same actions => identical episode. Ref: [R14]"""
    a, ra = run_episode(seed=7)
    b, rb = run_episode(seed=7)
    np.testing.assert_array_equal(a, b)
    assert ra == rb


def test_different_seed_different_episode():
    a, _ = run_episode(seed=7)
    b, _ = run_episode(seed=8)
    assert not np.array_equal(a, b)
