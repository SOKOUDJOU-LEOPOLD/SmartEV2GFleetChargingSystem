"""Tests for the central agent with per-charger credit (ADR-0005, agents/credit_ppo.py)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch as th
import yaml
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from smartev2g.agents.credit_ppo import CreditActorCriticPolicy, CreditRolloutBuffer
from smartev2g.agents.train import train
from smartev2g.envs.make_env import resolve_path

N_STEPS, N_ENVS, N_CHARGERS = 7, 3, 4
OBS = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float64)
ACT = spaces.Box(-1.0, 1.0, shape=(N_CHARGERS,), dtype=np.float64)


def fill(buffer, rewards, values, starts):
    for t in range(N_STEPS):
        buffer.add(
            np.zeros((N_ENVS, 5)),
            np.zeros((N_ENVS, N_CHARGERS)),
            rewards[t],
            starts[t],
            th.as_tensor(values[t]),
            th.zeros(values[t].shape),
        )


def test_per_charger_advantages_sum_to_standard_ppo_advantage():
    """GAE is linear: Σ_i GAE(r_i, V_i) = GAE(Σ r_i, Σ V_i), i.e. the per-charger
    advantages add up to what normal PPO computes for the site reward."""
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=(N_STEPS, N_ENVS, N_CHARGERS)).astype(np.float32)
    values = rng.normal(size=(N_STEPS, N_ENVS, N_CHARGERS)).astype(np.float32)
    starts = (rng.random((N_STEPS, N_ENVS)) < 0.3).astype(np.float32)  # some episode ends
    last_values = rng.normal(size=(N_ENVS, N_CHARGERS)).astype(np.float32)
    dones = np.array([True, False, False])
    kw = {"gamma": 0.99, "gae_lambda": 0.95, "n_envs": N_ENVS}

    credit = CreditRolloutBuffer(N_STEPS, OBS, ACT, reward_dim=N_CHARGERS, **kw)
    fill(credit, rewards, values, starts)
    credit.compute_returns_and_advantage(th.as_tensor(last_values), dones)

    standard = RolloutBuffer(N_STEPS, OBS, ACT, **kw)
    fill(standard, rewards.sum(-1), values.sum(-1), starts)
    standard.compute_returns_and_advantage(th.as_tensor(last_values.sum(-1)), dones)

    np.testing.assert_allclose(credit.advantages.sum(-1), standard.advantages, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(credit.returns.sum(-1), standard.returns, rtol=1e-5, atol=1e-5)


def test_samples_keep_one_column_per_charger():
    rng = np.random.default_rng(1)
    buf = CreditRolloutBuffer(
        N_STEPS, OBS, ACT, reward_dim=N_CHARGERS, gamma=0.99, gae_lambda=0.95, n_envs=N_ENVS
    )
    fill(
        buf,
        rng.normal(size=(N_STEPS, N_ENVS, N_CHARGERS)),
        rng.normal(size=(N_STEPS, N_ENVS, N_CHARGERS)),
        np.zeros((N_STEPS, N_ENVS)),
    )
    buf.compute_returns_and_advantage(th.zeros(N_ENVS, N_CHARGERS), np.zeros(N_ENVS, bool))
    batch = next(buf.get(batch_size=6))
    for t in [batch.old_values, batch.old_log_prob, batch.advantages, batch.returns]:
        assert t.shape == (6, N_CHARGERS)


def test_policy_has_one_value_per_charger_and_exact_log_prob_split():
    policy = CreditActorCriticPolicy(OBS, ACT, lambda _: 3e-4)
    obs = th.randn(8, 5)
    actions, values, log_prob = policy(obs)
    assert actions.shape == (8, N_CHARGERS)
    assert values.shape == (8, N_CHARGERS) and log_prob.shape == (8, N_CHARGERS)
    # Diagonal Gaussian: joint log-prob = sum of per-charger log-probs (no approximation).
    joint = policy.get_distribution(obs).log_prob(actions)
    th.testing.assert_close(log_prob.sum(-1), joint)
    # The optimizer trains the new 4-output critic head.
    params = {id(p) for g in policy.optimizer.param_groups for p in g["params"]}
    assert all(id(p) in params for p in policy.value_net.parameters())


def test_tiny_central_credit_run(tmp_path, monkeypatch):
    """End to end with 2 simulators (one reward per charger from each): trains,
    validates, saves, reloads the best model with CreditPPO and tests it with the
    normal site evaluation. (2, not 1: with 1 simulator shape bugs can hide by
    broadcasting.)"""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text(
        yaml.safe_dump(
            {
                "validation": {"start": 1_000_000, "count": 1},
                "test": {"start": 2_000_000, "count": 2},
            }
        )
    )
    cfg = yaml.safe_load(resolve_path("configs/experiments/p2_central_credit_ppo.yaml").read_text())
    cfg["scenarios"] = str(scenarios)
    cfg["training"].update(
        total_timesteps=896, n_envs=2, eval_every_steps=448, checkpoint_every_steps=448
    )
    cfg["ppo"].update(n_steps=224, batch_size=112, n_epochs=2)
    cfg["ppo"]["policy_kwargs"]["net_arch"] = [16, 16]
    path = tmp_path / "credit.yaml"
    path.write_text(yaml.safe_dump(cfg))

    run_dir = train(path, seed=0, results_dir=tmp_path / "results", verbose=False)
    info = json.loads((run_dir / "run_info.json").read_text())["results"]
    assert info["agent_design"] == "central_credit"
    assert info["total_timesteps"] == 896
    progress = pd.read_csv(run_dir / "train_progress.csv")
    # SB3 writes its first log row before the first update (no losses yet), as in PPO.
    losses = progress["train/value_loss"].dropna()
    assert len(losses) >= 1 and np.isfinite(losses).all()
    episodes = pd.read_csv(run_dir / "train_episodes.csv")
    assert len(episodes) == 8 and "reward_overload_penalty" in episodes.columns
    test = pd.read_csv(run_dir / "test_episodes.csv")
    assert list(test["seed"]) == [2_000_000, 2_000_001]


def test_central_credit_rejects_reward_normalization(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    cfg = yaml.safe_load(resolve_path("configs/experiments/p2_central_credit_ppo.yaml").read_text())
    cfg["training"].update(normalize_reward=True, n_envs=1)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="normalize_reward"):
        train(path, seed=0, results_dir=tmp_path / "results", verbose=False, run_test=False)
