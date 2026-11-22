"""Tests for the imitation warm start (agents/imitation.py)."""

import numpy as np
import pytest
import yaml

from smartev2g.agents import imitation
from smartev2g.agents.train import env_kwargs, load_config, load_trained_policy, train
from smartev2g.envs.make_env import resolve_path

CONFIG = "configs/experiments/p2_imitation_ppo.yaml"


def test_discounted_returns_restart_every_day():
    rewards = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    days = np.array([7, 7, 7, 8, 8])
    out = imitation.discounted_returns(rewards, days, gamma=0.5)
    # day 7: 3, 2 + 0.5·3, 1 + 0.5·3.5 ; day 8: 20, 10 + 0.5·20
    np.testing.assert_allclose(out, [2.75, 3.5, 3.0, 20.0, 20.0])


def test_teacher_data_is_cached_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(imitation, "CACHE_DIR", tmp_path)
    kw = env_kwargs(load_config(CONFIG))
    first = imitation.collect_expert_data(kw, days=2, first_day=900_000, workers=2)
    assert first["obs"].shape == (224, 179) and first["actions"].shape == (224, 20)
    assert set(first["days"]) == {900_000, 900_001}
    assert (np.abs(first["actions"]) <= 1 + 1e-9).all()
    assert len(list(tmp_path.glob("*.npz"))) == 1
    again = imitation.collect_expert_data(kw, days=2, first_day=900_000, workers=2)
    np.testing.assert_array_equal(first["actions"], again["actions"])


def test_warm_start_copies_the_teacher(tmp_path, monkeypatch):
    """End to end: train() clones the teacher, saves the copied agent, then runs PPO."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(imitation, "CACHE_DIR", tmp_path / "cache")
    cfg = yaml.safe_load(resolve_path(CONFIG).read_text())
    cfg["pretrain"].update(days=4, workers=2, epochs=15, batch_size=64, log_std=-1.0)
    cfg["training"].update(total_timesteps=224, n_envs=1, eval_every_steps=10**9)
    cfg["ppo"].update(n_steps=224, batch_size=112, n_epochs=1)
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run = train(path, seed=0, results_dir=tmp_path / "results", verbose=False, run_test=False)

    import pandas as pd

    history = pd.read_csv(run / "pretrain.csv")
    assert len(history) == 15
    assert history["actor_mse"].iloc[-1] < history["actor_mse"].iloc[0] / 2
    assert history["critic_mse"].iloc[-1] < history["critic_mse"].iloc[0]
    assert (run / "pretrained_model.zip").exists()

    # The copied agent acts like the teacher on a day it was taught on.
    policy, name = load_trained_policy(load_config(run / "config_resolved.yaml"), run, "pretrained")
    assert name == "pretrained"
    assert float(np.exp(policy.model.policy.log_std.detach().numpy()).mean()) == pytest.approx(
        np.exp(-1.0), rel=1e-5
    )
    data = imitation.collect_expert_data(env_kwargs(cfg), days=4, first_day=900_000, workers=2)
    obs = policy.vec_normalize.normalize_obs(data["obs"][:112].astype(np.float64))
    actions, _ = policy.model.predict(obs, deterministic=True)
    close = np.mean(np.abs(np.clip(actions, -1, 1) - data["actions"][:112]) < 0.25)
    assert close > 0.8
