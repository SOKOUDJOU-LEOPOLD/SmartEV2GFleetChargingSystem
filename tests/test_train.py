"""End-to-end smoke test of PPO training: a tiny run must produce every log file.

Checks the logging rules (CODING_RULES §5-6, docs/plan/METRICS_AND_LOGGING.md).
"""

import json

import pandas as pd
import pytest
import yaml

from smartev2g.agents.train import latest_checkpoint, train
from smartev2g.envs.make_env import resolve_path

EXPERIMENT = "configs/experiments/p1_sanity_ppo.yaml"


@pytest.fixture()
def tiny_config(tmp_path, monkeypatch):
    """The real experiment config, shrunk: 2 envs, few steps, 2 val days, 2 test days."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text(
        yaml.safe_dump(
            {
                "validation": {"start": 1_000_000, "count": 2},
                "test": {"start": 2_000_000, "count": 2},
            }
        )
    )
    cfg = yaml.safe_load(resolve_path(EXPERIMENT).read_text())
    cfg["scenarios"] = str(scenarios)
    cfg["training"].update(
        total_timesteps=1792,
        n_envs=2,
        eval_every_steps=896,
        checkpoint_every_steps=896,
        torch_threads=1,
    )
    cfg["ppo"].update(n_steps=448, batch_size=448, n_epochs=1)
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_tiny_training_run_writes_all_outputs(tiny_config, tmp_path):
    run_dir = train(tiny_config, seed=0, results_dir=tmp_path / "results", verbose=False)

    for name in [
        "config_resolved.yaml",
        "run_info.json",
        "train_progress.csv",
        "train_episodes.csv",
        "val_curve.csv",
        "test_episodes.csv",
        "test_evs.csv",
        "best_model.zip",
        "final_model.zip",
    ]:
        assert (run_dir / name).exists(), name
    assert not (run_dir / "checkpoints").exists()  # only best + final are kept

    progress = pd.read_csv(run_dir / "train_progress.csv")
    for col in ["train/policy_gradient_loss", "train/value_loss", "train/entropy_loss"]:
        assert col in progress.columns  # loss curves for the paper

    episodes = pd.read_csv(run_dir / "train_episodes.csv")
    assert len(episodes) == 1792 // 112  # 16 finished simulated days
    assert {"return", "total_profits", "average_user_satisfaction"} <= set(episodes.columns)

    val = pd.read_csv(run_dir / "val_curve.csv")
    assert list(val["timesteps"]) == [896, 1792]

    test = pd.read_csv(run_dir / "test_episodes.csv")
    assert list(test["seed"]) == [2_000_000, 2_000_001]
    assert test["checkpoint"].iloc[0] == "best"

    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["seeds"] == [0]
    assert info["results"]["total_timesteps"] == 1792
    assert "test/return" in info["results"]


def test_latest_checkpoint_picks_highest_step(tmp_path):
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for steps in [896, 10752, 1792]:
        (ckpts / f"ppo_{steps}_steps.zip").touch()
        (ckpts / f"ppo_vecnormalize_{steps}_steps.pkl").touch()
    model, norm = latest_checkpoint(tmp_path)
    assert model.name == "ppo_10752_steps.zip"
    assert norm.name == "ppo_vecnormalize_10752_steps.pkl"


def test_apply_settings_changes_nested_values_and_rejects_typos():
    from smartev2g.agents.train import apply_settings

    cfg = {"ppo": {"gamma": 0.99}, "training": {"n_envs": 16}}
    apply_settings(cfg, {"ppo.gamma": 0.999, "training.n_envs": 4})
    assert cfg == {"ppo": {"gamma": 0.999}, "training": {"n_envs": 4}}
    with pytest.raises(KeyError):
        apply_settings(cfg, {"ppo.gama": 1.0})
