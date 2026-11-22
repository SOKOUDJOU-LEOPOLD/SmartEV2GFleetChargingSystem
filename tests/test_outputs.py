"""One tiny run must produce EVERY file and column of docs/plan/METRICS_AND_LOGGING.md.

If this test fails after a change, a paper metric stopped being logged.
"""

import importlib.util
import json
import sys

import pandas as pd
import yaml

from smartev2g.agents.train import train
from smartev2g.envs.make_env import REPO_ROOT, resolve_path

FILES = [
    "config_resolved.yaml",
    "run_info.json",
    "train_progress.csv",
    "train_episodes.csv",
    "val_curve.csv",
    "best_model.zip",
    "best_vecnormalize.pkl",
    "final_model.zip",
    "final_vecnormalize.pkl",
    "test_episodes.csv",
    "test_evs.csv",
    "eval/test_summary.csv",
    "eval/test_p0.5_episodes.csv",
    "eval/test_p0.5_evs.csv",
    "eval/test_p0.5_trajectories.csv.gz",
]
COLUMNS = {
    # every PPO update: losses and training health
    "train_progress.csv": [
        "rollout/ep_rew_mean",
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/explained_variance",
        "train/learning_rate",
        "time/fps",
        "time/total_timesteps",
    ],
    # every training day: domain metrics and each reward term separately
    "train_episodes.csv": [
        "timesteps",
        "day_seed",
        "n_early",
        "return",
        "total_profits",
        "total_energy_discharged",
        "total_transformer_overload",
        "reward_econ",
        "reward_overload_penalty",
        "reward_departure_penalty",
    ],
    "val_curve.csv": ["timesteps", "return", "is_best", "trade_spread"],
    # proposal §11 metrics + trade timing, per test day
    "eval/test_p0.5_episodes.csv": [
        "return",
        "total_profits",
        "total_energy_discharged",
        "total_transformer_overload",
        "average_user_satisfaction",
        "pct_evs_met_target",
        "mean_shortfall",
        "n_early",
        "pct_early_met_target",
        "mean_shortfall_early",
        "reward_departure_penalty",
        "avg_buy_price",
        "avg_sell_price",
        "trade_spread",
        "switches_per_car",
    ],
    # per car: announced vs real departure, target vs final level
    "eval/test_p0.5_evs.csv": [
        "arrival_step",
        "departure_step",
        "announced_departure",
        "early",
        "soc_target",
        "soc_final",
        "shortfall",
    ],
    "eval/test_p0.5_trajectories.csv.gz": [
        "step",
        "soc_before",
        "soc_after",
        "action",
        "buy_price",
        "steps_to_announced",
        "steps_to_real",
        "stay_fraction",
        "margin",
    ],
}


def test_tiny_run_writes_every_file_and_column(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text(
        yaml.safe_dump(
            {
                "validation": {"start": 1_000_000, "count": 1},
                "test": {"start": 2_000_000, "count": 1},
            }
        )
    )
    cfg = yaml.safe_load(resolve_path("configs/experiments/p4_main.yaml").read_text())
    cfg["scenarios"] = str(scenarios)
    cfg["pretrain"] = None
    cfg["training"].update(total_timesteps=448, n_envs=1, eval_every_steps=224)
    cfg["ppo"].update(n_steps=224, batch_size=112, n_epochs=1)
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run = train(path, seed=0, results_dir=tmp_path, verbose=False)

    spec = importlib.util.spec_from_file_location(
        "evaluate_model", REPO_ROOT / "scripts" / "evaluate_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["x", "--run", str(run), "--p-early", "0.5", "--trajectories"])
    module.main()

    missing = [f for f in FILES if not (run / f).exists()]
    assert not missing, f"missing files: {missing}"
    for name, columns in COLUMNS.items():
        df = pd.read_csv(run / name)
        absent = [c for c in columns if c not in df.columns]
        assert not absent, f"{name} lacks {absent}"
        assert len(df) > 0, f"{name} is empty"
    info = json.loads((run / "run_info.json").read_text())
    for key in ("git", "hardware", "packages", "start_time", "end_time", "wall_clock_s", "config"):
        assert key in info, f"run_info.json lacks {key}"
    assert {"commit", "dirty"} <= set(info["git"])
