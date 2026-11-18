"""Train a PPO agent on EV2Gym from one experiment config + one seed.

What happens (plain language):
1. Start `n_envs` copies of the simulator, each in its own CPU process, so the
   agent experiences many simulated days in parallel. Ref: [R13] (SubprocVecEnv)
2. Rescale observations/rewards to similar sizes (VecNormalize). Ref: [R13]
3. PPO repeatedly: plays `n_steps` in every copy, then improves its policy
   from that experience in small, safe steps. Ref: [R8]
4. While training: log losses (train_progress.csv), every finished day
   (train_episodes.csv), and validation scores + best model (val_curve.csv).
5. At the end: test the best model on the fixed test days
   (test_episodes.csv, test_evs.csv) and write run_info.json.
Output folder: results/<run_id>/. See docs/plan/METRICS_AND_LOGGING.md.
"""

from __future__ import annotations

import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import (
    CSVOutputFormat,
    HumanOutputFormat,
    Logger,
    TensorBoardOutputFormat,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from smartev2g.agents.callbacks import EpisodeMetricsCallback, ValidationCallback
from smartev2g.envs.make_env import make_env, resolve_path
from smartev2g.eval.evaluate import SB3Policy, evaluate_policy, scenario_seeds
from smartev2g.utils.logging import init_wandb
from smartev2g.utils.run_info import RESULTS_DIR, RunInfo, make_run_dir, make_run_id

# EV2Gym draws later training days from seeds 0..999,999 (see configs/eval/scenarios.yaml).
# The first day of env copy i uses seed*SEED_STRIDE + i, which must stay below that.
SEED_STRIDE = 1000
MAX_TRAIN_SEED = 1_000_000


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(resolve_path(path).read_text())


def env_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """make_env arguments from the 'env' section of an experiment config."""
    e = cfg["env"]
    return {
        "config_path": e["config"],
        "state_function": e["state_function"],
        "reward_function": e["reward_function"],
    }


def _env_factory(kwargs: dict[str, Any]):
    def _init():
        # Monitor records each episode's raw return and length. Ref: [R13]
        return Monitor(make_env(**kwargs))

    return _init


def make_training_env(
    cfg: dict[str, Any], seed: int, n_envs: int, restart: int = 0
) -> VecNormalize:
    """`n_envs` simulator copies (one process each) + VecNormalize.

    Copy i starts on day seed*1000 + restart*n_envs + i; later days follow from it.
    `restart` (number of resumes so far) gives a resumed run fresh days instead
    of replaying the ones it started with.
    """
    first_seed = seed * SEED_STRIDE + restart * n_envs
    if seed < 0 or n_envs * (restart + 1) > SEED_STRIDE:
        raise ValueError("training seeds would overlap another seed's days")
    if first_seed + n_envs > MAX_TRAIN_SEED:
        raise ValueError(f"seed {seed} too large: training seeds must stay < {MAX_TRAIN_SEED}")
    factories = [_env_factory(env_kwargs(cfg)) for _ in range(n_envs)]
    venv = SubprocVecEnv(factories) if n_envs > 1 else DummyVecEnv(factories)
    venv.seed(first_seed)
    t = cfg["training"]
    return VecNormalize(
        venv,
        norm_obs=t["normalize_obs"],
        norm_reward=t["normalize_reward"],
        gamma=cfg["ppo"]["gamma"],
    )


def make_logger(run_dir: Path, restart: int = 0, verbose: bool = True) -> Logger:
    """SB3 logger -> train_progress.csv (all losses) + TensorBoard (+ console).

    SB3 overwrites its CSV file, so a resumed run writes train_progress_resume<k>.csv.
    """
    name = "train_progress.csv" if restart == 0 else f"train_progress_resume{restart}.csv"
    formats = [
        CSVOutputFormat(str(run_dir / name)),
        TensorBoardOutputFormat(str(run_dir / "tb")),
    ]
    if verbose:
        formats.append(HumanOutputFormat(sys.stdout))
    return Logger(str(run_dir), formats)


def latest_checkpoint(run_dir: Path) -> tuple[Path, Path] | None:
    """Most recent (model.zip, vecnormalize.pkl) pair in run_dir/checkpoints."""
    ckpts = sorted(
        (run_dir / "checkpoints").glob("ppo_*_steps.zip"),
        key=lambda p: int(re.search(r"_(\d+)_steps", p.name).group(1)),
    )
    if not ckpts:
        return None
    steps = re.search(r"_(\d+)_steps", ckpts[-1].name).group(1)
    return ckpts[-1], run_dir / "checkpoints" / f"ppo_vecnormalize_{steps}_steps.pkl"


def train(
    config_path: str | Path,
    seed: int,
    results_dir: Path = RESULTS_DIR,
    resume_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
    run_test: bool = True,
    verbose: bool = True,
) -> Path:
    """Train one agent; returns its results folder. `overrides` patch cfg['training']."""
    cfg = load_config(config_path)
    cfg["training"].update(overrides or {})
    t = cfg["training"]

    if resume_dir is not None:
        run_dir = Path(resume_dir)
        run_id = run_dir.name
    else:
        run_id = make_run_id(f"{cfg['name']}_s{seed}")
        run_dir = make_run_dir(run_id, root=results_dir)
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    info = RunInfo(
        run_dir,
        "scripts/train.py",
        config_path,
        seeds=[seed],
        args={"overrides": overrides or {}, "resumed": resume_dir is not None},
    )
    info.data["config"] = cfg  # experiment config incl. overrides
    info.data["env_config"] = load_config(cfg["env"]["config"])

    set_random_seed(seed)  # seeds PyTorch, NumPy, random for the agent itself. Ref: [R14]
    # The network is tiny: by default PyTorch would spread each update over all
    # CPU cores, which wastes CPU when several seeds run side by side.
    # Value chosen by scripts/timing_test.py (see ADR-0003).
    if t.get("torch_threads"):
        torch.set_num_threads(t["torch_threads"])
    ckpt = latest_checkpoint(run_dir) if resume_dir is not None else None
    if resume_dir is not None and ckpt is None:
        raise FileNotFoundError(f"No checkpoint to resume from in {run_dir / 'checkpoints'}")
    restart = len(list(run_dir.glob("train_progress*.csv"))) if ckpt is not None else 0
    venv = make_training_env(cfg, seed, t["n_envs"], restart)

    if ckpt is not None:
        venv = VecNormalize.load(str(ckpt[1]), venv.venv)  # restore rescaling statistics
        model = PPO.load(ckpt[0], env=venv, device=t["device"])
        print(f"Resuming from {ckpt[0].name} ({model.num_timesteps} steps)")
    else:
        model = PPO("MlpPolicy", venv, seed=seed, device=t["device"], verbose=0, **cfg["ppo"])

    # wandb starts AFTER the simulator processes, so they do not inherit its threads.
    tags = [cfg["agent"], f"seed{seed}", cfg["env"]["reward_function"]]
    wandb_run = init_wandb(cfg, run_id, run_dir, group=cfg["name"], tags=tags)
    model.set_logger(make_logger(run_dir, restart, verbose))

    callbacks = CallbackList(
        [
            EpisodeMetricsCallback(run_dir / "train_episodes.csv", wandb_run),
            ValidationCallback(
                scenario_seeds("validation", cfg["scenarios"]),
                env_kwargs(cfg),
                t["eval_every_steps"],
                run_dir,
                wandb_run,
            ),
            # Resumable checkpoints (CODING_RULES §7). save_freq counts calls = steps / n_envs.
            CheckpointCallback(
                save_freq=max(t["checkpoint_every_steps"] // t["n_envs"], 1),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="ppo",
                save_vecnormalize=True,
            ),
        ]
    )

    t0 = time.perf_counter()
    start_steps = model.num_timesteps
    model.learn(
        total_timesteps=t["total_timesteps"] - start_steps,
        callback=callbacks,
        reset_num_timesteps=ckpt is None,
        progress_bar=False,
    )
    train_seconds = time.perf_counter() - t0
    model.save(run_dir / "final_model.zip")
    venv.save(str(run_dir / "final_vecnormalize.pkl"))
    venv.close()

    results: dict[str, Any] = {
        "total_timesteps": model.num_timesteps,
        "train_seconds": round(train_seconds, 1),
        "steps_per_second": round((model.num_timesteps - start_steps) / train_seconds, 1),
    }
    if run_test:
        results.update(test_best_model(cfg, run_dir, wandb_run))

    info.finish(results=results)
    if wandb_run is not None:
        wandb_run.summary.update(results)
        wandb_run.save(str(run_dir / "run_info.json"), base_path=str(run_dir))
        wandb_run.finish()
    # Keep only best + final models (disk is limited); checkpoints were for resuming.
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
    return run_dir


def test_best_model(cfg: dict[str, Any], run_dir: Path, wandb_run=None) -> dict[str, Any]:
    """Evaluate the best validation checkpoint (or the final model) on the fixed test days."""
    name = "best" if (run_dir / "best_model.zip").exists() else "final"
    model = PPO.load(run_dir / f"{name}_model.zip", device="cpu")
    vec_norm = None
    norm_path = run_dir / f"{name}_vecnormalize.pkl"
    if norm_path.exists():
        dummy = DummyVecEnv([_env_factory(env_kwargs(cfg))])
        vec_norm = VecNormalize.load(str(norm_path), dummy)
        vec_norm.training = False  # freeze the rescaling statistics during testing
    policy = SB3Policy(model, vec_norm, name=cfg["agent"])
    episodes, cars = evaluate_policy(
        policy, scenario_seeds("test", cfg["scenarios"]), env_kwargs(cfg)
    )
    episodes.insert(1, "checkpoint", name)
    episodes.to_csv(run_dir / "test_episodes.csv", index=False)
    cars.to_csv(run_dir / "test_evs.csv", index=False)
    means = episodes.drop(columns=["policy", "checkpoint", "seed"]).mean(numeric_only=True)
    summary = {f"test/{k}": float(v) for k, v in means.items()}
    summary["test/checkpoint"] = name
    if wandb_run is not None:
        import wandb

        wandb_run.log({"test/episodes": wandb.Table(dataframe=episodes)})
    return summary
