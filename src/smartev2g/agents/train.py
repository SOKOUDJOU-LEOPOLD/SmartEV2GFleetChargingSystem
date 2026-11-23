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

import pandas as pd
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
from smartev2g.agents.credit_ppo import CreditPPO
from smartev2g.agents.imitation import collect_expert_data, pretrain_policy
from smartev2g.envs.charger_env import (
    ChargerVecEnv,
    SiteChargerEnv,
    SiteDummyVecEnv,
    VecNormalizeObsOnly,
)
from smartev2g.envs.make_env import make_env, resolve_path
from smartev2g.envs.wrappers import EpisodeInfo
from smartev2g.eval.evaluate import evaluate_policy, make_sb3_policy, scenario_seeds
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
        "reward_kwargs": e.get("reward_kwargs"),
        "charging_target": e.get("charging_target"),  # e.g. 0.8 (ADR-0006); None = 100 %
        "departures": e.get("departures"),  # hidden early departures; None = all on time
    }


# Agent designs (see eval/evaluate.py POLICY_CLASSES):
#   site           one network, whole site, one site reward (Phase 1)
#   shared_charger one small network shared by all chargers (ADR-0004)
#   central_credit one network, whole site, per-charger credit (ADR-0005)
VECTOR_REWARD_DESIGNS = {"shared_charger", "central_credit"}  # one reward per charger


def agent_design(cfg: dict[str, Any]) -> str:
    design = cfg.get("agent_design", "site")
    if design not in {"site", "shared_charger", "central_credit"}:
        raise ValueError(f"Unknown agent_design {design!r}")
    return design


def algorithm(design: str):
    """SB3 algorithm class: PPO, or our CreditPPO for per-charger credit (ADR-0005)."""
    return CreditPPO if design == "central_credit" else PPO


def _in_process_vec_env(design: str):
    """In-process vec env; vector-reward designs need room for one reward per charger."""
    return SiteDummyVecEnv if design in VECTOR_REWARD_DESIGNS else DummyVecEnv


def n_chargers(cfg: dict[str, Any]) -> int:
    env_cfg = load_config(cfg["env"]["config"])
    return env_cfg["number_of_charging_stations"] * env_cfg["number_of_ports_per_cs"]


def step_scale(cfg: dict[str, Any]) -> int:
    """SB3 steps per simulator step: 1 for 'site', n_chargers for 'shared_charger'
    (SB3 sees every charger as its own environment). Configs and logs always count
    SIMULATOR steps, so both designs are compared on the same simulation budget."""
    return n_chargers(cfg) if agent_design(cfg) == "shared_charger" else 1


def _env_factory(kwargs: dict[str, Any], design: str = "site"):
    def _init():
        env = EpisodeInfo(make_env(**kwargs))
        if design in VECTOR_REWARD_DESIGNS:
            # One reward per charger; records the site's episode return itself.
            return SiteChargerEnv(env, per_charger_obs=design == "shared_charger")
        # Monitor records each episode's raw return and length. Ref: [R13]
        return Monitor(env)

    return _init


def _wrap_sites(venv, design: str):
    """For the shared per-charger policy, show every charger as its own env (ADR-0004)."""
    return ChargerVecEnv(venv) if design == "shared_charger" else venv


def first_day_seed(seed: int, n_envs: int, restart: int = 0) -> int:
    """Seed of the first simulated day of env copy 0 (copy i uses this + i).

    = seed*1000 + restart*n_envs, so different training seeds never share days.
    `restart` (number of resumes so far) gives a resumed run fresh days instead
    of replaying the ones it started with.
    """
    first_seed = seed * SEED_STRIDE + restart * n_envs
    if seed < 0 or n_envs * (restart + 1) > SEED_STRIDE:
        raise ValueError("training seeds would overlap another seed's days")
    if first_seed + n_envs > MAX_TRAIN_SEED:
        raise ValueError(f"seed {seed} too large: training seeds must stay < {MAX_TRAIN_SEED}")
    return first_seed


def make_training_env(
    cfg: dict[str, Any], seed: int, n_envs: int, restart: int = 0
) -> VecNormalize:
    """`n_envs` simulator copies (one process each) + VecNormalize.

    Copy i starts on day first_day_seed(...) + i; later days follow from it.
    """
    first_seed = first_day_seed(seed, n_envs, restart)
    design = agent_design(cfg)
    factories = [_env_factory(env_kwargs(cfg), design) for _ in range(n_envs)]
    in_process = _in_process_vec_env(design)
    venv = SubprocVecEnv(factories) if n_envs > 1 else in_process(factories)
    venv.seed(first_seed)
    venv = _wrap_sites(venv, design)
    t = cfg["training"]
    if design == "central_credit":
        if t["normalize_reward"]:
            raise ValueError("central_credit: set normalize_reward: false and use reward_scale")
        return VecNormalizeObsOnly(venv, norm_obs=t["normalize_obs"], gamma=cfg["ppo"]["gamma"])
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
    """Most recent COMPLETE (model.zip, vecnormalize.pkl) pair in run_dir/checkpoints.

    A job can stop between saving the model and saving its rescaling statistics;
    such a half-saved checkpoint is skipped and the previous complete one is used.
    """
    ckpts = sorted(
        (run_dir / "checkpoints").glob("ppo_*_steps.zip"),
        key=lambda p: int(re.search(r"_(\d+)_steps", p.name).group(1)),
    )
    for model in reversed(ckpts):
        steps = re.search(r"_(\d+)_steps", model.name).group(1)
        norm = run_dir / "checkpoints" / f"ppo_vecnormalize_{steps}_steps.pkl"
        if norm.is_file():
            return model, norm
    return None


def load_run_config(config_path: str | Path | None, resume_dir: Path | None) -> dict[str, Any]:
    """Config for a run. A resumed run reloads ITS OWN saved config (config_resolved.yaml),
    so earlier --set changes are kept; the experiment YAML is only used for new runs."""
    if resume_dir is not None:
        return load_config(Path(resume_dir).resolve() / "config_resolved.yaml")
    if config_path is None:
        raise ValueError("config_path is required for a new run")
    return load_config(config_path)


def apply_settings(cfg: dict[str, Any], settings: dict[str, Any]) -> None:
    """Change config values by dotted name, e.g. {"ppo.gamma": 0.999}. Unknown keys fail."""
    for dotted, value in settings.items():
        *parents, leaf = dotted.split(".")
        node = cfg
        for key in parents:
            node = node[key]
        if leaf not in node:
            raise KeyError(f"Unknown config key: {dotted}")
        node[leaf] = value


def train(
    config_path: str | Path | None,
    seed: int,
    results_dir: Path = RESULTS_DIR,
    resume_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    variant: str | None = None,
    run_test: bool = True,
    verbose: bool = True,
) -> Path:
    """Train one agent; returns its results folder.

    Args:
        config_path: experiment YAML (ignored when resuming: the run's saved config is used).
        overrides: patch cfg['training'] (e.g. {"total_timesteps": 1000}).
        settings: patch any config value by dotted name (e.g. {"ppo.gamma": 0.999}).
        variant: short label added to the run name and wandb tags (e.g. "gamma0.999").
    """
    cfg = load_run_config(config_path, resume_dir)
    cfg["training"].update(overrides or {})
    apply_settings(cfg, settings or {})
    if variant:
        cfg["variant"] = variant
    t = cfg["training"]
    design, scale = agent_design(cfg), step_scale(cfg)

    if resume_dir is not None:
        run_dir = Path(resume_dir)
        run_id = run_dir.name
        if cfg.get("seed") != seed:
            raise ValueError(f"{run_id} was started with seed {cfg.get('seed')}, not {seed}")
    else:
        label = f"{cfg['name']}-{variant}" if variant else cfg["name"]
        run_id = make_run_id(f"{label}_s{seed}")
        run_dir = make_run_dir(run_id, root=results_dir)
        cfg["seed"] = seed  # saved now (run_info.json only exists once a run finishes)
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    info = RunInfo(
        run_dir,
        "scripts/train.py",
        # A resumed run records the config it actually uses: its own saved copy.
        run_dir / "config_resolved.yaml" if resume_dir is not None else config_path,
        seeds=[seed],
        args={
            "overrides": overrides or {},
            "settings": settings or {},
            "variant": variant,
            "resumed": resume_dir is not None,
        },
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
    # Imitation warm start (agents/imitation.py): the teacher's days are collected
    # BEFORE the simulator processes start (fresh runs only; a resumed run already has it).
    pre = cfg.get("pretrain") if ckpt is None else None
    expert = None
    if pre:
        expert = collect_expert_data(
            env_kwargs(cfg),
            pre["days"],
            pre["first_day"],
            workers=pre.get("workers", 16),
            expert=pre.get("expert", "optimum-announced"),
        )
    venv = make_training_env(cfg, seed, t["n_envs"], restart)

    if ckpt is not None:
        venv = VecNormalize.load(str(ckpt[1]), venv.venv)  # restore rescaling statistics
        model = algorithm(design).load(ckpt[0], env=venv, device=t["device"])
        print(f"Resuming from {ckpt[0].name} ({model.num_timesteps} steps)")
    else:
        extra = {"reward_scale": t.get("reward_scale", 1.0)} if design == "central_credit" else {}
        model = algorithm(design)(
            "MlpPolicy", venv, seed=seed, device=t["device"], verbose=0, **cfg["ppo"], **extra
        )
    if expert is not None:
        history = pretrain_policy(
            model,
            venv,
            expert,
            epochs=pre["epochs"],
            batch_size=pre["batch_size"],
            learning_rate=pre["learning_rate"],
            log_std=pre.get("log_std"),
            train_critic=pre.get("critic", True),
            seed=seed,
        )
        pd.DataFrame(history).to_csv(run_dir / "pretrain.csv", index=False)
        model.save(run_dir / "pretrained_model.zip")  # the copied agent, before any PPO
        venv.save(str(run_dir / "pretrained_vecnormalize.pkl"))
        info.data["pretrain"] = {
            "samples": int(len(expert["obs"])),
            "teacher_seconds": float(expert["seconds"]),
            "final": history[-1],
        }
    # PPO(seed=...) also re-seeds the envs with seed, seed+1, ... (SB3 set_random_seed),
    # which would make training seeds 0 and 1 replay almost the same days. Restore our
    # day seeds; they take effect at the first reset, inside learn().
    venv.seed(first_day_seed(seed, t["n_envs"], restart))

    # wandb starts AFTER the simulator processes, so they do not inherit its threads.
    tags = [cfg["agent"], f"seed{seed}", cfg["env"]["reward_function"]]
    if cfg.get("variant"):
        tags.append(cfg["variant"])
    for key, value in (cfg["env"].get("reward_kwargs") or {}).items():
        tags.append(f"{key.replace('lambda_', 'lam_')}{value:g}")  # e.g. "lam_dep50"
    wandb_run = init_wandb(cfg, run_id, run_dir, group=cfg["name"], tags=tags)
    model.set_logger(make_logger(run_dir, restart, verbose))

    callbacks = CallbackList(
        [
            EpisodeMetricsCallback(run_dir / "train_episodes.csv", wandb_run, scale),
            ValidationCallback(
                scenario_seeds("validation", cfg["scenarios"]),
                env_kwargs(cfg),
                t["eval_every_steps"],
                run_dir,
                wandb_run,
                design=design,
                step_scale=scale,
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
        total_timesteps=t["total_timesteps"] * scale - start_steps,
        callback=callbacks,
        reset_num_timesteps=ckpt is None,
        progress_bar=False,
    )
    train_seconds = time.perf_counter() - t0
    model.save(run_dir / "final_model.zip")
    venv.save(str(run_dir / "final_vecnormalize.pkl"))
    venv.close()

    results: dict[str, Any] = {
        "agent_design": design,
        "total_timesteps": model.num_timesteps // scale,  # simulator steps
        "sb3_timesteps": model.num_timesteps,  # = simulator steps x chargers if shared
        "train_seconds": round(train_seconds, 1),
        "steps_per_second": round((model.num_timesteps - start_steps) / scale / train_seconds, 1),
    }
    test_episodes = None
    if run_test:
        test_summary, test_episodes = test_best_model(cfg, run_dir)
        results.update(test_summary)

    # Local files (source of truth) are complete BEFORE anything is sent to wandb,
    # so a wandb outage cannot lose results.
    info.finish(results=results)
    if wandb_run is not None:
        if test_episodes is not None:
            import wandb

            wandb_run.log({"test/episodes": wandb.Table(dataframe=test_episodes)})
        wandb_run.summary.update(results)
        wandb_run.save(str(run_dir / "run_info.json"), base_path=str(run_dir))
        wandb_run.finish()
    # Keep only best + final models (disk is limited); checkpoints were for resuming.
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
    return run_dir


def load_trained_policy(cfg: dict[str, Any], run_dir: Path, checkpoint: str | None = None):
    """The trained agent of a run, ready to evaluate: (policy, checkpoint name).

    checkpoint: "best" (chosen on validation days), "final", or "pretrained" (after the
    imitation warm start, before any PPO); default: best if saved.
    """
    name = checkpoint or ("best" if (run_dir / "best_model.zip").exists() else "final")
    design = agent_design(cfg)
    model = algorithm(design).load(run_dir / f"{name}_model.zip", device="cpu")
    vec_norm = None
    norm_path = run_dir / f"{name}_vecnormalize.pkl"
    if norm_path.exists():
        in_process = _in_process_vec_env(design)
        dummy = _wrap_sites(in_process([_env_factory(env_kwargs(cfg), design)]), design)
        vec_norm = VecNormalize.load(str(norm_path), dummy)
        vec_norm.training = False  # freeze the rescaling statistics during testing
    return make_sb3_policy(design, model, vec_norm, name=cfg["agent"]), name


def test_best_model(cfg: dict[str, Any], run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate the best validation checkpoint (or the final model) on the fixed test days,
    in the same world the agent was trained in (scripts/evaluate_model.py tests others)."""
    policy, name = load_trained_policy(cfg, run_dir)
    episodes, cars = evaluate_policy(
        policy, scenario_seeds("test", cfg["scenarios"]), env_kwargs(cfg)
    )
    episodes.insert(1, "checkpoint", name)
    episodes.to_csv(run_dir / "test_episodes.csv", index=False)
    cars.to_csv(run_dir / "test_evs.csv", index=False)
    means = episodes.drop(columns=["policy", "checkpoint", "seed"]).mean(numeric_only=True)
    summary = {f"test/{k}": float(v) for k, v in means.items()}
    summary["test/checkpoint"] = name
    return summary, episodes
