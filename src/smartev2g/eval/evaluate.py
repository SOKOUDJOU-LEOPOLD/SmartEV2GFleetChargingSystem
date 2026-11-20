"""Play a policy on a fixed list of simulated days and collect the results.

Used for validation during training, for final tests, and for the baselines,
so every agent is measured in exactly the same way on exactly the same days.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml

from smartev2g.envs.make_env import make_env, resolve_path
from smartev2g.eval.metrics import episode_row, ev_rows


class Policy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def act(self, obs: np.ndarray, env) -> np.ndarray: ...


class SB3Policy:
    """Wrap a trained Stable-Baselines3 model so it looks like the other policies.

    If training used VecNormalize, the model expects *rescaled* observations, so
    we rescale them with the statistics saved at training time. Ref: [R13]
    """

    def __init__(self, model, vec_normalize=None, name: str = "PPO", deterministic=True):
        self.model = model
        self.vec_normalize = vec_normalize
        self.name = name
        # deterministic=True: take the most likely action (no exploration noise).
        self.deterministic = deterministic

    def reset(self, seed: int) -> None:
        pass

    def act(self, obs: np.ndarray, env) -> np.ndarray:
        if self.vec_normalize is not None:
            obs = self.vec_normalize.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action


def scenario_seeds(split: str, scenarios_path: str | Path = "configs/eval/scenarios.yaml"):
    """Seeds of the fixed 'validation' or 'test' days (configs/eval/scenarios.yaml)."""
    spec = yaml.safe_load(resolve_path(scenarios_path).read_text())[split]
    return list(range(spec["start"], spec["start"] + spec["count"]))


def run_episode(policy: Policy, env, seed: int):
    """Play one simulated day; return (episode summary row, per-car rows).

    `env.reset(seed)` rebuilds the whole day (date, cars, prices), so one env can
    be reused for many days: a reused env gives exactly the same day as a fresh
    one (checked in tests/test_evaluate.py). Building an env takes ~2 s.
    """
    obs, _ = env.reset(seed=seed)
    policy.reset(seed)
    total, done, info = 0.0, False, {}
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs, env))
        total += reward
        done = terminated or truncated
    evs = ev_rows(env, seed)
    row = episode_row(seed, total, info, evs)
    # Our reward's separate terms for this day (only rewards that record them).
    for k, v in (getattr(env, "reward_terms_episode", None) or {}).items():
        row[f"reward_{k}"] = float(v)
    return row, evs


def evaluate_policy(
    policy: Policy, seeds: list[int], env_kwargs: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run `policy` on every seed. Returns (episodes table, cars table)."""
    env = make_env(seed=seeds[0], **(env_kwargs or {}))
    episodes, cars = [], []
    for seed in seeds:
        ep, evs = run_episode(policy, env, seed)
        episodes.append(ep)
        cars += evs
    episodes_df = pd.DataFrame(episodes)
    cars_df = pd.DataFrame(cars)
    episodes_df.insert(0, "policy", policy.name)
    if len(cars_df):
        cars_df.insert(0, "policy", policy.name)
    return episodes_df, cars_df
