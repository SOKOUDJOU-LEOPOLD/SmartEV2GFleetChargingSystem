"""Imitation warm start: teach PPO by example before it learns by trial and error.

Why (EXPERIMENT_LOG 2026-11-21, PPO gap diagnosis): PPO alone settles ~25 points
below the perfect-knowledge optimum whatever we tune. So before PPO training we
first copy a good teacher ("behaviour cloning", the classic imitation-learning
baseline [R20]), then let PPO improve from there.

The teacher is Optimum-announced (eval/optimum.py): the perfect plan that uses the
day-ahead prices and the ANNOUNCED departures, i.e. (almost) only what the agent can
see. It knows nothing about hidden early departures; learning to hedge is still PPO's
job. (Small differences: the plan also knows which cars will arrive later, and the
prices beyond what is already published.)

Three steps (`warm_start`):
1. collect: the teacher plays `days` training days (seeds from `first_day`, never
   validation/test days); we record what the agent would see, what the teacher did,
   and the reward. Cached on disk, so runs with other seeds reuse the same data.
2. normalise: set VecNormalize's observation and return statistics from that data,
   so PPO continues on the same scale.
3. pre-train: the actor learns to output the teacher's action (mean squared error);
   the critic learns how good each situation was on the teacher's days (discounted
   sum of the normalised rewards). Without this, the critic's first wrong guesses
   would push PPO away from what it just copied. Ref: [R8], [R21]
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import torch

from smartev2g.envs.make_env import REPO_ROOT, make_env
from smartev2g.eval.optimum import PerfectPlanPolicy

CACHE_DIR = REPO_ROOT / "results" / "expert_data"


# Config value `pretrain.expert` -> what the teacher plan knows (eval/optimum.py).
EXPERTS = {"optimum-announced": "announced", "optimum-real": "real"}


def _play_days(args: tuple[dict[str, Any], list[int], str]) -> dict[str, np.ndarray]:
    """Teacher plays these days (runs in a worker process)."""
    kwargs, days, knows = args
    lam = kwargs.get("reward_kwargs") or {}
    teacher = PerfectPlanPolicy(knows, lam.get("lambda_dep", 500.0), lam.get("lambda_over", 0.1))
    env = make_env(seed=days[0], **kwargs)
    obs_l, act_l, rew_l, day_l = [], [], [], []
    for day in days:
        obs, _ = env.reset(seed=day)
        teacher.reset(day)
        done = False
        while not done:
            action = teacher.act(obs, env)
            obs_l.append(obs)
            act_l.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            rew_l.append(reward)
            day_l.append(day)
            done = terminated or truncated
    return {
        "obs": np.array(obs_l, dtype=np.float32),
        "actions": np.array(act_l, dtype=np.float32),
        "rewards": np.array(rew_l, dtype=np.float64),
        "days": np.array(day_l, dtype=np.int64),
    }


def collect_expert_data(
    env_kwargs: dict[str, Any],
    days: int,
    first_day: int,
    workers: int = 16,
    expert: str = "optimum-announced",
) -> dict[str, np.ndarray]:
    """Teacher's (observation, action, reward) on `days` training days; cached on disk.

    expert: "optimum-announced" (trusts announced departures; PPO-Trusting/RiskAware)
    or "optimum-real" (knows the real ones; PPO-Oracle).
    """
    knows = EXPERTS[expert]
    spec = {"env": env_kwargs, "days": days, "first": first_day}
    if knows != "announced":  # (kept out of the key for the announced teacher: old caches)
        spec["knows"] = knows
    key = json.dumps(spec, sort_keys=True)
    path = CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()[:12]}.npz"
    if path.exists():
        return dict(np.load(path))
    seeds = list(range(first_day, first_day + days))
    chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
    t0 = time.perf_counter()
    if len(chunks) == 1:
        parts = [_play_days((env_kwargs, chunks[0], knows))]
    else:
        with ProcessPoolExecutor(len(chunks)) as pool:
            parts = list(pool.map(_play_days, [(env_kwargs, c, knows) for c in chunks]))
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["seconds"] = np.array(time.perf_counter() - t0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    (path.with_suffix(".json")).write_text(key)  # what the cache file contains
    return data


def discounted_returns(rewards: np.ndarray, days: np.ndarray, gamma: float) -> np.ndarray:
    """Sum of future rewards (discounted by gamma) from every step to the end of its day."""
    out = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        if i == len(rewards) - 1 or days[i + 1] != days[i]:
            running = 0.0  # a new day starts after step i
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def set_normalisation(vec_norm, data: dict[str, np.ndarray], gamma: float) -> np.ndarray:
    """Set VecNormalize's statistics from the teacher's days; returns normalised rewards.

    VecNormalize divides each reward by the std of the running discounted return
    (SB3 [R13]); we compute that return over the teacher's days the same way.
    """
    obs = data["obs"].astype(np.float64)
    vec_norm.obs_rms.mean = obs.mean(axis=0)
    vec_norm.obs_rms.var = obs.var(axis=0)
    vec_norm.obs_rms.count = float(len(obs))
    rewards = data["rewards"]
    if not vec_norm.norm_reward:
        return rewards
    running, rets = 0.0, np.zeros_like(rewards)
    for i, r in enumerate(rewards):  # forward running return, reset at each new day
        if i > 0 and data["days"][i] != data["days"][i - 1]:
            running = 0.0
        running = running * gamma + r
        rets[i] = running
    vec_norm.ret_rms.mean = float(rets.mean())
    vec_norm.ret_rms.var = float(rets.var())
    vec_norm.ret_rms.count = float(len(rets))
    scaled = rewards / np.sqrt(vec_norm.ret_rms.var + vec_norm.epsilon)
    return np.clip(scaled, -vec_norm.clip_reward, vec_norm.clip_reward)


def pretrain_policy(
    model,
    vec_norm,
    data: dict[str, np.ndarray],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    log_std: float | None,
    train_critic: bool,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Behaviour cloning (actor) + value regression (critic). Returns one row per epoch."""
    policy = model.policy
    gamma = model.gamma
    rewards = set_normalisation(vec_norm, data, gamma)
    obs = vec_norm.normalize_obs(data["obs"].astype(np.float64)).astype(np.float32)
    targets = discounted_returns(rewards, data["days"], gamma).astype(np.float32)
    actions = np.clip(data["actions"], -1, 1).astype(np.float32)

    obs_t = torch.as_tensor(obs, device=policy.device)
    act_t = torch.as_tensor(actions, device=policy.device)
    ret_t = torch.as_tensor(targets, device=policy.device)
    params = [p for name, p in policy.named_parameters() if name != "log_std"]
    optimiser = torch.optim.Adam(params, lr=learning_rate)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(obs))
        sums = {"actor_mse": 0.0, "critic_mse": 0.0}
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start : start + batch_size], device=policy.device)
            features = policy.extract_features(obs_t[idx], policy.pi_features_extractor)
            latent_pi, latent_vf = policy.mlp_extractor(features)
            actor = torch.mean((policy.action_net(latent_pi) - act_t[idx]) ** 2)
            loss = actor
            critic = torch.zeros(())
            if train_critic:
                critic = torch.mean((policy.value_net(latent_vf).squeeze(-1) - ret_t[idx]) ** 2)
                loss = loss + critic
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            n = len(idx)
            sums["actor_mse"] += actor.item() * n
            sums["critic_mse"] += critic.item() * n
        history.append({"epoch": epoch + 1, **{k: v / len(order) for k, v in sums.items()}})
    if log_std is not None:  # less random actions at the start of PPO: stay near the teacher
        with torch.no_grad():
            policy.log_std.fill_(log_std)
    return history
