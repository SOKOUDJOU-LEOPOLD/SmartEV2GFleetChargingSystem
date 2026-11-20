"""Present each charger of a site as its own environment to Stable-Baselines3 (ADR-0004).

Two layers:
1. `SiteChargerEnv` (inside each simulator process): one EV2Gym site whose
   observation is one row per charger (envs/charger_obs.py) and whose reward is
   one number per charger (rewards/shortfall_reward.py, per-charger split).
2. `ChargerVecEnv` (in the main process): takes K such sites (e.g. a SubprocVecEnv)
   and shows PPO K x 20 small environments: each with a 26-number observation and
   ONE action. So one shared network decides for every charger. Ref: [R18]

Site-level episode information (return of the whole site, EV2Gym statistics,
reward terms, day seed) is attached to charger 0 of each site only, so every
simulated day is logged exactly once.
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from smartev2g.envs.charger_obs import N_FEATURES, charger_observations


class SiteChargerEnv(gym.Wrapper):
    """One site; observation (n_chargers, 26), reward (n_chargers,), action (n_chargers,)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        base = env.unwrapped
        if any(cs.n_ports != 1 for cs in base.charging_stations):
            raise ValueError("The shared per-charger policy assumes 1 port per charger")
        self.n_chargers = len(base.charging_stations)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.n_chargers, N_FEATURES), dtype=np.float64
        )
        self._return, self._length, self._t0 = 0.0, 0, time.time()

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        self._return, self._length, self._t0 = 0.0, 0, time.time()
        return charger_observations(self.env.unwrapped), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(np.asarray(action).reshape(-1))
        base = self.env.unwrapped
        rewards = getattr(base, "charger_rewards", None)
        if rewards is None:
            raise ValueError("Per-charger rewards need reward_function: squared_shortfall")
        self._return += float(reward)
        self._length += 1
        if terminated or truncated:
            # Same format as SB3's Monitor: the site's raw return for this day. [R13]
            info = {
                **info,
                "episode": {
                    "r": self._return,
                    "l": self._length,
                    "t": round(time.time() - self._t0, 6),
                },
            }
        obs = charger_observations(base)
        return obs, np.array(rewards, dtype=np.float64), terminated, truncated, info


class SiteDummyVecEnv(DummyVecEnv):
    """SB3's DummyVecEnv (all sites in this process) with room for one reward per
    charger: the original stores a single reward number per env."""

    def __init__(self, env_fns) -> None:
        super().__init__(env_fns)
        n_chargers = self.observation_space.shape[0]
        self.buf_rews = np.zeros((self.num_envs, n_chargers), dtype=np.float32)


class ChargerVecEnv(VecEnv):
    """K sites (a VecEnv of SiteChargerEnv) shown to SB3 as K x n_chargers environments."""

    def __init__(self, venv: VecEnv) -> None:
        self.venv = venv
        self.n_sites = venv.num_envs
        self.n_chargers, n_features = venv.observation_space.shape
        obs_space = spaces.Box(-np.inf, np.inf, shape=(n_features,), dtype=np.float64)
        act_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        super().__init__(self.n_sites * self.n_chargers, obs_space, act_space)

    # --- core API -------------------------------------------------------------
    def reset(self):
        obs = self.venv.reset()
        return obs.reshape(self.num_envs, -1)

    def step_async(self, actions) -> None:
        self.venv.step_async(np.asarray(actions).reshape(self.n_sites, self.n_chargers))

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        flat_infos = []
        for site_info in infos:
            terminal = site_info.get("terminal_observation")
            for i in range(self.n_chargers):
                # Site-level info (episode return, statistics) only on charger 0.
                info = {k: v for k, v in site_info.items() if k != "terminal_observation"}
                if i > 0:
                    info = {"TimeLimit.truncated": site_info.get("TimeLimit.truncated", False)}
                if terminal is not None:
                    info["terminal_observation"] = terminal[i]
                flat_infos.append(info)
        return (
            obs.reshape(self.num_envs, -1),
            np.asarray(rewards, dtype=np.float32).reshape(-1),
            np.repeat(dones, self.n_chargers),
            flat_infos,
        )

    def seed(self, seed: int | None = None):
        # Seeds are per simulated day (site), not per charger.
        return self.venv.seed(seed)

    def close(self) -> None:
        self.venv.close()

    # --- attribute helpers: answer per site, repeated for each of its chargers ---
    def _sites(self, indices):
        if indices is None:
            return list(range(self.n_sites)), self.num_envs
        idx = [indices] if isinstance(indices, int) else list(indices)
        return sorted({i // self.n_chargers for i in idx}), len(idx)

    def get_attr(self, attr_name, indices=None):
        sites, _ = self._sites(indices)
        values = dict(zip(sites, self.venv.get_attr(attr_name, sites), strict=True))
        idx = range(self.num_envs) if indices is None else indices
        idx = [idx] if isinstance(idx, int) else idx
        return [values[i // self.n_chargers] for i in idx]

    def set_attr(self, attr_name, value, indices=None) -> None:
        sites, _ = self._sites(indices)
        self.venv.set_attr(attr_name, value, sites)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        sites, _ = self._sites(indices)
        return self.venv.env_method(method_name, *method_args, indices=sites, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        sites, n = self._sites(indices)
        wrapped = self.venv.env_is_wrapped(wrapper_class, sites)
        return [all(wrapped)] * n
