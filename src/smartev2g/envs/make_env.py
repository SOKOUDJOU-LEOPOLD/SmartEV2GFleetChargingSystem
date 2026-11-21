"""Build the EV2Gym charging-site simulator in one standard way.

Every script (training, evaluation, tests) creates its environment through
`make_env`, so all experiments use exactly the same setup.

Plain-language picture of the environment (Ref: [R1] EV2Gym):
- A site with 20 chargers shares one transformer (max 100 kW).
- Every 15 minutes (one "step") the agent picks, for each charger, a number in
  [-1, 1]: positive = charge the car, negative = sell energy back to the grid
  (V2G), 0 = do nothing.
- Cars arrive and leave during the day. Each car announces when it will leave.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from importlib import resources
from pathlib import Path

import yaml
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent import reward as ev2gym_reward
from ev2gym.rl_agent import state as ev2gym_state

from smartev2g.envs import state as our_state
from smartev2g.rewards import sanity as sanity_rewards
from smartev2g.rewards import shortfall_reward

# Repo root = three folders above this file (src/smartev2g/envs/make_env.py).
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ENV_CONFIG = "configs/env/v2g_profitmax_20port.yaml"

# Name -> function tables, so YAML configs can choose functions by name.
# A "state function" builds what the agent sees (its observation).
# A "reward function" gives the score after every step.
STATE_FUNCTIONS: dict[str, Callable] = {
    "V2G_profit_max": ev2gym_state.V2G_profit_max,
    # Ours: V2G_profit_max + per car the steps of charging still needed and the
    # spare time (laxity) until its announced departure (envs/state.py).
    "V2G_profit_max_laxity": our_state.V2G_profit_max_laxity,
}
# Numbers per port in each state function (for observation_names).
PORT_FEATURES = {
    "V2G_profit_max": ["soc", "t_dep"],
    "V2G_profit_max_laxity": our_state.PORT_FEATURES_LAXITY,
}
REWARD_FUNCTIONS: dict[str, Callable] = {
    # EV2Gym's default V2G reward: profit minus a penalty for unhappy drivers.
    # Used only for the Phase 1 sanity check; Phase 2 adds our own reward.
    "profit_maximization": ev2gym_reward.profit_maximization,
    # Phase 1 pipeline check only: reward = kWh delivered to cars; best policy = AFAP.
    "energy_delivered": sanity_rewards.energy_delivered,
}
# Rewards that need settings (e.g. λ weights): name -> function that builds the reward
# from those settings (`reward_kwargs` in make_env / the `env` section of configs).
REWARD_FACTORIES: dict[str, Callable] = {
    # OUR reward (Phase 2+): profit − λ_over·overload² − λ_dep·shortfall². See its module.
    "squared_shortfall": shortfall_reward.make_squared_shortfall,
}


def get_reward_function(name: str, reward_kwargs: dict | None = None) -> Callable:
    """Look up a reward by name; rewards with settings are built from `reward_kwargs`."""
    if name in REWARD_FACTORIES:
        return REWARD_FACTORIES[name](**(reward_kwargs or {}))
    if reward_kwargs:
        raise ValueError(f"Reward {name!r} takes no settings, got {reward_kwargs}")
    if name not in REWARD_FUNCTIONS:
        known = sorted(REWARD_FUNCTIONS) + sorted(REWARD_FACTORIES)
        raise KeyError(f"Unknown reward {name!r}; known: {known}")
    return REWARD_FUNCTIONS[name]


# Number of future prices the V2G_profit_max observation contains (fixed in EV2Gym).
PRICE_HORIZON = our_state.PRICE_HORIZON


def resolve_path(path: str | Path) -> Path:
    """Turn a repo-relative path (e.g. 'configs/env/x.yaml') into an absolute one."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# Prefix for files shipped inside an installed package, e.g.
# "pkg://ev2gym/data/ev_specs_v2g_enabled2024.json". Works for any install location.
PKG_PREFIX = "pkg://"

# Config keys that hold file paths and must be resolved before EV2Gym reads them.
PATH_KEYS = ["ev_specs_file"]


def resolve_data_path(value: str) -> str:
    """'pkg://<package>/<file>' -> file inside that installed package; else repo-relative."""
    if value.startswith(PKG_PREFIX):
        package, _, rel = value[len(PKG_PREFIX) :].partition("/")
        path = Path(str(resources.files(package).joinpath(rel)))
    else:
        path = resolve_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Config file path not found: {value!r} -> {path}")
    return str(path)


def load_env_config(config_path: str | Path = DEFAULT_ENV_CONFIG) -> dict:
    """Read an EV2Gym YAML config and turn its file paths into absolute paths."""
    config = yaml.safe_load(resolve_path(config_path).read_text())
    for key in PATH_KEYS:
        if key in config:
            config[key] = resolve_data_path(config[key])
    return config


def make_env(
    config_path: str | Path = DEFAULT_ENV_CONFIG,
    seed: int | None = None,
    state_function: str = "V2G_profit_max",
    reward_function: str = "profit_maximization",
    reward_kwargs: dict | None = None,
    verbose: bool = False,
) -> EV2Gym:
    """Create an EV2Gym environment.

    Args:
        config_path: EV2Gym YAML config (repo-relative or absolute).
        seed: random seed. The same seed gives the same cars, prices and day,
            so experiments can be repeated exactly. Ref: [R14]
        state_function: name in STATE_FUNCTIONS (what the agent observes).
        reward_function: name in REWARD_FUNCTIONS or REWARD_FACTORIES (how the agent
            is scored).
        reward_kwargs: settings of the reward, e.g. {"lambda_dep": 50, "lambda_over": 0.1}.
        verbose: print EV2Gym's step-by-step messages.

    Note:
        EV2Gym re-seeds Python's and NumPy's *global* random generators on every
        reset. Our own code must therefore use private generators
        (`np.random.default_rng(seed)`) so it does not change EV2Gym's random
        draws (see ADR-0002).
    """
    # EV2Gym only accepts a config *file*, so we write the resolved config to a
    # temporary file, build the env (it reads everything in its constructor),
    # then delete the file.
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="ev2gym_cfg_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(load_env_config(config_path), f)
        return EV2Gym(
            config_file=tmp,
            seed=seed,
            state_function=STATE_FUNCTIONS[state_function],
            reward_function=get_reward_function(reward_function, reward_kwargs),
            save_replay=False,
            save_plots=False,
            lightweight_plots=True,
            verbose=verbose,
        )
    finally:
        os.remove(tmp)


def observation_names(env: EV2Gym, state_function: str = "V2G_profit_max") -> list[str]:
    """Name of every number in the observation, in order.

    Layout (Ref: [R1], EV2Gym `rl_agent/state.py`; ours in envs/state.py):
      [t, P_prev, price_0..price_19, then for each port k its features]
    - t: current step number
    - P_prev: total charging power of the site at the previous step (kW)
    - price_i: electricity price i steps ahead (a 20-step forecast)
    - soc_k: battery level of the car at port k (0 = empty, 1 = full; 0 if no car)
    - t_dep_k: steps left until the car's ANNOUNCED departure (0 if no car).
      This is the number our wrapper keeps "wrong" for early leavers (POMDP, [R10]).
    - steps_needed_k, laxity_k (V2G_profit_max_laxity only): full-power charging
      steps still needed, and spare time = t_dep_k − steps_needed_k.
    """
    names = ["t", "P_prev"] + [f"price_{i}" for i in range(PRICE_HORIZON)]
    for k in range(env.number_of_ports):
        names += [f"{feature}_{k}" for feature in PORT_FEATURES[state_function]]
    return names
