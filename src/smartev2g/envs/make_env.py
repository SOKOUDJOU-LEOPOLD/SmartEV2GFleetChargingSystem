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

import numpy as np
import yaml
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent import reward as ev2gym_reward
from ev2gym.rl_agent import state as ev2gym_state

from smartev2g.envs import state as our_state
from smartev2g.envs.departure_models import make_departure_model
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
    # Same + every published day-ahead price of the next 24 h instead of 5 h.
    "V2G_profit_max_laxity_dayahead": our_state.V2G_profit_max_laxity_dayahead,
    # PPO-Oracle: same, but the REAL departure time of early leavers (Phase 4).
    "V2G_profit_max_laxity_dayahead_oracle": our_state.V2G_profit_max_laxity_dayahead_oracle,
}
# Numbers per port in each state function (for observation_names).
PORT_FEATURES = {
    "V2G_profit_max": ["soc", "t_dep"],
    "V2G_profit_max_laxity": our_state.PORT_FEATURES_LAXITY,
    "V2G_profit_max_laxity_dayahead": our_state.PORT_FEATURES_LAXITY,
    "V2G_profit_max_laxity_dayahead_oracle": our_state.PORT_FEATURES_LAXITY,
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


# Fixed stream id: the hidden departures of a day come from their own random
# generator, seeded by (day seed, this id), never from EV2Gym's global ones (ADR-0002).
DEPARTURE_RNG_STREAM = 20261121


ARRIVAL_RULES = ("ev2gym", "target")


class ProjectEV2Gym(EV2Gym):
    """EV2Gym with our additions, all applied right after `reset()` builds the
    day's car list (`EVs_profiles`; arriving cars are copies of these):

    1. Charging target (ADR-0006): every driver wants `charging_target` × battery.
       Why not `desired_capacity` in the EV2Gym config? EV2Gym also uses it when it
       creates cars (arrival levels above the target are re-drawn at random), which
       would change the simulated days themselves.
    2. Hidden early departures (ADR-0002): a departure model decides, once per day,
       which cars will REALLY leave early and when (`departure_plan`). The car keeps its
       ANNOUNCED time (`ev.time_of_departure`), which is all the agent sees. Right
       before the step at the real departure ("just-in-time swap"), we set
       `ev.time_of_departure` to the real step, so EV2Gym's normal code removes the car
       in that step; the next observation no longer contains it, so the real time
       never appears in any observation. We also refresh `max_energy_AFAP`, which
       EV2Gym cached at plug-in from the announced time (ADR-0002).
    3. Arrival charge (ADR-0008), `arrival_rule`:
       - "ev2gym" (default): EV2Gym's rule, arrival = FULL battery − the energy the
         driver takes (ElaadNL data). With an 80 % target, drivers then need only
         ~5 points on average and 37 % arrive already above target.
       - "target": arrival = TARGET − the energy the driver takes, so every driver
         still adds exactly the kWh of the data; only the stopping point is the
         target. Never below EV2Gym's minimum battery level (`min_battery_capacity`).
    """

    def __init__(
        self, *args, charging_target=None, departure_model=None, arrival_rule="ev2gym", **kwargs
    ) -> None:
        if charging_target is not None and not 0 < charging_target <= 1:
            raise ValueError("charging_target must be in (0, 1]")
        if arrival_rule not in ARRIVAL_RULES:
            raise ValueError(f"arrival_rule must be one of {ARRIVAL_RULES}")
        if arrival_rule == "target" and charging_target is None:
            raise ValueError('arrival_rule "target" needs a charging_target')
        # Set first: EV2Gym's __init__ already calls reset().
        self.charging_target = charging_target
        self.arrival_rule = arrival_rule
        self.departure_model = departure_model
        self.departure_plan: dict[tuple[int, int], tuple[int, int]] = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def car_key(ev) -> tuple[int, int]:
        """(charger, arrival step): unique per car in a day (one port per charger)."""
        return (ev.location, ev.time_of_arrival)

    @staticmethod
    def shift_arrival(ev, target: float) -> None:
        """Arrival = target − the energy the driver takes (ADR-0008).

        EV2Gym built the car as arrival = full − energy, so the energy is
        `battery − arrival`; we keep it and move the arrival down by (1 − target) ×
        battery. Arriving cars are copies made with `ev.reset()`, which restarts from
        `battery_capacity_at_arrival`, so changing the profile is enough."""
        shifted = ev.battery_capacity_at_arrival - (1 - target) * ev.battery_capacity
        ev.battery_capacity_at_arrival = max(shifted, ev.min_battery_capacity)
        ev.reset()  # current level and EV2Gym's required_energy follow the new arrival

    def reset(self, seed=None, options=None, **kwargs):
        _, info = super().reset(seed=seed, options=options, **kwargs)
        if self.charging_target is not None:
            for ev in self.EVs_profiles:
                ev.desired_capacity = self.charging_target * ev.battery_capacity
                if self.arrival_rule == "target":
                    self.shift_arrival(ev, self.charging_target)
        # departure_plan: car -> (announced step, real step), only for early leavers.
        self.departure_plan = {}
        if self.departure_model is not None:
            rng = np.random.default_rng([int(self.seed), DEPARTURE_RNG_STREAM])
            for ev in self.EVs_profiles:
                real = self.departure_model.sample(ev.time_of_arrival, ev.time_of_departure, rng)
                if real is not None:
                    self.departure_plan[self.car_key(ev)] = (ev.time_of_departure, real)
        return self._get_observation(), info

    def step(self, actions, visualize=False):
        # Just-in-time swap for cars whose hidden real departure is THIS step.
        for cs in self.charging_stations:
            for ev in cs.evs_connected:
                if ev is None:
                    continue
                plan = self.departure_plan.get(self.car_key(ev))
                if plan is not None and plan[1] == self.current_step:
                    ev.time_of_departure = plan[1]
                    ev.calculate_max_energy_with_AFAP(cs.get_max_power())
        return super().step(actions, visualize=visualize)


def make_env(
    config_path: str | Path = DEFAULT_ENV_CONFIG,
    seed: int | None = None,
    state_function: str = "V2G_profit_max",
    reward_function: str = "profit_maximization",
    reward_kwargs: dict | None = None,
    charging_target: float | None = None,
    departures: dict | None = None,
    arrival_rule: str = "ev2gym",
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
        charging_target: battery level (0..1) every driver wants when leaving, e.g. 0.8
            (ADR-0006). None = the EV2Gym config's `desired_capacity` (1.0 = full).
        departures: hidden early departures, e.g. {"p_early": 0.1, "cut_min": 0.1,
            "cut_max": 0.5} (envs/departure_models.py). None = everyone leaves on time.
        arrival_rule: "ev2gym" (arrival = full − the driver's energy) or "target"
            (arrival = charging_target − the driver's energy; ADR-0008).
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
        env_class, extra = EV2Gym, {}
        if charging_target is not None or departures or arrival_rule != "ev2gym":
            env_class = ProjectEV2Gym
            extra = {
                "charging_target": charging_target,
                "departure_model": make_departure_model(departures),
                "arrival_rule": arrival_rule,
            }
        return env_class(
            config_file=tmp,
            seed=seed,
            state_function=STATE_FUNCTIONS[state_function],
            reward_function=get_reward_function(reward_function, reward_kwargs),
            save_replay=False,
            save_plots=False,
            lightweight_plots=True,
            verbose=verbose,
            **extra,
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
    - (…_dayahead: price_0..price_95 = every PUBLISHED day-ahead price of the next
      24 h, 0 if not yet published; n_known = how many are real)
    - steps_needed_k, laxity_k (laxity state functions only): full-power charging
      steps still needed, and spare time = t_dep_k − steps_needed_k.
    """
    if state_function.startswith("V2G_profit_max_laxity_dayahead"):  # 24 h of prices + count
        prices = [f"price_{i}" for i in range(our_state.DAYAHEAD_HORIZON)] + ["n_known"]
    else:
        prices = [f"price_{i}" for i in range(PRICE_HORIZON)]
    names = ["t", "P_prev"] + prices
    for k in range(env.number_of_ports):
        names += [f"{feature}_{k}" for feature in PORT_FEATURES[state_function]]
    return names
