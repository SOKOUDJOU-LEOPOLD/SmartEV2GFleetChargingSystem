"""Describe what a typical simulated day looks like (Phase 1 exploration).

Runs several episodes with the "charge as fast as possible" rule (AFAP, [R1])
and prints statistics about the cars: how many arrive, how long they stay,
how full they arrive, and how many still leave below their target even when
charged at full speed. These numbers guide the wrapper design (Phase 2) and are
recorded in docs/decisions/ADR-0002-env-internals.md.

Usage:
    python scripts/inspect_env.py --episodes 20 --config configs/env/v2g_profitmax_20port.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible

from smartev2g.envs.make_env import DEFAULT_ENV_CONFIG, make_env, observation_names


def summarize(values: list[float]) -> str:
    v = np.asarray(values, dtype=float)
    return (
        f"mean {v.mean():7.2f} | median {np.median(v):7.2f} | "
        f"min {v.min():7.2f} | max {v.max():7.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    n_evs, stay_steps, arrival_soc, final_soc, short_afap = [], [], [], [], []
    timescale = None
    for seed in range(args.episodes):
        env = make_env(args.config, seed=seed)
        env.reset(seed=seed)
        timescale = env.timescale
        afap = ChargeAsFastAsPossible()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(afap.get_action(env))
            done = terminated or truncated

        n_evs.append(len(env.EVs))
        for ev in env.EVs:
            stay_steps.append(ev.time_of_departure - ev.time_of_arrival)
            arrival_soc.append(ev.battery_capacity_at_arrival / ev.battery_capacity)
            soc = ev.current_capacity / ev.battery_capacity
            target = ev.desired_capacity / ev.battery_capacity
            final_soc.append(soc)
            # Car left (or episode ended) below its target even at full speed.
            short_afap.append(float(soc < target - 1e-3))

    print(f"Config: {args.config}  |  episodes: {args.episodes}")
    print(f"Observation size: {len(observation_names(env))}  (first: {observation_names(env)[:3]})")
    print(
        f"Actions: {env.action_space.shape[0]} (one per port), step = {timescale} min, "
        f"episode = {env.simulation_length} steps"
    )
    print(f"Cars per episode            : {summarize(n_evs)}")
    print(f"Announced stay (steps)      : {summarize(stay_steps)}")
    print(f"Announced stay (hours)      : {summarize([s * timescale / 60 for s in stay_steps])}")
    print(f"SoC at arrival              : {summarize(arrival_soc)}")
    print(f"SoC at departure (AFAP)     : {summarize(final_soc)}")
    print(f"Share below target with AFAP: {np.mean(short_afap) * 100:.1f}% of cars")


if __name__ == "__main__":
    main()
