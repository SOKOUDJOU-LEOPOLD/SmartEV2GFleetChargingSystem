"""Describe what a typical simulated day looks like (Phase 1 exploration).

Runs several episodes with the "charge as fast as possible" rule (AFAP, [R1])
and prints statistics about the cars: how many arrive, how long they stay,
how full they arrive, and how many still leave below their target even when
charged at full speed. These numbers guide the wrapper design (Phase 2) and are
recorded in docs/decisions/ADR-0002-env-internals.md.

Saved in results/<run_id>/ (CODING_RULES.md §5-6):
- evs.csv       one row per car (seed, arrival, departure, SoC, below target?)
- episodes.csv  one row per episode (seed, number of cars, % below target)
- run_info.json config, seeds, git commit, versions, hardware, timing, summary

Usage:
    python scripts/inspect_env.py --episodes 30 --config configs/env/v2g_profitmax_20port.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible

from smartev2g.envs.make_env import DEFAULT_ENV_CONFIG, make_env, observation_names
from smartev2g.utils.run_info import RESULTS_DIR, RunInfo, make_run_dir, make_run_id

# A car counts as "below target" if it is more than this fraction short (float rounding).
TARGET_TOLERANCE = 1e-3


def summarize(values) -> str:
    v = np.asarray(values, dtype=float)
    return (
        f"mean {v.mean():7.2f} | median {np.median(v):7.2f} | "
        f"min {v.min():7.2f} | max {v.max():7.2f}"
    )


def run_episode(config: str, seed: int) -> tuple[list[dict], dict]:
    """Play one AFAP episode; return per-car rows and env facts."""
    env = make_env(config, seed=seed)
    env.reset(seed=seed)
    afap = ChargeAsFastAsPossible()
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(afap.get_action(env))
        done = terminated or truncated

    rows = []
    for i, ev in enumerate(env.EVs):  # env.EVs = every car that arrived, in arrival order
        soc = ev.current_capacity / ev.battery_capacity
        target = ev.desired_capacity / ev.battery_capacity
        rows.append(
            {
                "seed": seed,
                "ev_index": i,  # unique within the episode
                "charger": ev.location,
                "port": ev.id,  # EV2Gym's ev.id is the port number, not a unique id
                "arrival_step": ev.time_of_arrival,
                "departure_step": ev.time_of_departure,
                "stay_steps": ev.time_of_departure - ev.time_of_arrival,
                "stay_hours": (ev.time_of_departure - ev.time_of_arrival) * env.timescale / 60,
                "battery_kwh": ev.battery_capacity,
                "soc_arrival": ev.battery_capacity_at_arrival / ev.battery_capacity,
                "soc_final": soc,
                "soc_target": target,
                # Car left (or episode ended) below its target even at full speed.
                "below_target": bool(soc < target - TARGET_TOLERANCE),
            }
        )
    facts = {
        "timescale_min": env.timescale,
        "episode_steps": env.simulation_length,
        "n_actions": env.action_space.shape[0],
        "obs_size": len(observation_names(env)),
    }
    return rows, facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer")

    seeds = list(range(args.episodes))
    run_dir = make_run_dir(make_run_id("inspect-env"), root=args.results_dir)
    info = RunInfo(run_dir, "scripts/inspect_env.py", args.config, seeds, vars(args))

    all_rows, facts = [], {}
    for seed in seeds:
        rows, facts = run_episode(args.config, seed)
        all_rows += rows

    evs = pd.DataFrame(all_rows)
    episodes = (
        evs.groupby("seed")
        .agg(n_evs=("ev_index", "size"), pct_below_target=("below_target", "mean"))
        .reset_index()
    )
    episodes["pct_below_target"] *= 100
    evs.to_csv(run_dir / "evs.csv", index=False)
    episodes.to_csv(run_dir / "episodes.csv", index=False)

    summary = {
        **facts,
        "episodes": args.episodes,
        "cars_per_episode_mean": float(episodes["n_evs"].mean()),
        "stay_hours_mean": float(evs["stay_hours"].mean()),
        "stay_steps_median": float(evs["stay_steps"].median()),
        "soc_arrival_mean": float(evs["soc_arrival"].mean()),
        "pct_cars_below_target_afap": float(evs["below_target"].mean() * 100),
    }
    info.finish(results=summary)

    print(f"Config: {args.config}  |  episodes: {args.episodes}")
    print(f"Observation size: {facts['obs_size']}")
    print(
        f"Actions: {facts['n_actions']} (one per port), step = {facts['timescale_min']} min, "
        f"episode = {facts['episode_steps']} steps"
    )
    print(f"Cars per episode            : {summarize(episodes['n_evs'])}")
    print(f"Announced stay (steps)      : {summarize(evs['stay_steps'])}")
    print(f"Announced stay (hours)      : {summarize(evs['stay_hours'])}")
    print(f"SoC at arrival              : {summarize(evs['soc_arrival'])}")
    print(f"SoC at departure (AFAP)     : {summarize(evs['soc_final'])}")
    print(f"Share below target with AFAP: {summary['pct_cars_below_target_afap']:.1f}% of cars")
    print(f"Saved: {run_dir}")


if __name__ == "__main__":
    main()
