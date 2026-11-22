"""How does an agent hedge? (proposal Deliverable 1, item 7)

Hedging = keeping a safety margin because a car MIGHT leave before its announced
time. A hedging agent should (a) bring cars to their target EARLIER in their stay and
(b) sell less energy from a car late in its stay, when early leavers drive off.

To compare cars with different stays we put all of them on one clock:
    stay_fraction = (now − arrival) / (announced departure − arrival)
0 = just arrived, 1 = announced departure. Our early leavers cut 10–50 % of their
announced stay, so they leave between stay_fraction 0.5 and 0.9: the "risk window".

Input: the per-step trajectories written by `scripts/evaluate_model.py --trajectories`
(eval/metrics.py TrajectoryRecorder). Every number is first computed per trained agent
(seed), then averaged over seeds, so the spread shows seed-to-seed variation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RISK_WINDOW = (0.5, 0.9)  # where early leavers leave (cut 10–50 % of the stay)
N_BINS = 10


def load_trajectories(run_dirs: list[Path], p_early: float, split: str = "test") -> pd.DataFrame:
    """Trajectories of several trained agents in one test world, with a `run` column."""
    frames = []
    for run in run_dirs:
        path = Path(run) / "eval" / f"{split}_p{p_early:g}_trajectories.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"{path} (run evaluate_model.py --trajectories first)")
        df = pd.read_csv(path)
        df.insert(0, "run", Path(run).name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _add_columns(traj: pd.DataFrame) -> pd.DataFrame:
    traj = traj.copy()
    traj["sold_kwh"] = (-traj["energy_kwh"]).clip(lower=0)  # energy taken out of the car
    traj["selling"] = traj["energy_kwh"] < -0.02  # same threshold as TradeTracker
    frac = traj["stay_fraction"].clip(0, 0.999)
    traj["stay_bin"] = (frac * N_BINS).astype(int) / N_BINS + 0.5 / N_BINS  # bin centre
    return traj


def profile(traj: pd.DataFrame) -> pd.DataFrame:
    """Per run and stay-fraction bin: mean margin above target, kWh sold, share selling."""
    traj = _add_columns(traj)
    return (
        traj.groupby(["run", "stay_bin"])
        .agg(
            margin=("margin", "mean"),
            sold_kwh=("sold_kwh", "mean"),
            share_selling=("selling", "mean"),
            car_steps=("margin", "size"),
        )
        .reset_index()
    )


def risk_window_summary(traj: pd.DataFrame) -> pd.DataFrame:
    """Per run: behaviour inside the risk window (stay fraction 0.5–0.9).

    - margin: mean battery level above (+) / below (−) the target;
    - pct_at_target: share of car-steps already at or above the target;
    - sold_kwh: mean energy sold per car-step.
    """
    traj = _add_columns(traj)
    lo, hi = RISK_WINDOW
    w = traj[(traj["stay_fraction"] >= lo) & (traj["stay_fraction"] < hi)]
    return (
        w.groupby("run")
        .agg(
            margin=("margin", "mean"),
            pct_at_target=("margin", lambda m: 100 * float(np.mean(m >= -1e-3))),
            sold_kwh=("sold_kwh", "mean"),
            car_steps=("margin", "size"),
        )
        .reset_index()
    )


def over_seeds(per_run: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Mean and std over runs (seeds) of every numeric column."""
    numeric = [c for c in per_run.columns if c not in ("run", *(by or []))]
    if by:
        return per_run.groupby(by)[numeric].agg(["mean", "std"])
    return per_run[numeric].agg(["mean", "std"])
