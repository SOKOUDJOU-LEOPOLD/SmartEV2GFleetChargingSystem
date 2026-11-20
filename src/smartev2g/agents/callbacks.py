"""Code that runs *during* PPO training to record what the agent is learning.

SB3 calls a callback after every environment step (Ref: [R13], SB3 callbacks).
- EpisodeMetricsCallback: when a simulated day ends, save its statistics
  (profit, energy, overload, driver satisfaction) -> train_episodes.csv
- ValidationCallback: every N steps, test the current agent on the fixed
  validation days, save the scores -> val_curve.csv, and keep the best model.
Both also send the numbers to wandb.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from smartev2g.eval.evaluate import evaluate_policy, make_sb3_policy
from smartev2g.eval.metrics import EV2GYM_STATS
from smartev2g.utils.logging import CSVLogger


class EpisodeMetricsCallback(BaseCallback):
    """Log one row per finished training episode (simulated day)."""

    def __init__(self, csv_path: Path, wandb_run=None, step_scale: int = 1) -> None:
        super().__init__()
        self.csv = CSVLogger(csv_path)
        # SB3 counts one step per environment; with the shared per-charger policy one
        # simulated step = n_chargers SB3 steps. We log SIMULATOR steps everywhere.
        self.step_scale = step_scale
        self.wandb_run = wandb_run
        self.n_episodes = 0

    def _on_step(self) -> bool:
        for env_idx, done in enumerate(self.locals["dones"]):
            info = self.locals["infos"][env_idx]
            # Per-charger envs carry the site's episode info on charger 0 only.
            if not done or "episode" not in info:
                continue
            self.n_episodes += 1
            row: dict[str, Any] = {
                "timesteps": self.num_timesteps // self.step_scale,
                "episode": self.n_episodes,
                "env_idx": env_idx,
                "day_seed": info.get("day_seed"),  # which simulated day was played
                # Monitor's numbers are the RAW reward (before VecNormalize rescaling).
                "return": info["episode"]["r"],
                "length": info["episode"]["l"],
            }
            row.update({k: float(info[k]) for k in EV2GYM_STATS if k in info})
            # Our reward's separate terms for this day (econ, overload, departure penalty).
            for k, v in info.get("reward_terms_episode", {}).items():
                row[f"reward_{k}"] = float(v)
            self.csv.log(row)
            if self.wandb_run is not None:
                metrics = {f"train_ep/{k}": v for k, v in row.items() if k != "env_idx"}
                self.wandb_run.log({"timesteps": row["timesteps"], **metrics})
        return True


class ValidationCallback(BaseCallback):
    """Every `every_steps`, evaluate on the fixed validation days and keep the best model.

    "Best" = highest mean return on the validation days. The final test uses this
    checkpoint, never one chosen by looking at test days. Ref: [R14]
    """

    def __init__(
        self,
        seeds: list[int],
        env_kwargs: dict[str, Any],
        every_steps: int,
        run_dir: Path,
        wandb_run=None,
        design: str = "site",
        step_scale: int = 1,
    ) -> None:
        super().__init__()
        self.design = design  # which policy adapter to evaluate with (eval/evaluate.py)
        self.step_scale = step_scale  # SB3 steps per simulator step (see EpisodeMetricsCallback)
        self.seeds = seeds
        self.env_kwargs = env_kwargs
        self.every_steps = every_steps
        self.run_dir = Path(run_dir)
        self.csv = CSVLogger(self.run_dir / "val_curve.csv")
        self.wandb_run = wandb_run
        self.best_return = -np.inf
        val_path = self.run_dir / "val_curve.csv"
        if val_path.exists():  # resuming: remember the best score so far
            self.best_return = float(pd.read_csv(val_path)["return"].max())
        self._next_eval = every_steps

    @property
    def sim_steps(self) -> int:
        """Simulator steps so far (the unit of every_steps and of our logs)."""
        return self.num_timesteps // self.step_scale

    def _on_training_start(self) -> None:
        # When resuming, continue the schedule from the current step count.
        while self._next_eval <= self.sim_steps:
            self._next_eval += self.every_steps

    def _on_step(self) -> bool:
        if self.sim_steps >= self._next_eval:
            self._next_eval += self.every_steps
            self.evaluate()
        return True

    def evaluate(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        vec_norm = self.model.get_vec_normalize_env()
        policy = make_sb3_policy(self.design, self.model, vec_norm, name="PPO")
        episodes, _ = evaluate_policy(policy, self.seeds, self.env_kwargs)
        numeric = episodes.drop(columns=["policy", "seed"]).mean(numeric_only=True)
        row = {"timesteps": self.sim_steps, **numeric.to_dict()}
        row["eval_seconds"] = round(time.perf_counter() - t0, 2)
        is_best = row["return"] > self.best_return
        row["is_best"] = bool(is_best)
        if is_best:
            self.best_return = row["return"]
            self.model.save(self.run_dir / "best_model.zip")
            if vec_norm is not None:
                vec_norm.save(str(self.run_dir / "best_vecnormalize.pkl"))
        self.csv.log(row)
        if self.wandb_run is not None:
            metrics = {f"val/{k}": v for k, v in row.items()}
            self.wandb_run.log({"timesteps": self.sim_steps, **metrics})
        if self.verbose:
            print(f"[val] step {self.sim_steps}: return {row['return']:.2f} best={is_best}")
        return row
