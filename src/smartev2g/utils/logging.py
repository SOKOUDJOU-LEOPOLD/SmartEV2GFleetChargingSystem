"""Logging helpers: local CSV files (source of truth) + Weights & Biases (dashboard).

Local CSVs are always written, so nothing is lost if wandb is down.
See docs/plan/METRICS_AND_LOGGING.md and CODING_RULES.md §6.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from smartev2g.envs.make_env import REPO_ROOT


class CSVLogger:
    """Append one row (a dict) at a time to a CSV file; columns come from the first row.

    Writing row by row means the file is useful even if a long run crashes.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fields: list[str] | None = None
        if self.path.exists():  # resuming a run: keep the existing header
            with self.path.open() as f:
                self._fields = next(csv.reader(f), None)

    def log(self, row: dict[str, Any]) -> None:
        new_file = self._fields is None
        if new_file:
            self._fields = list(row.keys())
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def wandb_mode(config_mode: str) -> str:
    """WANDB_MODE env var (e.g. 'disabled' in CI/tests) wins over the config value."""
    return os.environ.get("WANDB_MODE", config_mode)


def init_wandb(
    cfg: dict[str, Any],
    run_id: str,
    run_dir: Path,
    group: str,
    tags: list[str],
):
    """Start a wandb run (or a no-op run if disabled). Returns the run object.

    The API key comes only from `wandb login` or WANDB_API_KEY in .env (never in code).
    """
    load_dotenv(REPO_ROOT / ".env")  # loads WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT
    import wandb

    log_cfg = cfg.get("logging", {})
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT") or log_cfg.get("project", "smartev2g"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=run_id,
        group=group,
        tags=tags,
        config=cfg,
        dir=str(run_dir),
        mode=wandb_mode(log_cfg.get("wandb", "online")),
        sync_tensorboard=True,  # streams SB3's TensorBoard losses to wandb
    )
    # Our own metrics use "timesteps" (environment steps so far) as the x-axis.
    run.define_metric("timesteps")
    for prefix in ("train_ep/*", "val/*"):
        run.define_metric(prefix, step_metric="timesteps")
    return run
