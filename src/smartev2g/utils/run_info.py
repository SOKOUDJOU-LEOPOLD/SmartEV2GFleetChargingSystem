"""Record *how* a run was made, so every number in the paper can be traced back.

Each run gets its own folder `results/<run_id>/` and a `run_info.json` with:
the exact config used, the seeds, the git commit of the code, library versions,
the hardware, and start/end time + wall-clock duration.
Ref: [R14] (reproducibility: Henderson et al. 2018, "Deep RL that Matters").
See CODING_RULES.md §5 and docs/plan/METRICS_AND_LOGGING.md.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from smartev2g.envs.make_env import REPO_ROOT, resolve_path

RESULTS_DIR = REPO_ROOT / "results"

# Libraries whose versions can change results; recorded in every run_info.json.
TRACKED_PACKAGES = [
    "ev2gym",
    "gymnasium",
    "stable-baselines3",
    "sb3-contrib",
    "torch",
    "numpy",
    "pandas",
]


def make_run_id(name: str) -> str:
    """Unique, sortable run name, e.g. '20261117-153012_inspect-env'."""
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}"


def make_run_dir(run_id: str, root: Path = RESULTS_DIR) -> Path:
    """Create `results/<run_id>/` (fails if it already exists, so runs never overwrite)."""
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_info() -> dict[str, Any]:
    """Commit hash + whether there were uncommitted changes (a 'dirty' run is not reproducible)."""
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": _git("status", "--porcelain") not in ("", "unknown"),
    }


def package_versions() -> dict[str, str]:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def hardware_info() -> dict[str, Any]:
    """CPU model, core count and GPUs (needed for the compute section of the paper)."""
    cpu = platform.processor() or "unknown"
    try:  # Linux: the readable CPU name is in /proc/cpuinfo
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpus: list[str] = []
    try:
        import torch

        if torch.cuda.is_available():
            gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return {
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "gpus": gpus,
    }


class RunInfo:
    """Collects run details; call `finish()` at the end to write run_info.json.

    Usage:
        info = RunInfo(run_dir, script="inspect_env", config_path=cfg, seeds=[0, 1])
        ... run ...
        info.finish(results={"cars_per_episode_mean": 14.9})
    """

    def __init__(
        self,
        run_dir: Path,
        script: str,
        config_path: str | Path,
        seeds: list[int],
        args: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self._t0 = time.perf_counter()
        config_file = resolve_path(config_path)
        self.data: dict[str, Any] = {
            "run_id": self.run_dir.name,
            "script": script,
            "args": args or {},
            "config_path": str(config_path),
            # Full config content, so the run is still known if the file changes later.
            "config": yaml.safe_load(config_file.read_text()),
            "seeds": seeds,
            "git": git_info(),
            "packages": package_versions(),
            "hardware": hardware_info(),
            "start_time": datetime.now().isoformat(timespec="seconds"),
        }

    def finish(self, results: dict[str, Any] | None = None) -> Path:
        """Add end time, wall-clock and summary results; write run_info.json."""
        self.data["end_time"] = datetime.now().isoformat(timespec="seconds")
        self.data["wall_clock_s"] = round(time.perf_counter() - self._t0, 2)
        self.data["results"] = results or {}
        path = self.run_dir / "run_info.json"
        path.write_text(json.dumps(self.data, indent=2, default=str))
        return path
