"""Launch a grid of training runs (+ their test evaluation), detached and resumable.

A sweep file (configs/sweeps/*.yaml) lists the base config, the seeds and the grid:

    name: p4-main-grid
    config: configs/experiments/p4_main.yaml
    seeds: [0, 1, 2]
    max_parallel: 10                 # runs at the same time (each uses ~17 CPU processes)
    evaluate: {p_early: [0, 0.05, 0.1, 0.2], trajectories: true}
    grid:                            # every combination of one option per dimension
      agent:
        - {label: trusting, set: {agent: PPO-Trusting, env.departures: null}}
        - {label: riskaware-p0.1, set: {agent: PPO-RiskAware, env.departures: {p_early: 0.1}}}
      lambda:
        - {label: lam500, set: {env.reward_kwargs.lambda_dep: 500}}

Each job = one combination × one seed: train (scripts/train.py logic, variant =
"trusting-lam500") then `evaluate_model.py` on the listed test worlds.

Usage (the driver itself must survive a closed session, CODING_RULES §7):
    setsid nohup python scripts/sweep.py configs/sweeps/<file>.yaml > results/logs/<name>.log 2>&1 &
    python scripts/sweep.py configs/sweeps/<file>.yaml --dry-run   # show what would run

Safe to start again at any time: finished jobs are skipped, jobs whose process is
still alive are waited for, interrupted runs resume from their last checkpoint,
and runs that finished training but not testing are only tested.
Status: results/sweeps/<name>/status.csv (one row per job, updated every poll).
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from smartev2g.envs.make_env import REPO_ROOT, resolve_path
from smartev2g.utils.run_info import RESULTS_DIR

POLL_SECONDS = 30
MAX_ATTEMPTS = 2  # launches per job before it is marked "failed"


def load_sweep(path: str | Path) -> dict:
    spec = yaml.safe_load(resolve_path(path).read_text())
    for key in ("name", "config", "seeds", "grid"):
        if key not in spec:
            raise ValueError(f"sweep file needs '{key}'")
    spec.setdefault("max_parallel", 4)
    spec.setdefault("evaluate", None)
    return spec


def expand(spec: dict) -> list[dict]:
    """All jobs: one per grid combination and seed (in a stable order)."""
    dims = list(spec["grid"].values())
    jobs = []
    for combo in itertools.product(*dims):
        variant = "-".join(option["label"] for option in combo)
        settings: dict = {}
        for option in combo:
            settings.update(option.get("set") or {})
        for seed in spec["seeds"]:
            jobs.append({"variant": variant, "seed": int(seed), "settings": settings})
    names = [(j["variant"], j["seed"]) for j in jobs]
    if len(set(names)) != len(names):
        raise ValueError("two grid combinations have the same label")
    return jobs


def run_name(spec: dict, job: dict) -> str:
    """Folder suffix train.py uses: <config name>-<variant>_s<seed>."""
    cfg = yaml.safe_load(resolve_path(spec["config"]).read_text())
    return f"{cfg['name']}-{job['variant']}_s{job['seed']}"


def find_run(results_dir: Path, name: str) -> Path | None:
    """Most recent results folder of this job (folders start with a timestamp)."""
    runs = sorted(p for p in results_dir.glob(f"*_{name}") if p.is_dir())
    return runs[-1] if runs else None


# Jobs started by THIS driver: name -> process handle. A finished child stays a
# "zombie" (still answers os.kill) until its parent collects it with poll().
_CHILDREN: dict[str, subprocess.Popen] = {}


def pid_alive(pid_file: Path) -> bool:
    if not pid_file.exists():
        return False
    child = _CHILDREN.get(pid_file.stem)
    if child is not None:
        return child.poll() is None  # also collects a finished child
    try:
        os.kill(int(pid_file.read_text()), 0)  # signal 0 = "does it exist?"
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def job_state(
    spec: dict, job: dict, results_dir: Path, state_dir: Path | None
) -> tuple[str, Path | None]:
    """done | running | evaluate | resume | train (what the job still needs).

    state_dir=None skips the "still running?" check (used by the job itself).
    """
    name = run_name(spec, job)
    if state_dir is not None and pid_alive(state_dir / f"{name}.pid"):
        return "running", None
    run = find_run(results_dir, name)
    if run is None:
        return "train", None
    if (run / "run_info.json").exists():
        if spec["evaluate"] and not (run / "eval" / "test_summary.csv").exists():
            return "evaluate", run
        return "done", run
    if any((run / "checkpoints").glob("ppo_*_steps.zip")):
        return "resume", run
    return "train", None  # died before its first checkpoint: start again


def launch(
    sweep_file: str, job: dict, spec: dict, state_dir: Path, log_dir: Path, results_dir: Path
) -> None:
    """Start one job as its own detached process (setsid), with a pid file."""
    name = run_name(spec, job)
    log = open(log_dir / f"{name}.log", "a")  # noqa: SIM115 (kept open by the child)
    cmd = [sys.executable, __file__, sweep_file, "--job", job["variant"], str(job["seed"])]
    cmd += ["--results-dir", str(results_dir)]
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=REPO_ROOT
    )
    (state_dir / f"{name}.pid").write_text(str(proc.pid))
    _CHILDREN[name] = proc


def run_job(sweep_file: str, variant: str, seed: int, results_dir: Path) -> None:
    """Inside a job process: train (or resume), then evaluate."""
    from smartev2g.agents.train import train

    spec = load_sweep(sweep_file)
    job = next(j for j in expand(spec) if j["variant"] == variant and j["seed"] == seed)
    state, run = job_state(spec, job, results_dir, None)
    if state == "train":
        run = train(
            spec["config"], seed, results_dir=results_dir, settings=job["settings"], variant=variant
        )
    elif state == "resume":
        run = train(None, seed, results_dir=results_dir, resume_dir=run)
    if spec["evaluate"] and run is not None:
        ev = spec["evaluate"]
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "evaluate_model.py"), "--run", str(run)]
        cmd += ["--p-early", *[str(p) for p in ev.get("p_early", [0.0, 0.05, 0.1, 0.2])]]
        if ev.get("trajectories"):
            cmd.append("--trajectories")
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def prepare_teacher_data(spec: dict, jobs: list[dict]) -> None:
    """Collect every warm-start teacher dataset once, BEFORE jobs start in parallel
    (otherwise several jobs would collect the same days at the same time)."""
    from smartev2g.agents.imitation import collect_expert_data
    from smartev2g.agents.train import apply_settings, env_kwargs, load_config

    seen = set()
    for job in jobs:
        cfg = load_config(spec["config"])
        apply_settings(cfg, job["settings"])
        pre = cfg.get("pretrain")
        if not pre:
            continue
        key = yaml.safe_dump(env_kwargs(cfg), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        collect_expert_data(env_kwargs(cfg), pre["days"], pre["first_day"], pre.get("workers", 16))


def drive(sweep_file: str, results_dir: Path, dry_run: bool) -> None:
    spec = load_sweep(sweep_file)
    jobs = expand(spec)
    state_dir = results_dir / "sweeps" / spec["name"]
    log_dir = results_dir / "logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        for job in jobs:
            state, run = job_state(spec, job, results_dir, state_dir)
            print(f"{state:9s} {run_name(spec, job)}" + (f"  ({run.name})" if run else ""))
        return
    prepare_teacher_data(spec, jobs)
    attempts: dict[str, int] = {}  # launches per job in THIS driver (a job may fail)
    while True:
        states = []
        for job in jobs:
            state, run = job_state(spec, job, results_dir, state_dir)
            if (
                state in ("train", "resume", "evaluate")
                and attempts.get(run_name(spec, job), 0) >= MAX_ATTEMPTS
            ):
                state = "failed"  # see results/logs/<job>.log; restart the driver to retry
            states.append((job, state, run))
        running = sum(s == "running" for _, s, _ in states)
        for job, state, _ in states:
            if state in ("train", "resume", "evaluate") and running < spec["max_parallel"]:
                launch(sweep_file, job, spec, state_dir, log_dir, results_dir)
                attempts[run_name(spec, job)] = attempts.get(run_name(spec, job), 0) + 1
                running += 1
        rows = [
            {"job": run_name(spec, j), "state": s, "run": r.name if r else ""} for j, s, r in states
        ]
        pd.DataFrame(rows).to_csv(state_dir / "status.csv", index=False)
        if all(s in ("done", "failed") for _, s, _ in states):
            failed = [run_name(spec, j) for j, s, _ in states if s == "failed"]
            print(
                f"Sweep {spec['name']}: {len(jobs) - len(failed)} done, failed: {failed}",
                flush=True,
            )
            return
        time.sleep(POLL_SECONDS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sweep", help="sweep file (configs/sweeps/*.yaml)")
    p.add_argument("--dry-run", action="store_true", help="only show each job's state")
    p.add_argument("--job", nargs=2, metavar=("VARIANT", "SEED"), help=argparse.SUPPRESS)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args()
    if args.job:
        run_job(args.sweep, args.job[0], int(args.job[1]), args.results_dir)
    else:
        drive(args.sweep, args.results_dir, args.dry_run)


if __name__ == "__main__":
    main()
