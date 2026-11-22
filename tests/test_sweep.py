"""Tests for scripts/sweep.py (grid launcher, resumable)."""

import importlib.util
import os

import pandas as pd
import pytest
import yaml

from smartev2g.envs.make_env import REPO_ROOT, resolve_path

spec_ = importlib.util.spec_from_file_location("sweep", REPO_ROOT / "scripts" / "sweep.py")
sweep = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(sweep)


def tiny_setup(tmp_path, evaluate=True):
    """A tiny experiment config + a 2-option × 1-seed sweep file in tmp_path."""
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text(
        yaml.safe_dump(
            {
                "validation": {"start": 1_000_000, "count": 1},
                "test": {"start": 2_000_000, "count": 1},
            }
        )
    )
    cfg = yaml.safe_load(resolve_path("configs/experiments/p4_main.yaml").read_text())
    cfg["name"] = "tiny"
    cfg["scenarios"] = str(scenarios)
    cfg["pretrain"] = None  # keep the test fast (warm start has its own tests)
    cfg["logging"]["wandb"] = "disabled"
    cfg["training"].update(total_timesteps=224, n_envs=1, eval_every_steps=10**9)
    cfg["ppo"].update(n_steps=224, batch_size=112, n_epochs=1)
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(cfg))
    spec = {
        "name": "tiny-sweep",
        "config": str(config),
        "seeds": [0],
        "max_parallel": 2,
        "evaluate": {"p_early": [0.0, 0.5]} if evaluate else None,
        "grid": {
            "agent": [
                {"label": "trusting", "set": {"agent": "PPO-Trusting", "env.departures": None}},
                {
                    "label": "riskaware",
                    "set": {"agent": "PPO-RiskAware", "env.departures": {"p_early": 0.1}},
                },
            ]
        },
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def test_grid_expands_to_every_combination_and_seed(tmp_path):
    spec = sweep.load_sweep(tiny_setup(tmp_path))
    spec["seeds"] = [0, 1, 2]
    spec["grid"]["lam"] = [{"label": "lam200"}, {"label": "lam500"}]
    jobs = sweep.expand(spec)
    assert len(jobs) == 2 * 2 * 3
    assert {j["variant"] for j in jobs} >= {"trusting-lam200", "riskaware-lam500"}
    spec["grid"]["lam"] = [{"label": "same"}, {"label": "same"}]
    with pytest.raises(ValueError):
        sweep.expand(spec)


def test_job_states(tmp_path):
    spec = sweep.load_sweep(tiny_setup(tmp_path))
    job = sweep.expand(spec)[0]
    results, state_dir = tmp_path / "results", tmp_path / "state"
    results.mkdir()
    state_dir.mkdir()
    name = sweep.run_name(spec, job)
    assert name == "tiny-trusting_s0"
    assert sweep.job_state(spec, job, results, state_dir)[0] == "train"
    run = results / f"20990101-000000_{name}"
    (run / "checkpoints").mkdir(parents=True)
    assert sweep.job_state(spec, job, results, state_dir)[0] == "train"  # no checkpoint yet
    (run / "checkpoints" / "ppo_100_steps.zip").write_text("x")
    assert sweep.job_state(spec, job, results, state_dir) == ("resume", run)
    (run / "run_info.json").write_text("{}")
    assert sweep.job_state(spec, job, results, state_dir)[0] == "evaluate"
    (run / "eval").mkdir()
    (run / "eval" / "test_summary.csv").write_text("x")
    assert sweep.job_state(spec, job, results, state_dir)[0] == "done"
    (state_dir / f"{name}.pid").write_text(str(os.getpid()))  # this process is alive
    assert sweep.job_state(spec, job, results, state_dir)[0] == "running"


def test_sweep_runs_trains_and_evaluates(tmp_path, monkeypatch):
    """End to end: two detached jobs train, get tested, and a restart skips them."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(sweep, "POLL_SECONDS", 2)
    path = tiny_setup(tmp_path)
    results = tmp_path / "results"
    sweep.drive(str(path), results, dry_run=False)
    status = pd.read_csv(results / "sweeps" / "tiny-sweep" / "status.csv")
    assert set(status["state"]) == {"done"}
    for name in ("tiny-trusting_s0", "tiny-riskaware_s0"):
        run = sweep.find_run(results, name)
        summary = pd.read_csv(run / "eval" / "test_summary.csv")
        assert list(summary["test_p_early"]) == [0.0, 0.5]
    risk = yaml.safe_load(
        (sweep.find_run(results, "tiny-riskaware_s0") / "config_resolved.yaml").read_text()
    )
    assert risk["agent"] == "PPO-RiskAware" and risk["env"]["departures"]["p_early"] == 0.1
    # Starting again launches nothing new.
    before = sorted(p.name for p in results.iterdir())
    sweep.drive(str(path), results, dry_run=False)
    assert sorted(p.name for p in results.iterdir()) == before
