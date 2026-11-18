"""Tests for run provenance (run_info.json) and the inspect_env script outputs."""

import json
import subprocess
import sys

import pandas as pd

from smartev2g.envs.make_env import DEFAULT_ENV_CONFIG, REPO_ROOT
from smartev2g.utils.run_info import RunInfo, make_run_dir

SCRIPT = REPO_ROOT / "scripts" / "inspect_env.py"


def test_run_info_has_required_fields(tmp_path):
    """CODING_RULES §5: config, seed, git commit, versions, hardware, timing."""
    run_dir = make_run_dir("unit-test", root=tmp_path)
    path = RunInfo(run_dir, "test", DEFAULT_ENV_CONFIG, seeds=[0]).finish({"x": 1})
    info = json.loads(path.read_text())
    for key in ["config", "seeds", "git", "packages", "hardware", "start_time", "end_time"]:
        assert key in info
    assert info["config"]["number_of_charging_stations"] == 20
    assert info["results"] == {"x": 1}
    assert info["wall_clock_s"] >= 0


def test_inspect_env_rejects_nonpositive_episodes(tmp_path):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--episodes", "0", "--results-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
    assert "--episodes must be a positive integer" in out.stderr
    assert not any(tmp_path.iterdir())  # no empty run folder left behind


def test_inspect_env_saves_outputs(tmp_path):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--episodes", "2", "--results-dir", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    (run_dir,) = tmp_path.iterdir()
    evs = pd.read_csv(run_dir / "evs.csv")
    episodes = pd.read_csv(run_dir / "episodes.csv")
    info = json.loads((run_dir / "run_info.json").read_text())
    assert set(evs["seed"]) == {0, 1}
    assert len(episodes) == 2
    assert episodes["n_evs"].sum() == len(evs)
    assert info["seeds"] == [0, 1]
    assert "pct_cars_below_target_afap" in info["results"]
