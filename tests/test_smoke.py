"""Smoke tests: check the environment is installed correctly.

They do not train anything; they only prove that a fresh clone + `pip install`
gives a working setup (Phase 0 exit criterion).
"""

import importlib

import pytest

# Every library the project needs at runtime.
REQUIRED_MODULES = [
    "smartev2g",
    "ev2gym",
    "gymnasium",
    "stable_baselines3",
    "sb3_contrib",
    "torch",
    "numpy",
    "pandas",
    "yaml",
    "wandb",
]


@pytest.mark.parametrize("name", REQUIRED_MODULES)
def test_import(name: str) -> None:
    """Each required library can be imported."""
    importlib.import_module(name)


def test_ev2gym_env_imports() -> None:
    """EV2Gym's main classes load (catches missing undeclared deps like pandapower/numba)."""
    from ev2gym.baselines.heuristics import ChargeAsFastAsPossible  # noqa: F401
    from ev2gym.models.ev2gym_env import EV2Gym  # noqa: F401
    from ev2gym.rl_agent.reward import profit_maximization  # noqa: F401
    from ev2gym.rl_agent.state import V2G_profit_max  # noqa: F401
