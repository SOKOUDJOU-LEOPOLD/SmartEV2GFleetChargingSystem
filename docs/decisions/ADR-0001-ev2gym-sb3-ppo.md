# ADR-0001: EV2Gym + Stable-Baselines3 PPO as the base stack

- **Date:** 2026-11-17
- **Status:** Accepted
- **Author(s):** Leopold Sokoudjou Gatsing, Yeshas Krishna

## Context
We need a realistic V2G fleet simulator and a reliable RL implementation, so our time goes into the research question (hidden early departures), not into building infrastructure.

## Decision
- Simulator: **EV2Gym** [R1] (`pip install ev2gym`, https://github.com/StavrosOrf/EV2Gym), V2G profit-maximization setting, 20 chargers, one transformer.
- RL library: **Stable-Baselines3** [R13] with **PPO** [R8]; **sb3-contrib RecurrentPPO** [R17] for the stretch goal.
- Experiment tracking: local CSV/TensorBoard (source of truth) + **Weights & Biases** (shared dashboard).
- Python 3.12 virtual environment (`.venv`) inside the repo.

## Why (and why not the alternatives)
- EV2Gym: Gymnasium-compatible, has V2G, real arrival/price data, AFAP and oracle baselines built in. ACN-Sim / SustainGym lack V2G; Chargym uses identical randomized EVs [R1].
- SB3 PPO: well-tested, standard continuous-control baseline, used in the EV2Gym benchmark; writing PPO ourselves would add bug risk without research value.
- PPO over SAC/TD3: clipped updates are robust to the rare large departure penalties; on-policy RecurrentPPO exists for the stretch goal. (Off-policy methods earned more profit in the EV2Gym benchmark — a known trade-off, noted in the report.)
- venv over conda: conda is not installed on the lab machine; venv + pinned `requirements.txt` is enough and works everywhere.

## Consequences
- We depend on EV2Gym internals for forcing early departures → documented in ADR-0002 (Phase 1).
- Training is mostly CPU-bound; a CPU vs GPU timing test in Phase 1 decides the device.
