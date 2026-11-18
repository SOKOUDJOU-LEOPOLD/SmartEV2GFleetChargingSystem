# ADR-0003: Train PPO on the CPU with 1 PyTorch thread

- **Date:** 2026-11-17
- **Status:** Accepted
- **Author(s):** Leopold Sokoudjou Gatsing

## Context
The server has 4× NVIDIA H100 GPUs and an AMD EPYC 9554 CPU (256 threads). Should PPO training use the GPU? Our PPO network is tiny (2 × 64 neurons) and the simulator (EV2Gym) runs only on the CPU, so most time is spent simulating, not computing gradients.

## Measurement
`scripts/timing_test.py`: the same training (config `configs/experiments/p1_sanity_ppo.yaml`, seed 0, 16 simulator copies, 100,352 steps, no validation/test) with 5 settings. Run `20261117-220030_timing-test`, commit `7b91365`, logged in `docs/experiments/EXPERIMENT_LOG.md`.

| Device | PyTorch threads | Steps / second | Time |
|---|---|---|---|
| **CPU** | **1** | **3,709** | 27.1 s |
| CPU | 16 | 3,600 | 27.9 s |
| CPU | 4 | 3,420 | 29.3 s |
| CPU | 128 (PyTorch default) | 3,082 | 32.6 s |
| GPU (H100 NVL) | 4 | 2,807 | 35.8 s |

## Decision
Train on the **CPU with `torch_threads: 1`** (`configs/experiments/*.yaml`, `training` section).

## Why (and why not the alternatives)
- **CPU, 1 thread:** fastest, and it uses the fewest cores, so many seeds/settings can run side by side (Phase 4 grid) without competing for cores. The differences between 1, 4 and 16 threads are small (one run each, ~8 %), so the main reason to pick 1 is to keep cores free for parallel runs.
- **GPU:** ~25 % slower. Copying each small batch to the GPU and back costs more than the GPU saves on such a small network.
- **PyTorch default (all cores):** slower and wastes CPU (in an early test a 27 s run used 6.5 min of CPU time).

## Consequences
- Parallelism comes from running several training processes at once (one per seed/setting), each with 16 simulator copies.
- Re-test before Phase 7: a recurrent (LSTM) policy is heavier and may benefit from the GPU.
