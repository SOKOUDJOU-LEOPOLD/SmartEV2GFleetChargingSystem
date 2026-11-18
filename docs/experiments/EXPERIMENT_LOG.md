# Experiment Log

One line per run (training or evaluation). Needed for the paper's experimental-setup and compute sections.
Full details of each run are in `results/<run_id>/run_info.json` and in wandb.

| Date | run_id | Who | Phase | Agent | Config | Seed | λ_dep | train p_early | Git commit | Hardware | Wall-clock | wandb link | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-17 | 20261117-203019_inspect-env | Leopold | 1 | AFAP | configs/env/v2g_profitmax_20port.yaml | 0–29 | — | 0 | be03327 | AMD EPYC 9554 (256 threads), CPU only | 60 s | — (wandb from Phase 3) | Env inspection: 14.9 cars/day, stay 7.4 h, arrival SoC 0.74, 0.9 % (4 cars) below target, all ≥ 0.99 SoC. Numbers in ADR-0002. |
