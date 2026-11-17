# Phase 5 — Analysis & Deliverable 1 Report

**Goal:** a clear, seed-robust answer to the main hypothesis.

**Status:** ⚪ Not started · **Milestone tag:** `v1.0-deliverable1`

## Branch
`<name>/docs/p5-results-analysis`

## Tasks
- [ ] Aggregate all runs: mean ± 95% CI over seeds (bootstrap; consider IQM — Agarwal et al. 2021)
- [ ] **Profit vs. shortfall trade-off curves** (Pareto front) per agent across λ values
- [ ] **Train/test heatmaps**: rows = training p_early, columns = test p_early, value = shortfall / profit
- [ ] **Hedging plots**: average SoC margin and V2G discharge vs. time-to-announced-departure, PPO-RiskAware vs PPO-Trusting
- [ ] **Gap closed vs. oracle**: how much of the PPO-Trusting → PPO-Oracle gap PPO-RiskAware closes
- [ ] Learning & loss curves for representative runs
- [ ] Results tables (proposal §11 metrics)
- [ ] Write-up in `reports/deliverable1/` (figures + summary CSV committed)

## GATE to the stretch goal
- [ ] Hypothesis answered clearly (positive or negative), robust across seeds and λ
- [ ] Both teammates agree results are solid

If not: iterate (Δ distribution, λ range, training length, hyperparameters) and re-run Phase 4.
