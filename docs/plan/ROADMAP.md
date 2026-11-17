# Project Roadmap — Risk-Aware RL for EV Charging Under Unexpected Departures

> **Read this first** (after [`CODING_RULES.md`](../../CODING_RULES.md)).
> This is the master plan. Each phase has its own file in this folder with the detailed task list, branch names, exit criteria and status.

## 1. What we are trying to answer

**Research question:** Can a model-free RL agent optimize V2G fleet scheduling under partial observability, implicitly managing the risk of unannounced early departures, and what economic or grid-performance trade-offs emerge from this robust policy?

In plain words: a charging site has 20 chargers. Every car says when it will leave. Some cars secretly leave **earlier** than they said. Can an AI controller (trained with PPO) learn by itself to keep a "safety reserve" of charge in the cars, and how much money / grid flexibility does that safety cost?

Source documents: [proposal](../proposal/projpropsal_updated.md), [literature review](../proposal/Literature_Review.md).

## 2. The agents we compare (names used everywhere: code, configs, figures)

| Name | Trained with hidden early departures? | Sees true departure time? | Role |
|---|---|---|---|
| **PPO-RiskAware** | Yes (`p_early > 0`) | No | **OUR agent** — the hypothesis is about this one |
| PPO-Trusting | No (`p_early = 0`) | No | Control: same agent, but trusts announced times (= "Baseline PPO" in the proposal) |
| PPO-Oracle | Yes | Yes (privileged info) | Upper bound: best possible with perfect information |
| AFAP | — (fixed rule, no learning) | — | Heuristic: Charge As Fast As Possible |

All agents are **tested in the same environments** (with and without hidden early departures).

**Main hypothesis:** PPO-RiskAware keeps a larger energy reserve and has fewer / smaller charge shortfalls at unexpected departures than PPO-Trusting, at the cost of some profit or V2G flexibility.

## 3. Reward (same for every learning agent)

```
R_t = R_economic,t  −  λ_overload · max(0, P_total,t − P_max)²  −  λ_departure · Σ_{EVs leaving at t} max(0, SoC*_i − SoC_i,t)²
```

- **R_economic**: money earned by selling energy (V2G) minus money paid to charge.
- **Overload penalty**: punishes going above the transformer's power limit.
- **Squared-shortfall penalty**: when a car leaves, `shortfall = target charge − actual charge` (0 if target reached). Squaring makes big misses hurt much more than small ones (0.1 short → 0.01; 0.3 short → 0.09). Applied identically to on-time and early departures. Received **only after** the car has left — the agent is never warned.

## 4. Phases at a glance

| Phase | Name | Goal | Milestone tag | Detail |
|---|---|---|---|---|
| 0 | Repo foundation | Organized repo, env, rules, CI | — | [phase-0](phase-0-foundation.md) |
| 1 | EV2Gym exploration & sanity PPO | Pipeline works end-to-end | `v0.1-pipeline` | [phase-1](phase-1-env-sanity-ppo.md) |
| 2 | Reward + UnexpectedDepartureWrapper + Oracle | The core contribution, fully tested | `v0.2-wrapper` | [phase-2](phase-2-wrapper-reward.md) |
| 3 | Training, logging & evaluation infrastructure | Every metric logged (CSV + wandb) | — | [phase-3](phase-3-train-eval-infra.md) |
| 4 | Main experiments | Full grid: agents × λ × seeds × test worlds | — | [phase-4](phase-4-main-experiments.md) |
| 5 | Analysis & Deliverable 1 report | Solid, seed-robust answer | `v1.0-deliverable1` | [phase-5](phase-5-analysis-report.md) |
| — | **GATE** | Deliverable 1 must have clear results before going further | | |
| 6 | Stretch: ACN-Data departure model | Realistic early-departure distribution | — | [phase-6](phase-6-acn-data.md) |
| 7 | Stretch: memory (PPO vs RecurrentPPO) | Does memory help infer hidden risk? | `v2.0-stretch` | [phase-7](phase-7-recurrent-ppo.md) |

Supporting documents:
- [Metrics & logging spec](METRICS_AND_LOGGING.md) — what every run must save (for the paper).
- [RL concepts glossary](../concepts/RL_CONCEPTS.md) — plain-language explanations + references.
- [Experiment log](../experiments/EXPERIMENT_LOG.md) — one line per run.
- [Decision records](../decisions/) — short "why we chose X" notes.

## 5. Deliverable 1 in plain words

1. Build the **UnexpectedDepartureWrapper**: an add-on to EV2Gym that secretly makes some cars leave earlier than they announced.
2. Train **PPO-RiskAware** in that tricky world and **PPO-Trusting** in the normal world.
3. Add reference points **PPO-Oracle** and **AFAP**.
4. Repeat training with several penalty strengths **λ_departure** (results depend on it — Chen & Shi 2022; Naghdizadegan Jahromi et al. 2026).
5. Repeat every training with **≥ 5 random seeds** so results are not luck.
6. Test every agent in every world (`p_early ∈ {0, 0.05, 0.10, 0.20}`) — the **train/test grid**.
7. **Hedging analysis**: does the agent keep more charge / sell less to the grid as a car's announced departure gets close?

## 6. The gate before the stretch goal

We move to Phase 6 **only when** Deliverable 1 has:
- a clear answer (positive or negative) to the main hypothesis,
- that holds across ≥ 5 seeds (mean ± 95% CI) and across λ values,
- with figures + write-up in `reports/deliverable1/`.

If results are flat or noisy, we iterate inside Phases 2–5 (early-departure distribution Δ, λ range, training length, PPO hyperparameters) before moving on.

## 7. Stretch goal in plain words

1. **Realistic departures (Phase 6):** use Caltech's ACN-Data (announced vs. real departure times) to build a data-driven early-departure model instead of a made-up one.
2. **Memory (Phase 7):** compare normal PPO with **RecurrentPPO** (LSTM = a network with memory) in two conditions:
   - **A — time-of-day risk** `p_early(t)`: the clock is already in the observation, so memory should *not* help (control).
   - **B — hidden daily risk level** (some days have many early leavers, never told to the agent but visible in history): memory *should* help.

## 8. Team & workflow

- Team: Yeshas Krishna, Leopold Sokoudjou Gatsing.
- Branch per logical task group, named `<firstname>/<type>/<phase>-<short-name>`; PR into `main`. See [`CONTRIBUTING.md`](../../CONTRIBUTING.md).
- Status of each task is tracked with checkboxes in the phase files — update them in the same PR that does the work.
