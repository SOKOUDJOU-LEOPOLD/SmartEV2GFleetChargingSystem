# ADR-0007: Frozen PPO setup for the main experiments (Phase 4)

- **Date:** 2026-11-22
- **Status:** Accepted
- **Author(s):** Leopold

## Context
Phase 4 trains many agents (λ_dep × training risk × ≥ 10 seeds). They must all use one fixed, documented setup, chosen BEFORE looking at Phase 4 results, so the comparison between PPO-Trusting and PPO-RiskAware is not tuned in anyone's favour. Phase 2 tested several setups (EXPERIMENT_LOG, 2026-11-21/22):
- plain PPO with SB3 defaults captured ~2 % of the gap between AFAP-target and the perfect-knowledge optimum; updates were too large (approx_kl 0.14–0.27) and it did not time its trades;
- calmer updates (setting B) and the day-ahead price view each gave +1–2 points;
- the imitation warm start made PPO trade properly (spread 3.9 c vs 1.4 c, positive money);
- runs were still improving at 3M steps, and picking the best model on 10 validation days was noisy (validation −4.6 vs test −12.4 for the same model).

## Decision
`configs/experiments/p4_main.yaml` is the frozen setup:

| Part | Choice |
|---|---|
| Agent | one PPO for the whole site (`site`, ADR-0005 outcome), net [256, 256] |
| Observation | `V2G_profit_max_laxity_dayahead`: laxity per car + every published day-ahead price of the next 24 h |
| Reward | squared shortfall, λ_over 0.1, 80 % target (ADR-0006); λ_dep from the grid |
| PPO | lr 1e-4, n_steps 448 × 16 envs, batch 2048, 5 epochs, γ 0.99, GAE λ 0.95, clip 0.2 ("setting B") |
| Warm start | clone Optimum-announced on 2,000 training days (60 epochs), critic pre-trained, start std 0.37 |
| Length | 6M simulator steps (~57 min on 17 CPU processes) |
| Model selection | best mean return on 30 validation days (`scenarios_val30.yaml`); 50 test days unchanged |

PPO-Trusting and PPO-RiskAware differ ONLY in the training world (`env.departures`), and use the same warm-start teacher, which trusts announced departures in both cases.

## Why (and why not the alternatives)
- **Setting B + warm start + day-ahead view:** the only combination that made PPO a real trader (~30 % of the gap at 6M steps); healthy updates (approx_kl ≈ 0.02).
- **No warm start:** simpler, but PPO then barely beats AFAP-target and learns no timing, so a hedging result would be about a weak agent.
- **Teacher that knows the real departures:** would hand PPO-RiskAware the answer; the hedging must be learned by RL.
- **Longer than 6M steps:** probably still helps a little, but doubles the cost of a grid of ~100 runs; 6M is the budget for all agents alike.
- **Tuning separately per agent:** would make the comparison unfair.

## Consequences
- Every Phase 4 result is "PPO warm-started from an expert planner"; the paper must say so.
- Changing any of these values needs a new config file and a new ADR (old results stay comparable to each other).
- Open for later: battery-wear cost per kWh (would stop the remaining charge/discharge flipping), longer training, the learning PPO-Oracle (Phase 4 grid).
