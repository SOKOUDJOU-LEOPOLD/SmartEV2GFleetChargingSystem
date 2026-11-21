# ADR-0005: One central agent with per-charger credit ("credit PPO")

- **Date:** 2026-11-20
- **Status:** Implemented and tested; **not adopted as the main design** (2026-11-20, see *Outcome*). Supersedes the *policy design* of ADR-0004; ADR-0004's per-charger reward split is kept and reused.
- **Author(s):** Leopold Sokoudjou Gatsing

## Context
Two designs have been tried with our squared-shortfall reward (no early departures yet, λ_dep = 50, λ_over = 1; 50 test days; EXPERIMENT_LOG Phase 2):

| Design | Test return | Main problem |
|---|---|---|
| AFAP (reference) | −46.8 | — |
| **Site agent**: one network, all 20 actions, one site reward (best of 6 runs, up to 10M steps) | −59.3 … −79.5 | **Credit**: one number for 20 decisions → slow, plateaus below AFAP |
| **Shared per-charger policy** (ADR-0004), 3M steps | −188.1 (−22.6 without the overload term) | **Coordination**: chargers decide independently → they overload the transformer together (17/50 days, up to 2,369 penalty in a day) |

The shared policy showed that per-charger credit works (drivers much better served: mean SoC at departure 0.86, departure penalty 23 vs 55+), but independent decisions cannot coordinate the shared transformer. Alternatives considered for that (power limiter, "fair-share" features) either take overload avoidance away from the agent or hard-code an equal split that breaks with urgent cars, solar or building loads. We want an agent that **learns** to avoid overload and can **flexibly** coordinate.

## Decision
Keep **one central agent** that sees the whole site and chooses all 20 actions together (as the site agent), but **train it with per-charger credit**:

1. **Actor (unchanged idea):** one network, site observation in, 20 action means out (diagonal Gaussian: one independent normal per charger, given the observation).
2. **Critic with 20 outputs:** `V_i(s)` = expected future reward *of charger i* (its own profit − its car's shortfall penalty − its overload share), instead of one value for the whole site.
3. **Per-charger rewards:** the existing split from ADR-0004 (`charger_terms`), which adds up exactly to the site reward.
4. **Per-charger advantages:** GAE [R12] computed separately for each charger: `A_i = GAE(r_i, V_i)`.
5. **Per-charger PPO loss:** the probability ratio of action *i* is clipped and multiplied by `A_i` (its own advantage), and the losses are summed over chargers [R8]. Value loss = mean over chargers of `(V_i − R_i)²`. Entropy as in PPO.

So each action is judged by **its own charger's outcome** (clear credit), while one brain decides all actions **together** (coordination: e.g. "3 cars need charging, the transformer allows 2½ → charge two fully, one at 70 %"). Overload stays a learned penalty (no limiter).

**Overload weight:** with λ_over = 1, one 32 kW overload step costs 1,024 while a day's profit is ~±40, which made learning unstable. New default **λ_over = 0.1** (≈ 10 kW over costs about as much as one car leaving 45 % short, ~10 points); both λ are calibrated in Phase 4.

**Reward scaling:** SB3's `VecNormalize` can only rescale one reward per environment, so per-charger rewards are rescaled by a fixed `reward_scale` from the config instead (observations are still normalized by VecNormalize).

## Why it is sound (and its approximation)
- Without the split (one reward), this *is* standard PPO. With the split, the sum of the per-charger advantages equals the standard site advantage (GAE is linear in rewards and values) — a unit test checks this.
- Using `A_i` for action *i* assumes action *i* mainly affects charger *i*'s reward. That is exactly true for profit and for its own car's departure; for overload, the penalty is shared by contribution, so the charger that pushed more pays more. The remaining coupling (charger *i* could cause another charger to share an overload penalty) is a small, documented bias accepted for much lower variance. This is the same kind of trade-off as reward decomposition in cooperative multi-agent RL [R18].

## Why (and why not the alternatives)
- **Central agent + per-charger credit (chosen):** coordination *and* clear credit; learns overload avoidance itself; flexible — solar/building loads or new information are simply added to the site observation (EV2Gym already has `V2G_profit_max_loads`); keeps the original single-agent framing of the proposal.
- **Site agent, standard PPO:** tried extensively; credit problem.
- **Shared per-charger policy (ADR-0004):** good credit, no coordination → overload.
- **Power limiter / safety layer:** safe but the agent no longer learns to avoid overload (requirement).
- **"Fair share" features for the shared policy:** hand-made equal split; inflexible for urgent cars, solar, loads.
- **Attention over chargers / Lagrangian λ:** more complex; possible later extensions.

## Implementation (branch `leopold/feat/p2-central-credit-ppo`)
- `agents/credit_ppo.py`: `CreditPPO` (subclass of SB3 PPO: rollout collection with vector rewards, per-charger GAE, per-charger loss), `CreditRolloutBuffer`, `CreditActorCriticPolicy` (20-output critic, per-dimension log-probabilities).
- `envs/charger_env.py`: site observation + vector reward mode (reuses the per-charger reward split and episode info).
- `agent_design: central_credit` in experiment configs; evaluation uses the normal site policy adapter (same metrics as all agents).
- Tests: per-charger GAE sums to the site GAE; per-dimension log-probs sum to SB3's joint log-prob; actions/rewards/values have matching shapes; end-to-end tiny run; save/load.

## Consequences
- More custom code than plain SB3 PPO (≈ 300 lines + tests) — kept small, commented, and tested against SB3's own behaviour.
- The paper's method section describes "PPO with per-charger advantage decomposition" (one central policy); the agents (RiskAware / Trusting / Oracle) all use it.
- RecurrentPPO (Phase 7) would need the same changes in sb3-contrib's recurrent version; revisit then.

## Outcome (2026-11-20)
Built as planned (`agents/credit_ppo.py`, `agent_design: central_credit`); the two exactness tests pass. Results (3M steps, λ_dep 50, λ_over 0.1, 50 test days; EXPERIMENT_LOG):

| | Without laxity | With laxity observation (`V2G_profit_max_laxity`) |
|---|---|---|
| Site agent (standard PPO) | −68.7 | **−33.4** |
| Central agent + per-charger credit | −86.3 | −35.6 |
| AFAP | −46.0 | −46.0 |

The main obstacle was missing information, not credit assignment: the observation showed each car's SoC but not its battery size, so the agent could not tell how long a car needs to charge. With the laxity observation (time still needed at full power, spare time until the announced departure) both designs beat AFAP, and per-charger credit adds nothing (and still overloads on some days).

**Decision:** the main agent is the **site agent (standard SB3 PPO) with the laxity observation** (`configs/experiments/p2_site_laxity_ppo.yaml`). CreditPPO stays in the code base, tested and documented, as an alternative design (one seed so far; revisit if the early-departure experiments show credit problems again).
