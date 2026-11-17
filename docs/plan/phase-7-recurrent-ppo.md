# Phase 7 — Stretch: Does Memory Help? (PPO vs RecurrentPPO)

**Goal:** test whether a policy with memory (LSTM) can infer hidden departure risk from history.

**Status:** ⚪ Blocked by the Deliverable 1 gate · **Milestone tag:** `v2.0-stretch`

## Branches
| Branch | Content |
|---|---|
| `<name>/feat/p7-risk-conditions` | Time-of-day and hidden-daily-risk departure models |
| `<name>/feat/p7-recurrent-ppo` | RecurrentPPO training/eval support |
| `<name>/exp/p7-memory-grid` | Experiments |

## Tasks
- [ ] **Condition A — time-of-day risk** `p_early(t)` (from ACN-Data): clock is in the observation ⇒ memory should *not* help (control)
- [ ] **Condition B — hidden daily risk level**: each day secretly "low" or "high" early-departure rate; inferable from cars that disappeared early ⇒ memory *should* help
- [ ] sb3-contrib `RecurrentPPO("MlpLstmPolicy")`; evaluation must pass `lstm_states` and `episode_start` to `predict()`
- [ ] Cheap-memory control: PPO with frame stacking (last k observations)
- [ ] Tune LSTM size and sequence length (Ni et al. 2022: history length matters)
- [ ] Re-run CPU vs GPU timing test (LSTM is larger; GPU may win)
- [ ] Same seeds, metrics and logging as Deliverable 1
- [ ] Write-up section in `reports/stretch/`

## Exit criteria
- Clear answer on whether memory helps in Condition B but not in A.

## Optional later extensions
- Constrained RL (Lagrangian PPO) instead of a fixed λ (Chen & Shi 2022)
- Battery degradation in the reward
