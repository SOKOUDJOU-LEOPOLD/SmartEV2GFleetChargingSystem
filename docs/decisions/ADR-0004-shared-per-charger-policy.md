# ADR-0004: One shared per-charger policy with per-charger rewards

- **Date:** 2026-11-20
- **Status:** Accepted (2026-11-20)
- **Author(s):** Leopold Sokoudjou Gatsing

## Context
With our squared-shortfall reward (Phase 2, no early departures yet), the standard set-up — **one PPO network that sees the whole site (62 numbers) and outputs all 20 charger actions, scored by one site-wide reward** — does not learn a good policy:

| Run (test, 50 days, λ_dep = 50) | Test return | Cars meeting target |
|---|---|---|
| AFAP (charge as fast as possible) | **−46.8** | 99.8 % |
| Random | −75.3 | 0.4 % |
| PPO, 1M steps | −90.0 | 8 % |
| PPO, 3M steps, best of 5 variants (bigger net, 64 envs, less initial noise, γ 0.995, longer) | −65.4 … −79.5 | 47–62 % |

Every run first gets worse (learns that selling energy pays immediately), then slowly recovers as it discovers the delayed departure penalty (EXPERIMENT_LOG, Phase 2). Two causes are likely, both about *credit assignment* [R11]:
1. **Which charger?** Each step gives one number for 20 simultaneous decisions; the agent cannot tell which charger's car caused a penalty.
2. **When?** The departure penalty arrives hours after the decisions that caused it.

This ADR addresses (1); (2) can be addressed separately (see "Possible extension").

## Decision
Control the site with **one small policy shared by all chargers** ("parameter sharing"), trained with the same PPO, and **split our site reward into per-charger rewards that add up exactly to the site reward**.

### How it works
- **Observation of charger *i*** (~25 numbers): shared site information — time step, the 20 upcoming prices, last step's total site power, number of cars plugged in — plus *its own* car: SoC, steps left until the **announced** departure, and a plugged-in flag. (The oracle variant shows the true departure instead; the early-departure wrapper is unchanged because it acts inside the simulator.)
- **Action of charger *i***: one number in [−1, 1] (charge +, V2G discharge −). The 20 actions are assembled into EV2Gym's action vector.
- **Same network for all chargers**: it learns one rule — "for a car like this, at these prices, do this" — applied 20 times per step.
- **Reward of charger *i*** for a step:
  `r_i = profit_i − λ_dep · shortfall_i² · [car at i left this step] − overload share_i`
  - `profit_i`: money charger *i* earned/paid this step (from EV2Gym's per-charger accounting).
  - departure term: only the charger whose car left pays for that car's shortfall.
  - overload `λ_over · Σ overload²` is caused jointly → split equally between the 20 chargers (first version; a contribution-weighted split is a possible refinement).
  - **Σ_i r_i = site reward R_t** of `rewards/shortfall_reward.py` → same overall objective. Enforced by a unit test.
- **Training**: a wrapper presents one EV2Gym site as 20 environments to Stable-Baselines3, so 16 simulators give PPO 320 "chargers" per step. PPO settings, logging, validation, test and wandb stay as they are. Empty chargers get reward 0 (their action has no effect).
- **Evaluation**: an adapter builds the 20 observations, calls the shared policy, and plays the day in the existing evaluation code, so all metrics stay comparable (site-level return = Σ_i r_i).

### Code (new branch `leopold/feat/p2-shared-charger-policy`)
1. `envs/charger_obs.py` — per-charger observation builder (+ names, like `observation_names`).
2. `rewards/shortfall_reward.py` — per-charger split of the existing terms.
3. `envs/charger_vec_env.py` — one site → 20 SB3 environments.
4. `eval/evaluate.py` — adapter for the shared policy.
5. Tests: per-charger rewards sum to the site reward; observation layout; actions reach the right charger; same seed ⇒ same episode.

## Why (and why not the alternatives)
- **Shared per-charger policy (chosen):** the penalty for a car lands on the charger that served it (clear credit); 20× more learning examples per simulated step; a smaller problem (≈25 inputs, 1 output); hedging becomes a per-car behaviour, which is exactly what our analysis measures. Parameter-shared PPO is a strong, well-tested approach for cooperating units [R18].
- **Keep one site-wide agent, tune more** — tried (5 variants, up to 3M steps, and a 10M-step combined run, see Consequences); improvement is slow and does not reach AFAP.
- **Same sub-network per charger but still one site reward** — easier, but the reward is still one number, so the credit problem remains.
- **Reward shaping only** (earlier penalty, potential-based so the best policy is unchanged) — addresses the *timing* problem, not *which charger*. Kept as a possible extension.
- **Fully independent agents per charger (different weights)** — 20× fewer examples per agent and no reason for chargers to behave differently (they are identical).

## Consequences
- **Coordination is indirect.** Each charger decides alone; it sees last step's site power, not the other chargers' current actions. The 100 kW transformer limit is learned through the shared overload penalty. Overload has been rare so far (AFAP: 0.14 kW summed over 50 test days); we will monitor it and refine the split if needed.
- **Report wording:** the agents (PPO-RiskAware / PPO-Trusting / PPO-Oracle) are still PPO with the same reward; each is "one policy shared by all chargers". State this in the method section and cite [R18].
- **Recurrent PPO (Phase 7)** works the same way: each charger has its own LSTM memory, sharing weights.
- **Possible extension:** potential-based shaping of the departure penalty (Ng et al., 1999) if learning is still slow.
- If the 10M-step combined run of the current design (started 2026-11-20) reaches AFAP, revisit this ADR before implementing.
