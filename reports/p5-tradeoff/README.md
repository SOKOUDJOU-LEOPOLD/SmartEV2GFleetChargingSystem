# Phase 5: money vs driver protection, and does longer training help?

**Runs:** 46, all finished, none failed (2026-11-23, sweeps `p5-tradeoff-curve` and `p5-long-training`, 67–75 min per 6M run, 133–143 min per 12M run).
**Setup:** the "real charging need" world of ADR-0008 (`configs/experiments/p4_arrival.yaml`, cars arrive at 80 % − the driver's energy from the data). Combined with the λ 500 / 1000 runs of [`reports/p4-arrival/`](../p4-arrival/README.md), PPO-Trusting and PPO-RiskAware p0.2 now each have 10 seeds at **four penalty weights λ_dep = 200, 500, 1000, 2000**.

**Question:** raising λ is the simple way to make PPO protect drivers better. Does training with hidden early departures (RiskAware) buy *more* protection per euro than that?

## Files
| File | Content |
|---|---|
| `tradeoff.csv` | per agent and λ (test world 20 % early): money per day and % of early leavers below 70 % (mean, 95 % CI over 10 seeds) + AFAP-target / optimum references |
| `dominance.csv` | every RiskAware point vs every Trusting point: differences, Welch t-test p, Holm-corrected over all 32 tests, `better_on_both` |
| `tradeoff.png` | the curves |
| `lam200/`, `lam2000/` | same files as `reports/p4-arrival/lam*/` (summary, paired, drivers, grid, hedging, learning) |
| `long-training/` | λ 1000, seeds 0–2: 6M vs 12M steps, same files |

```bash
python scripts/make_tradeoff.py \
  --agent "PPO-Trusting=results/*_p4-arrival-lam{lam}-trusting_s?" \
  --agent "RiskAware p0.2=results/*_p4-arrival-lam{lam}-riskaware-p0.2_s?" \
  --lambdas 200 500 1000 2000 --optimum "results/*_optimum-test-lamdep{lam}-arrivaltarget" \
  --test-p 0.2 --out reports/p5-tradeoff
G() { echo "$1=results/*_p4-arrival-$2_s?"; }   # λ 2000 (λ 200 the same)
python scripts/make_figures.py --group "$(G PPO-Trusting lam2000-trusting)" \
  --group "$(G 'RiskAware p0.2' lam2000-riskaware-p0.2)" \
  --optimum "results/*_optimum-test-lamdep2000-arrivaltarget" --out reports/p5-tradeoff/lam2000
python scripts/make_figures.py --group "Trusting 6M=results/*_p4-arrival-lam1000-trusting_s[012]" \
  --group "Trusting 12M=results/*_p4-arrival-lam1000-trusting-12M_s?" \
  --group "RiskAware 6M=results/*_p4-arrival-lam1000-riskaware-p0.2_s[012]" \
  --group "RiskAware 12M=results/*_p4-arrival-lam1000-riskaware-p0.2-12M_s?" \
  --optimum "results/*_optimum-test-lamdep1000-arrivaltarget" --out reports/p5-tradeoff/long-training
```

## Results

### 1. The money-vs-safety curve (`tradeoff.png`, `tradeoff.csv`)
Test world: 20 % of drivers leave early without warning. Mean over 10 seeds ± 95 % CI.

| λ | PPO-Trusting: money €/day | Trusting: early leavers below 70 % | RiskAware p0.2: money €/day | RiskAware: early leavers below 70 % |
|---|---|---|---|---|
| 200 | −24.2 ± 1.1 | 32.4 ± 2.9 % | −26.5 ± 0.5 | 19.3 ± 1.6 % |
| 500 | −33.5 ± 0.6 | 25.7 ± 2.1 % | −35.3 ± 0.4 | 11.8 ± 1.0 % |
| 1000 | −38.3 ± 0.6 | 20.6 ± 1.9 % | −42.1 ± 1.2 | 7.7 ± 1.3 % |
| 2000 | −44.8 ± 1.5 | 17.9 ± 3.7 % | −50.5 ± 1.2 | 4.8 ± 0.9 % |

References (one deterministic run): AFAP-target −47.9 €/day, 0.6 %; Optimum-real (knows real departures and all prices) −17.5 to −26.7 €/day, 0.6 %; Optimum-announced (knows all prices, trusts announced departures) −19.9 to −28.4 €/day, 21–24 %.

**RiskAware's curve lies below Trusting's everywhere.** Formal test (`dominance.csv`, Welch t-test, Holm over 32 tests): a RiskAware point is **cheaper and safer at the same time** (both significant) than

| RiskAware at | Trusting at | Money difference | Difference in early leavers below 70 % |
|---|---|---|---|
| λ 200 | λ 500 | +7.0 €/day | −6.4 points |
| λ 500 | λ 1000 | +3.1 €/day | −8.8 points |
| λ 500 | λ 2000 | +9.5 €/day | −6.1 points |
| λ 1000 | λ 2000 | +2.7 €/day | −10.2 points |

So every Trusting point except its cheapest one (λ 200) is beaten on both money and safety by some RiskAware point. **Raising the penalty is not a substitute for training with early departures.**

- **Trusting hits a wall:** from λ 1000 to λ 2000 it pays 6.4 € more per day for only 2.7 points fewer early leavers below 70 % (within the CI). A higher penalty makes it charge more on average, but it still does not know *when* the risk is, so the unlucky early leavers stay unprotected.
- **Optimum-announced is on Trusting's side of the plot:** even a perfect price forecast leaves 21–24 % of early leavers below 70 % if the planner trusts the announced departures.
- **Where the simple rule wins:** AFAP-target (−47.9 €/day, 0.6 %) is cheaper *and* safer than RiskAware at λ 2000 (−50.5, 4.8 %). If an operator wants almost no driver below 70 %, today's PPO is not the right tool; RiskAware is the better choice in the middle range (roughly −26 to −42 €/day).
- The perfect planner that knows real departures (bottom right) shows how much room is left: much cheaper and fully safe.

### 2. Scores at λ 200 and 2000 (`lam*/paired.csv`)
RiskAware p0.2 − Trusting, same seed; p corrected (Holm) within each λ folder.

| λ | 0 % early | 10 % early | 20 % early |
|---|---|---|---|
| 200 | +0.1 ± 1.3 (5/10 wins, p 0.88) | +2.7 ± 1.7 (9/10, p 0.016) | **+4.5** ± 1.9 (9/10, p 0.002) |
| 2000 | **−4.3** ± 2.8 (0/10, p 0.016) | +21.1 ± 9.8 (10/10, p 0.003) | **+37.0** ± 15.4 (10/10, p 0.002) |

- With a real charging need, RiskAware helps **even at λ 200** (in the light-charging case it did not).
- At λ 2000 the insurance cost when nobody leaves early becomes significant (−4.3 per day), but one early leaver in five makes Trusting collapse (−103 ± 16 vs −66 ± 2).
- Against AFAP-target (score −48.2 to −48.9): PPO wins clearly at λ 200 (−36.9 / −36.8 at 0 % early; RiskAware −40.0 at 20 % early); at λ 500 it wins when nobody leaves early (−43.3) and is about level at 20 % early (RiskAware −50.3); at λ 1000 and 2000 it loses (λ 2000, 0 % early: Trusting −54.8, RiskAware −59.0).

### 3. Does longer training help? (`long-training/`, λ 1000, seeds 0–2)

| Agent | Steps | Score, 0 % early | Score, 20 % early | Money €/day (20 %) | Early leavers below 70 % |
|---|---|---|---|---|---|
| PPO-Trusting | 6M | −48.6 | −73.1 | −38.1 | 21.3 % |
| PPO-Trusting | 12M | −48.0 | −73.6 | −37.8 | 24.2 % |
| RiskAware p0.2 | 6M | −51.7 | −60.1 | −42.9 | 9.3 % |
| RiskAware p0.2 | 12M | −50.2 | −58.4 | −40.0 | 10.1 % |
| AFAP-target | — | −48.2 | −48.5 | −47.9 | 0.6 % |

- **Doubling the training gives at most ~1.5 points** (RiskAware), within the confidence intervals of 3 seeds; Trusting does not improve at all (its best checkpoint was before 7M in all 3 runs). RiskAware's best checkpoints were at 10.7–11.7M, but the validation gain over its best by 6M was only 0.1–3.2 points.
- PPO still does not catch up with AFAP-target at λ 1000. **Training length is not what keeps PPO ~20 points below the perfect planner**; the limit is elsewhere (what the agent sees, the policy class, or PPO itself). The comparison between agents in this report uses the same 6M budget for every agent and is fair.

## Caveats
- One training risk (p0.2) on the curve; p0.05 / p0.1 were only run in the light-charging case.
- Runs at different λ are separate trainings, so the dominance tests compare independent groups (Welch), not paired seeds.
- Long-training check with 3 seeds only (a check, not a result).
- One 80 % target for every driver; synthetic early-departure rule (ACN-Data planned in Phase 6).
- 20 λ 200 runs started after the log commit 66a46be (docs only) instead of 4138ea7; no code difference.
