# RL Concepts — Plain-Language Glossary

> Short explanations of every idea the project uses, with references.
> Code comments point here with tags like `[R10]`.

## Core ideas

**Reinforcement learning (RL).** An agent learns by trial and error: it takes an action, the environment answers with a new situation and a reward (a score), and the agent slowly learns which actions lead to high total score. *Model-free* means the agent never learns the rules of the simulator explicitly — it only learns from experience. [R3], [R11]

**MDP (Markov Decision Process).** The standard math description of an RL problem: states, actions, transition rules, rewards. "Markov" means the current state contains everything needed to decide — the past adds nothing. [R11]

**POMDP (Partially Observable MDP).** Like an MDP, but the agent cannot see the full state. In our project the **true departure time is hidden**; the agent only sees the announced one. Two situations that look identical can turn out differently. [R10]

**Belief state.** In a POMDP, the ideal agent keeps a "best guess" (a probability distribution) of what the hidden state is, and updates it as it sees more. Exact beliefs are too expensive for big problems, so practical agents use the current observation or a learned memory of the past. [R10]

**Policy.** The agent's decision rule: given what it sees, which action to take. Here: a neural network that outputs one charge/discharge level in [−1, 1] per charger.

**Return.** The total (discounted) reward over an episode. **Discount factor γ** (gamma): how much the agent cares about the future vs. now (γ close to 1 = very far-sighted). [R11]

**Episode.** One full simulation run (e.g. one simulated day at the charging site).

## PPO and its parts

**Actor-critic.** Two networks (or two heads): the *actor* chooses actions; the *critic* estimates how good a situation is (the *value*). The critic helps the actor know if an action was better or worse than expected. [R8]

**Advantage.** "How much better was this action than what I expected?" = actual outcome − critic's expectation. Positive ⇒ do it more often.

**GAE (Generalized Advantage Estimation).** A way to compute advantages that balances bias and noise using a parameter λ_GAE (not the same as our penalty λ). [R12]

**PPO (Proximal Policy Optimization).** Our learning algorithm. It improves the policy in small, safe steps: the *clipped objective* stops one batch of experience from changing the policy too much. This matters for us because rare, large departure penalties could otherwise cause wild updates. [R8]

**On-policy.** PPO only learns from experience collected by its current policy, then throws it away. Simple and stable, but needs many simulation steps.

**Training diagnostics we log:** *policy loss* (how the actor is changing), *value loss* (how wrong the critic is), *entropy* (how random/exploratory the policy still is), *approx KL* and *clip fraction* (how big each update is), *explained variance* (how well the critic predicts returns; close to 1 is good). [R8], [R13]

## Reward design

**Reward shaping / penalty weights.** The reward mixes several goals (profit, grid limits, drivers' charge). The weights (our λ_overload, λ_departure) decide the trade-off. Prior work shows the weight, not the algorithm, often decides where the agent ends up — so we test several λ values. [R4], [R7]

**Sparse / delayed reward and credit assignment.** Our departure penalty arrives only when a car leaves, possibly hours after the decision that caused it. The agent must figure out *which earlier actions* were responsible — this is the credit-assignment problem, and it is what makes learning to hedge hard. [R11]

**Constrained MDP / Lagrangian method (possible extension).** Instead of choosing λ by hand, state a rule ("shortfall must stay below X") and let the algorithm tune the weight automatically. [R4]

## Making training work well

**Observation & reward normalization.** Prices, power and penalties have very different scales; rescaling them to similar ranges makes neural-network training much more stable. In SB3: `VecNormalize`. [R13]

**Vectorized environments.** Running several copies of the simulator at once to collect experience faster (`SubprocVecEnv`). [R13]

**Seeds & reproducibility.** A seed fixes all randomness. RL results vary a lot between seeds, so we train every setting with ≥ 5 seeds and report mean ± 95% confidence interval. [R14], [R15]

**IQM & bootstrap confidence intervals.** Robust ways to summarize results over few seeds: interquartile mean ignores the extreme best/worst runs; bootstrap estimates uncertainty by resampling. [R15]

## Robustness, baselines, trade-offs

**Domain randomization / robust RL.** Training in many randomized versions of the environment so the policy works even when conditions change. Our wrapper is a form of this: hidden early departures are randomized during training. [R16], [R9]

**Oracle (privileged information) baseline.** An agent allowed to see the hidden information (true departure). It shows the best achievable result and how much of the gap our agent closes. [R6]

**Perfect-knowledge optimum (hindsight plan).** Not an RL agent: a planner that is told the whole day in advance (prices, cars, real departures) and solves it exactly as one optimisation problem (a linear program; squares are approximated by many touching straight lines). Nobody can score higher on that day, so it is the ceiling every agent is measured against. A second version trusts the announced departures; the gap between the two is the most that knowing early departures can be worth. Code: `eval/optimum.py`. [R1], [R19]

**Imitation warm start (behaviour cloning, then RL).** Before learning by trial and error, the agent first copies a teacher: it is shown many situations and the teacher's action in each, and learns to reproduce them (behaviour cloning). Its critic also learns how good each situation was on the teacher's days. Then normal PPO training continues from that starting point. Copying alone tends to drift: small errors add up and the copy never saw how to recover from them, so the RL phase is still needed. Our teacher is the perfect plan that trusts announced departures; code: `agents/imitation.py`. [R20], [R21]

**Many comparisons at once (Holm correction).** One test at the 5 % level is fooled by luck 1 time in 20. If we compare many pairs at once (e.g. 3 RiskAware variants × 4 test worlds = 12 per λ), a "significant" result can appear by luck alone. The Holm correction raises each p-value (the smallest is multiplied by the number of tests, the next by one less, ...) so the chance that ANY of them is a fluke stays below 5 %. `paired.csv` reports both the plain p-value and the one corrected over all rows of that file. [R22]

**Heuristic baseline (AFAP).** A fixed hand-written rule: charge every car as fast as possible. Very safe for drivers, but ignores prices and V2G. [R1]

**Hedging.** Playing it safe against a risk you cannot see — here, keeping extra charge or selling less energy to the grid as the announced departure gets close.

**Pareto front / trade-off curve.** When two goals conflict (profit vs. shortfall), we plot every agent as a point; the "front" is the set of best possible compromises. We compare agents by these curves, not by one total reward number. [R7]

**Train/test grid (cross-evaluation).** Test each trained agent in every environment (no early departures, few, many) to separate what an agent learned from where it was tested. [R7]

**Parameter sharing (one policy for many identical units).** Instead of one big network deciding for all 20 chargers at once, one small network decides for one charger at a time and is reused for every charger. Each charger gets its own share of the reward, so a penalty lands on the charger that caused it (better credit assignment), and every simulated step gives 20 learning examples. [R18]

## Memory (stretch goal)

**Recurrent policy (LSTM).** A neural network with memory: it carries a hidden summary of past observations. Useful when the current observation is not enough, e.g. to infer that "today many cars are leaving early". In SB3-contrib: `RecurrentPPO` with `MlpLstmPolicy`. [R9], [R17]

**History length / BPTT.** How many past steps the LSTM learns from (backpropagation through time). Ni et al. found this is a key setting to tune. [R9]

**Frame stacking.** A cheap alternative to memory: give the agent the last *k* observations glued together.

## Domain terms

**V2G (Vehicle-to-Grid).** Cars can also *sell* energy back to the grid (discharge), not only charge. [R1]

**SoC (State of Charge).** Battery level, from 0 (empty) to 1 (full).

**EVSE.** A charger port. **Transformer limit P_max.** Maximum total power the site can draw safely.

---

## References

| Tag | Reference |
|---|---|
| R1 | S. Orfanoudakis et al., "EV2Gym: A Flexible V2G Simulator for EV Smart Charging Research and Benchmarking," IEEE T-ITS, 2025. arXiv:2404.01849. Code: https://github.com/StavrosOrf/EV2Gym |
| R2 | Z. J. Lee, T. Li, S. H. Low, "ACN-Data: Analysis and Applications of an Open EV Charging Dataset," e-Energy 2019. https://ev.caltech.edu/dataset |
| R3 | Z. Wan et al., "Model-Free Real-Time EV Charging Scheduling Based on Deep RL," IEEE T-SG, 2019. |
| R4 | G. Chen, X. Shi, "A Deep RL-Based Charging Scheduling Approach with Augmented Lagrangian for EV," arXiv:2209.09772, 2022. |
| R5 | N. Sadeghianpourhamami et al., "Definition and Evaluation of Model-Free Coordination of EV Charging With RL," IEEE T-SG, 2020. |
| R6 | G. Gabriele et al., "Forecasting What Matters: Decision-Focused RL for Controlled EV Charging with Unknown Departure Times," e-Energy 2026. |
| R7 | S. Naghdizadegan Jahromi et al., "Adaptive EV Charging Scheduling in Real-Time Imbalance Markets Using Model-Free RL," Applied Energy, 2026. |
| R8 | J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017. |
| R9 | T. Ni, B. Eysenbach, R. Salakhutdinov, "Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs," ICML 2022. |
| R10 | L. P. Kaelbling, M. L. Littman, A. R. Cassandra, "Planning and Acting in Partially Observable Stochastic Domains," Artificial Intelligence, 1998. |
| R11 | R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018. http://incompleteideas.net/book/the-book-2nd.html |
| R12 | J. Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation," ICLR 2016. arXiv:1506.02438 |
| R13 | A. Raffin et al., "Stable-Baselines3: Reliable RL Implementations," JMLR, 2021. Docs: https://stable-baselines3.readthedocs.io/ |
| R14 | P. Henderson et al., "Deep Reinforcement Learning that Matters," AAAI 2018. |
| R15 | R. Agarwal et al., "Deep RL at the Edge of the Statistical Precipice," NeurIPS 2021 (rliable). |
| R16 | J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," IROS 2017. |
| R17 | SB3-Contrib RecurrentPPO docs: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html |
| R18 | C. Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games," NeurIPS 2022 (Datasets and Benchmarks). arXiv:2103.01955 |
| R19 | Q. Huangfu, J. A. J. Hall, "Parallelizing the dual revised simplex method," Mathematical Programming Computation, 2018 (the HiGHS solver, used through SciPy `linprog(method="highs")`). https://highs.dev |
| R20 | D. A. Pomerleau, "Efficient Training of Artificial Neural Networks for Autonomous Navigation," Neural Computation, 1991 (behaviour cloning). |
| R21 | A. Rajeswaran et al., "Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations," RSS 2018 (pre-train on demonstrations, then policy-gradient RL). arXiv:1709.10087 |
| R22 | S. Holm, "A Simple Sequentially Rejective Multiple Test Procedure," Scandinavian Journal of Statistics, 1979. |

R1–R10 match the numbering of the [literature review](../proposal/Literature_Review.md).
