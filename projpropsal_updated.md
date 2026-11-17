Risk-Aware Deep Reinforcement Learning for EV
Charging Under Unexpected Departures
Yeshas Krishna Leopold Sokoudjou Gatsing
1 Team Members
• Yeshas Krishna
• Leopold Sokoudjou Gatsing
2 Proposed Starting Point
We will use EV2Gym, an open-source Gymnasium-compatible simulator for electric vehicle smart
charging and Vehicle-to-Grid (V2G) control, as our primary code base. We will use Stable-
Baselines3 for reinforcement learning, beginning with Proximal Policy Optimization (PPO).
EV2Gym already models EV arrivals and departures, state of charge (SoC), charging and dis-
charging actions, electricity prices, charging stations, transformer constraints, and V2G operation.
In EV2Gym’s V2G profit-maximization setting, the controller is given the EV’s scheduled de-
parture time and desired battery capacity while the EV remains connected. This provides the
controller with highly reliable departure information and allows it to treat the remaining charging
window as effectively deterministic.
Our project modifies this assumption by introducing unannounced early departures. We
will implement a custom Gymnasium wrapper around EV2Gym that introduces a hidden actual
departure time that may differ from the departure time reported to the controller.
When an EV connects, the wrapper will retain the reported departure time
Tscheduled
i
but may privately sample an earlier actual departure time
Tactual < Tscheduled.
i i
The controller will continue to observe only the scheduled departure time. It will receive no
warning that the vehicle is going to leave early and will only observe the consequence after the EV
has disconnected.
This creates a partially observable charging problem in which the controller must learn to
manage departure risk without being directly told when an unexpected departure will occur.
Wedonotcurrentlyrequireadditionalcodeordatafromtheinstructors. Additionalcomputing
resourcesmaybeusefulforrunningmultipleseedsandparametersweeps, buttheyarenotrequired
for the initial implementation.
1

3 Main Research Question
Our main research question is:
Can a model-free reinforcement learning agent optimize V2G fleet schedul-
ing under partial observability while implicitly managing the risk of unan-
nouncedearlydepartures, andwhateconomicorgrid-performancetrade-offs
emerge from this more robust policy?
A controller may normally observe that an EV does not leave for several hours and decide to
discharge energy for V2G profit, expecting to recharge the vehicle later.
However, if the driver unexpectedly leaves earlier than reported, the vehicle may depart with
insufficient energy.
Our goal is to determine whether an RL agent that repeatedly experiences this hidden risk
during training learns to maintain an energy reserve and becomes more robust to unexpected
departures.
4 Why the Problem is Sequential
The impact of a charging or discharging action is not limited to the current timestep.
Discharging an EV now may:
• generate immediate V2G profit,
• help reduce grid demand,
• lower the EV’s current SoC,
• require additional charging later, and
• create a future SoC shortfall if the EV unexpectedly leaves.
Therefore, actions have delayed and cascading consequences along the state trajectory:
(s ,a ) → s → s → ··· → s .
t t t+1 t+2 departure
The controller must balance immediate economic or grid benefits against future uncertainty in
vehicle availability.
5 Partial Observability
For each EV i, the environment contains both a scheduled departure time
Tscheduled
i
and a possibly different actual departure time
Tactual.
i
2

The controller observes only Tscheduled.
i
If an early departure is sampled,
Tactual < Tscheduled,
i i
but Tactual remains hidden.
i
Because this hidden variable affects future transitions but is not part of the controller’s obser-
vation, the modified environment can be viewed as a Partially Observable Markov Decision Process
(POMDP).
The agent must therefore learn a policy based on incomplete information.
6 Observation Space
We will initially use the EV2Gym V2G observation structure:
[︂ ]︂ [︂ ]︂ [︂ ]︂
S = t, Ptot, cch ∪ P˜L −P˜PV , PDR ∪ SoC , tdep−t .
t t−1 t:t+h w,t:t+h w,t:t+h w,t:t+h j,i,t j,i
The observation includes:
• t: current simulation timestep,
• Ptot: total EV charging/discharging power at the previous step,
t−1
• cch : charging/discharging electricity price forecast over horizon h,
t:t+h
• P˜L −P˜PV : forecasted net inflexible load at transformer w,
w,t:t+h w,t:t+h
• PDR : demand-response information,
w,t:t+h
• SoC : current SoC of the EV connected to EVSE j at charging station i,
j,i,t
• tdep−t: remaining time until the EV’s scheduled departure.
j,i
The hidden actual departure time introduced by our wrapper will not be included in the obser-
vation.
7 Action Space
The controller will output one continuous action for each EVSE:
a = [a ,a ,...,a ],
t 1,t 2,t N,t
where
a ∈ [−1,1].
i,t
Conceptually,
a > 0 ⇒ charging,
i,t
3

a = 0 ⇒ idle,
i,t
and
a < 0 ⇒ V2G discharge.
i,t
The magnitude determines the requested charging or discharging power.
8 Transition Function
The agent is model-free and interacts with EV2Gym only through sampled transitions returned by
the environment.
The normal physical dynamics are handled by EV2Gym:
s ∼ P (s | s ,a ).
t+1 EV2Gym t+1 t t
Our wrapper adds an additional hidden event.
When an EV connects, the wrapper may privately sample an earlier actual departure time:
Tactual = Tscheduled−∆ ,
i i i
where
∆ > 0.
i
The agent does not observe ∆ or Tactual.
i i
When the hidden departure time is reached, the EV becomes disconnected.
Thus, two identical observations and actions may lead to different next observations depending
on whether a hidden early departure occurs.
Thisuncertaintyiswhatforcesthecontrollertolearnarisk-hedgingpolicyratherthanassuming
that the reported departure schedule is guaranteed.
9 Reward Function
We will use the same reward structure for all learning agents so that the primary experimental
difference is whether the controller was exposed to unexpected departures during training.
The preliminary reward is:
R = R −λ C −λ C .
t economic,t overload overload,t departure departure,t
The economic reward will reflect the profit or cost of charging and discharging:
R = −Pchcch+Pdiscdis.
economic,t t t t t
The overload penalty will penalize power above the transformer limit:
4

(︂ )︂2
C = max 0,Ptotal−Pmax .
overload,t t
The departure penalty will be applied whenever an EV leaves, whether the departure is sched-
uled or unexpected:
∑︂
C = [max(0,SoC∗−SoC )]2,
departure,t i i,t
i∈Dt
where D is the set of EVs leaving at timestep t.
t
For example, if an EV unexpectedly leaves with
SoC = 0.50
i,t
and has a desired departure SoC of
SoC∗ = 0.80,
i
the SoC deficit is
0.80−0.50 = 0.30,
producing an unscaled penalty of
0.302 = 0.09.
Importantly, the penalty is received only after the departure occurs. It does not warn the agent
that an unexpected departure is coming.
Battery degradation will remain available as an evaluation metric in EV2Gym but will not be
part of the core reward in the initial experiment. It may be considered as a later extension.
10 What We Plan to Do and How We Plan to Do It
The goal of this project is to train a smart charging controller for a fleet of EVs sharing constrained
charging infrastructure.
We will create an UnexpectedDepartureWrapper around EV2Gym.
When an EV connects, it will have a reported departure time. With probability
p ,
early
the wrapper will privately sample an earlier actual departure time.
Initially, we will use a synthetic distribution for early departures. We will vary both:
p
early
and the departure shift
∆ .
i
5

For example, we may evaluate:
p ∈ {0, 0.05, 0.10, 0.20}.
early
The controller will never observe which EVs are selected for early departure.
We will train the controller using PPO through Stable-Baselines3. PPO provides a well-
established continuous-control baseline that is compatible with EV2Gym and allows us to directly
compare controllers trained under different assumptions about departure uncertainty.
We will compare two primary PPO controllers:
1. Baseline PPO: trained in standard EV2Gym without hidden early departures.
2. Unexpected-Departure PPO: trained in the modified environment with hidden early depar-
tures.
The two PPO controllers will use the same architecture, action space, observation structure, re-
ward function, and training procedure. The main difference will be whether unexpected departures
are present during training.
We will evaluate both agents on identical test environments containing hidden early departures.
We may also compare against EV2Gym’s Charge-As-Fast-As-Possible (AFAP) heuristic.
AFAP is expected to provide strong departure readiness but may sacrifice economic efficiency or
grid flexibility.
11 Evaluation Metrics
We will evaluate several outcomes simultaneously:
• percentage of unexpected departures meeting target SoC,
• average SoC shortfall at unexpected departure,
• total charging cost or V2G profit,
• total V2G energy discharged,
• transformer overload,
• average user satisfaction,
• cumulative reinforcement learning return.
The main trade-off is:
economic and grid flexibility versus robustness to unexpected departure .
A controller could trivially reduce early-departure failures by always keeping every EV fully
charged. However, this would reduce the flexibility and financial benefits available from smart V2G
scheduling.
The more interesting question is whether the RL controller can learn an effective compromise.
6

12 Simplest Possible First Result
Beforetestingourmainhypothesis,wewillfirsttrainastandardPPOagentinunmodifiedEV2Gym
and verify that the training pipeline functions correctly.
This will serve as a sanity check for the simulator, observation space, action space, reward
function, and training loop.
Our first substantive experiment will then compare:
• PPO trained under the standard deterministic departure assumption,
• PPO trained with hidden unexpected departures.
Bothagentswillbeevaluatedunderthesametestconditionscontainingunexpecteddepartures.
Our main hypothesis is:
APPOagenttrainedwithhiddenearlydepartureswilllearntomaintainalargerenergy
reserve and will therefore experience fewer SoC shortfalls during unexpected departures
than a PPO agent trained under standard departure assumptions, at the cost of some
economic or V2G flexibility.
If the wrapper-trained controller produces fewer undercharged departures while maintaining a
reasonable amount of profit, this will provide evidence that an RL policy can learn to hedge against
departure risk even when that risk is never directly revealed.
13 Stretch Goal
Our initial experiment intentionally keeps the unexpected departure hidden.
Asastretchgoal,wewillinvestigatewhetherthesyntheticdeparture-riskmodelcanbereplaced
by a more realistic, data-driven model.
The public ACN-Data dataset records both the departure time requested by an EV user and
the actual disconnect time for charging sessions.
We can therefore estimate the empirical difference
∆T = T −T
requested actual
for sessions where the vehicle leaves earlier than reported.
This would allow us to build an empirical early-departure distribution rather than choosing the
distribution manually.
Afurtherstretchgoalistoinvestigatewhethertemporalhistorycanimprovepolicyperformance
under partial observability.
If the data-driven departure risk contains historical patterns that are not fully represented
by the current observation, we may compare feedforward PPO with Recurrent PPO using an
LSTM-based policy.
This would allow us to study whether memory helps the controller infer departure risk from
temporal context.
7

14 Initial References
1. S.I.Orfanoudakisetal.,“EV2Gym: AFlexibleV2GSimulatorforEVSmartChargingResearch
and Benchmarking,” 2024.
2. J.Schulman,F.Wolski,P.Dhariwal,A.Radford,andO.Klimov,“ProximalPolicyOptimization
Algorithms,” arXiv:1707.06347, 2017.
3. Stable-Baselines3 Documentation, https://stable-baselines3.readthedocs.io/.
4. California Institute of Technology, “ACN-Data: Electric Vehicle Charging Dataset,” https:
//ev.caltech.edu/dataset.
8
