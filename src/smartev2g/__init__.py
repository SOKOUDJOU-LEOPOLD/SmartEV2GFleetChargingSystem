"""smartev2g: risk-aware RL for EV fleet charging under unexpected early departures.

Sub-packages (filled phase by phase, see docs/plan/ROADMAP.md):
- envs:    EV2Gym environment factory and the UnexpectedDepartureWrapper (Phase 1-2)
- rewards: squared-shortfall reward (Phase 2)
- agents:  PPO training and the AFAP baseline (Phase 1, 3)
- eval:    metrics and hedging analysis (Phase 3)
- utils:   seeding, logging, run info (Phase 3)
"""

__version__ = "0.0.1"
