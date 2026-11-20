"""A deliberately simple reward used only to check that the training pipeline learns.

`energy_delivered`: score = net energy put into the cars during the step (kWh).
Charging counts +, selling back to the grid (V2G) counts -. The best possible
policy is known: charge every car as fast as possible (AFAP). So if PPO is set up
correctly, its score must climb towards AFAP's. Not used for research results.
Phase 1 pipeline check, see docs/plan/phase-1-env-sanity-ppo.md.
"""

from __future__ import annotations


def energy_delivered(env, total_costs, user_satisfaction_list, *args) -> float:
    """Net kWh delivered to all cars in the step that just finished.

    Same signature as EV2Gym's reward functions (ev2gym/rl_agent/reward.py).
    EV2Gym computes the reward AFTER moving to the next step, so the step that
    just finished is `current_step - 1`. `current_power_usage` is the site's net
    power in kW (charging +, discharging -); kW x hours = kWh.
    """
    step = env.current_step - 1
    hours = env.timescale / 60
    return float(env.current_power_usage[step] * hours)
