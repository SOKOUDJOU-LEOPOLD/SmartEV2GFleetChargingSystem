"""Small Gymnasium wrappers used around the EV2Gym environment."""

from __future__ import annotations

import gymnasium as gym


class EpisodeInfo(gym.Wrapper):
    """At the end of each episode, add to `info`:

    - `day_seed`: which simulated day (seed) was played, so every training episode
      can be traced and train/validation/test days can be checked to never mix. [R14]
    - `n_early`: how many cars left before their announced time (hidden departures).
    - `reward_terms_episode`: the day's separate reward terms (profit, overload
      penalty, departure penalty), for rewards that record them
      (docs/plan/METRICS_AND_LOGGING.md: "each reward term separately").
    The training callbacks only see `info`, so this is how they get these values.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            base = self.env.unwrapped
            info = {**info, "day_seed": int(base.seed)}
            # Number of cars that left before their announced time (hidden departures).
            info["n_early"] = len(getattr(base, "departure_plan", {}))
            terms = getattr(base, "reward_terms_episode", None)
            if terms is not None:
                info["reward_terms_episode"] = dict(terms)
        return obs, reward, terminated, truncated, info
