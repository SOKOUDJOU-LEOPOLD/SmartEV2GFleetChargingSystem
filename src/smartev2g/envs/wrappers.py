"""Small Gymnasium wrappers used around the EV2Gym environment."""

from __future__ import annotations

import gymnasium as gym


class RewardTermsInfo(gym.Wrapper):
    """At the end of each episode, copy the day's reward terms into `info`.

    Our reward stores its separate terms (profit, overload penalty, departure
    penalty) on the env; this makes them visible to the training callbacks,
    which only see `info` (docs/plan/METRICS_AND_LOGGING.md: "each reward term
    separately"). Does nothing for rewards that do not record terms.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        terms = getattr(self.env.unwrapped, "reward_terms_episode", None)
        if (terminated or truncated) and terms is not None:
            info = {**info, "reward_terms_episode": dict(terms)}
        return obs, reward, terminated, truncated, info
