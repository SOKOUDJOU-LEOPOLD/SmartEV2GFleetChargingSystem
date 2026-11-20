"""PPO for one central agent with per-charger credit ("credit PPO", ADR-0005).

Plain-language idea:
- ONE network sees the whole site and chooses all 20 charger actions together
  (so it can coordinate, e.g. not overload the shared transformer).
- The critic predicts 20 values: the expected future reward of EACH charger.
- Each charger gets its own reward (own profit, own car's shortfall penalty, its
  share of any overload; rewards/shortfall_reward.py, `charger_terms`).
- Each action is then improved with ITS OWN charger's advantage, instead of one
  site-wide number for all 20 actions (better credit assignment [R11]).

Everything else is standard PPO from Stable-Baselines3 [R8], [R13]: clipping,
GAE [R12], entropy bonus, gradient clipping, logging. Only these parts change:
- `CreditActorCriticPolicy`: critic with n outputs; log-probabilities per action
  dimension. For SB3's diagonal Gaussian policy the joint log-probability is
  exactly the SUM of the per-dimension ones, so nothing is approximated here.
- `CreditRolloutBuffer`: stores n rewards / values / log-probs per step and runs
  GAE separately for each charger. GAE is linear, so the per-charger advantages
  add up to the advantage normal PPO would compute for the site reward
  (tests/test_credit_ppo.py checks this).
- `CreditPPO`: collects vector rewards; loss = Σ_chargers clipped surrogate with
  that charger's advantage (mean over the batch), the same scale as normal PPO's
  joint log-probability (which is also a sum over dimensions).

Requirement: the environment returns one reward PER ACTION DIMENSION
(envs/charger_env.py, `SiteChargerEnv(per_charger_obs=False)`). SB3's VecNormalize
cannot rescale vector rewards, so use `normalize_reward: false` and the fixed
`reward_scale` instead (observation normalization still works).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from torch.nn import functional


class CreditActorCriticPolicy(ActorCriticPolicy):
    """Actor as in SB3; critic with one output per action dimension (= per charger)."""

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise ValueError("Credit PPO needs a continuous (diagonal Gaussian) action space")
        n = int(np.prod(self.action_space.shape))
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, n)
        if self.ortho_init:  # same initialization SB3 uses for its 1-output critic
            self.value_net.apply(partial(self.init_weights, gain=1))
        # Rebuild the optimizer so it trains the new critic head.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi_features, vf_features = features
        return (
            self.mlp_extractor.forward_actor(pi_features),
            self.mlp_extractor.forward_critic(vf_features),
        )

    @staticmethod
    def per_dim_log_prob(distribution, actions: th.Tensor) -> th.Tensor:
        """log π_i(a_i | s) for every dimension i, shape (batch, n). Their sum is SB3's
        joint log-probability (independent normals)."""
        return distribution.distribution.log_prob(actions)

    def forward(self, obs, deterministic: bool = False):
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)  # (batch, n)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = self.per_dim_log_prob(distribution, actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def evaluate_actions(self, obs, actions: th.Tensor):
        latent_pi, latent_vf = self._latents(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = self.per_dim_log_prob(distribution, actions)
        return self.value_net(latent_vf), log_prob, distribution.entropy()

    def predict_values(self, obs) -> th.Tensor:
        _, latent_vf = self._latents(obs)
        return self.value_net(latent_vf)


class CreditRolloutBuffer(RolloutBuffer):
    """RolloutBuffer with `reward_dim` rewards / values / log-probs per step."""

    def __init__(self, *args, reward_dim: int, **kwargs) -> None:
        self.reward_dim = reward_dim
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        shape = (self.buffer_size, self.n_envs, self.reward_dim)
        for name in ["rewards", "returns", "values", "log_probs", "advantages"]:
            setattr(self, name, np.zeros(shape, dtype=np.float32))

    def add(self, obs, action, reward, episode_start, value, log_prob) -> None:
        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action).reshape((self.n_envs, self.action_dim))
        self.rewards[self.pos] = np.array(reward).reshape((self.n_envs, self.reward_dim))
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().reshape((self.n_envs, -1))
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy().reshape((self.n_envs, -1))
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """GAE [R12] for each charger separately (same formula as SB3, one more axis)."""
        last_values = last_values.clone().cpu().numpy().reshape((self.n_envs, self.reward_dim))
        last_gae_lam = np.zeros((self.n_envs, self.reward_dim), dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            not_done = next_non_terminal[:, None]  # same for every charger of a site
            delta = self.rewards[step] + self.gamma * next_values * not_done - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * not_done * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds],  # (batch, n): NOT flattened, unlike SB3
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self.returns[batch_inds],
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class CreditPPO(PPO):
    """SB3 PPO with per-charger rewards, values and advantages (ADR-0005)."""

    policy_aliases = {"MlpPolicy": CreditActorCriticPolicy}

    def __init__(self, policy, env, *args, reward_scale: float = 1.0, **kwargs) -> None:
        # Fixed multiplier for the per-charger rewards (VecNormalize cannot rescale them).
        self.reward_scale = reward_scale
        super().__init__(policy, env, *args, **kwargs)

    def _setup_model(self) -> None:
        if not isinstance(self.action_space, spaces.Box):
            raise ValueError("Credit PPO needs a Box action space (one action per charger)")
        self.rollout_buffer_class = CreditRolloutBuffer
        self.rollout_buffer_kwargs = {
            **(self.rollout_buffer_kwargs or {}),
            "reward_dim": int(np.prod(self.action_space.shape)),
        }
        super()._setup_model()

    def collect_rollouts(self, env: VecEnv, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        """SB3's collect_rollouts, keeping one reward per charger."""
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            # The Gaussian can sample outside [-1, 1]; the simulator gets clipped actions.
            clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            rewards = np.asarray(rewards, dtype=np.float32).reshape(
                env.num_envs, rollout_buffer.reward_dim
            )
            rewards = rewards * self.reward_scale
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            # Time-limit endings: add the critic's value of the cut-off state (SB3 #633).
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value.cpu().numpy()

            rollout_buffer.add(
                self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        """SB3's PPO update; each action dimension uses its own charger's advantage."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        continue_training = True
        for _epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, rollout_data.actions
                )
                advantages = rollout_data.advantages  # (batch, n)
                if self.normalize_advantage and advantages.numel() > 1:
                    # One mean/std over all chargers: keeps their relative importance.
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Per-charger probability ratio and PPO clipping [R8].
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                surrogate_1 = advantages * ratio
                surrogate_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                # Sum over chargers (like PPO's joint log-prob), mean over the batch.
                policy_loss = -th.min(surrogate_1, surrogate_2).sum(dim=1).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():  # KL of the JOINT policy (sum over chargers)
                    log_ratio = (log_prob - rollout_data.old_log_prob).sum(dim=1)
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
