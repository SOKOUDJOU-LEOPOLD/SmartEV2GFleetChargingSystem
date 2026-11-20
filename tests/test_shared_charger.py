"""Tests for the shared per-charger policy design (ADR-0004)."""

from types import SimpleNamespace

import numpy as np
import pytest

from smartev2g.envs.charger_env import ChargerVecEnv, SiteChargerEnv, SiteDummyVecEnv
from smartev2g.envs.charger_obs import N_FEATURES, charger_observation_names, charger_observations
from smartev2g.envs.make_env import make_env
from smartev2g.envs.wrappers import EpisodeInfo
from smartev2g.rewards.shortfall_reward import charger_terms

LAMBDAS = {"lambda_dep": 50.0, "lambda_over": 1.0}
DAY = 1_000_000


def shortfall_env(seed=DAY):
    return make_env(seed=seed, reward_function="squared_shortfall", reward_kwargs=LAMBDAS)


def test_charger_observation_matches_ev2gym_observation():
    """Shared part = EV2Gym's (t, P_prev, prices); own part = that port's (soc, t_dep)."""
    env = make_env(seed=DAY)
    obs, _ = env.reset(seed=DAY)
    rng = np.random.default_rng(0)
    assert len(charger_observation_names()) == N_FEATURES == 26
    for _ in range(40):
        co = charger_observations(env)
        assert co.shape == (20, N_FEATURES)
        np.testing.assert_allclose(co[:, :22], np.tile(obs[:22], (20, 1)))
        np.testing.assert_allclose(co[:, 23], obs[22::2])  # soc of each port
        np.testing.assert_allclose(co[:, 24], obs[23::2])  # steps to announced departure
        assert co[0, 22] == co[:, 25].sum()  # n_cars = number of plugged chargers
        obs, *_ = env.step(rng.uniform(-1, 1, 20))


def test_charger_rewards_add_up_to_site_reward_every_step():
    env = shortfall_env()
    env.reset(seed=DAY)
    rng = np.random.default_rng(1)
    done, departures = False, 0
    while not done:
        _, reward, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 20))
        assert env.charger_rewards.sum() == pytest.approx(reward, abs=1e-9)
        for k, total in env.reward_terms.items():
            assert env.charger_reward_terms[k].sum() == pytest.approx(total, abs=1e-9)
        # The departure penalty lands on the charger whose car left, nowhere else.
        expected = np.zeros(20)
        for ev in env.departing_evs:
            target = ev.desired_capacity / ev.battery_capacity
            soc = ev.current_capacity / ev.battery_capacity
            expected[ev.location] += LAMBDAS["lambda_dep"] * max(0, target - soc) ** 2
        np.testing.assert_allclose(env.charger_reward_terms["departure_penalty"], expected)
        departures += len(env.departing_evs)
        done = terminated or truncated
    assert departures > 0


def test_overload_blame_goes_to_chargers_pushing_in_that_direction():
    """+130 kW on a 100 kW transformer: only the charging chargers share the penalty,
    in proportion to their power; the discharging one pays nothing."""
    stations = [
        SimpleNamespace(total_profits=0.0, current_power_output=p, connected_transformer=0)
        for p in [90.0, 60.0, -20.0]
    ]
    tr = SimpleNamespace(
        id=0, current_power=130.0, max_power=np.full(3, 100.0), min_power=np.full(3, -100.0)
    )
    env = SimpleNamespace(charging_stations=stations, transformers=[tr], departing_evs=[])
    for i, cs in enumerate(stations):
        cs.id, cs.n_ports = i, 1
    parts = charger_terms(env, step=0, lambda_dep=0.0, lambda_over=1.0)
    total = 30.0**2
    np.testing.assert_allclose(parts["overload_penalty"], [total * 0.6, total * 0.4, 0.0])


def test_vec_env_routes_each_action_to_its_own_charger():
    """Action +1 only for one plugged-in charger: only that charger draws power."""
    venv = ChargerVecEnv(SiteDummyVecEnv([lambda: SiteChargerEnv(EpisodeInfo(shortfall_env()))]))
    venv.seed(DAY)
    obs = venv.reset()
    assert obs.shape == (20, N_FEATURES) and venv.num_envs == 20
    target = None
    while target is None:  # wait until some car is plugged in
        plugged = np.flatnonzero(obs[:, 25] == 1)
        if len(plugged):
            target = int(plugged[0])
            break
        obs, *_ = venv.step(np.zeros((20, 1)))
    actions = np.zeros((20, 1))
    actions[target] = 1.0
    obs, rewards, dones, infos = venv.step(actions)
    site = venv.venv.envs[0].unwrapped
    power = np.array([cs.current_power_output for cs in site.charging_stations])
    assert power[target] > 0
    assert np.count_nonzero(power) == 1
    assert rewards.shape == (20,) and dones.shape == (20,) and len(infos) == 20


def test_vec_env_reports_each_day_once_with_site_return():
    venv = ChargerVecEnv(SiteDummyVecEnv([lambda: SiteChargerEnv(EpisodeInfo(shortfall_env()))]))
    venv.seed(DAY)
    venv.reset()
    rng = np.random.default_rng(0)
    site_return, done_infos = 0.0, []
    for _ in range(112):
        _, rewards, dones, infos = venv.step(rng.uniform(-1, 1, (20, 1)))
        site_return += rewards.sum()
        if dones.any():
            done_infos = infos
    episodes = [i for i in done_infos if "episode" in i]
    assert len(episodes) == 1  # only charger 0 carries the site's episode info
    assert episodes[0]["episode"]["r"] == pytest.approx(site_return, rel=1e-5)
    assert episodes[0]["day_seed"] == DAY
    assert all("terminal_observation" in i for i in done_infos)
