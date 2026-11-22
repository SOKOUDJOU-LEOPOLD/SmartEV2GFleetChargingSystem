"""Tests for our laxity observation (envs/state.py)."""

import numpy as np

from smartev2g.agents.baselines import AFAPPolicy
from smartev2g.envs.make_env import make_env, observation_names

DAY = 1_000_000
LAXITY = "V2G_profit_max_laxity"


def test_extends_ev2gym_observation():
    """Same t, P_prev, prices, soc, t_dep as EV2Gym's V2G_profit_max; + 2 numbers per port."""
    ours, theirs = make_env(seed=DAY, state_function=LAXITY), make_env(seed=DAY)
    o, _ = ours.reset(seed=DAY)
    e, _ = theirs.reset(seed=DAY)
    rng = np.random.default_rng(0)
    assert len(observation_names(ours, LAXITY)) == o.shape[0] == 2 + 20 + 4 * 20
    for _ in range(50):
        np.testing.assert_allclose(o[:22], e[:22])
        np.testing.assert_allclose(o[22::4], e[22::2])  # soc
        np.testing.assert_allclose(o[23::4], e[23::2])  # t_dep
        np.testing.assert_allclose(o[25::4], o[23::4] - o[24::4])  # laxity = t_dep − needed
        a = rng.uniform(-1, 1, 20)
        o, *_ = ours.step(a)
        e, *_ = theirs.step(a)


def test_laxity_predicts_whether_full_speed_reaches_the_target():
    """Charging at full speed from arrival: cars with laxity >= 1 step at arrival reach
    their target; cars with laxity <= -1 step do not."""
    env = make_env(seed=DAY, state_function=LAXITY)
    names = observation_names(env, LAXITY)
    afap = AFAPPolicy()
    checked = 0
    for day in range(DAY, DAY + 5):
        obs, _ = env.reset(seed=day)
        laxity_at_arrival, done = {}, False
        while not done:
            for k in range(env.number_of_ports):
                ev = env.charging_stations[k].evs_connected[0]
                if ev is not None and id(ev) not in laxity_at_arrival:
                    laxity_at_arrival[id(ev)] = (ev, obs[names.index(f"laxity_{k}")])
            obs, _, terminated, truncated, _ = env.step(afap.act(obs, env))
            done = terminated or truncated
        for ev, laxity in laxity_at_arrival.values():
            reached = ev.current_capacity >= ev.desired_capacity - 0.5  # within 0.5 kWh
            if laxity >= 1:
                assert reached, (laxity, ev.current_capacity, ev.desired_capacity)
                checked += 1
            elif laxity <= -1:
                assert not reached, laxity
                checked += 1
    assert checked > 20


def test_dayahead_prices_follow_the_publication_rule():
    """Prices until midnight are always known; tomorrow's only from 13:00 (day-ahead)."""
    from smartev2g.envs.state import DAYAHEAD_HORIZON

    state = "V2G_profit_max_laxity_dayahead"
    env = make_env(seed=2_000_000, state_function=state, charging_target=0.8)
    obs, _ = env.reset(seed=2_000_000)
    names = observation_names(env, state)
    assert len(obs) == len(names) == 2 + DAYAHEAD_HORIZON + 1 + 20 * 4
    start = env.sim_starting_date
    assert start.hour == 5 and env.timescale == 15
    seen = {}
    for _ in range(40):
        obs = env.unwrapped._get_observation()
        seen[env.current_step] = obs
        env.step(np.zeros(20))
    # 05:00 -> known until midnight = 19 h = 76 steps; the rest is 0.
    o = seen[0]
    n0 = int(o[names.index("n_known")])
    assert n0 == 76
    prices = o[2 : 2 + DAYAHEAD_HORIZON]
    np.testing.assert_allclose(prices[:n0], np.abs(env.charge_prices[0, :n0]))
    assert (prices[n0:] == 0).all()
    # 12:45 (step 31) still only until midnight; 13:00 (step 32) up to the day's end.
    assert int(seen[31][names.index("n_known")]) == 76 - 31
    assert int(seen[32][names.index("n_known")]) == min(DAYAHEAD_HORIZON, 112 - 32)
    # The per-car part is exactly the laxity observation's.
    lax = make_env(seed=2_000_000, state_function="V2G_profit_max_laxity", charging_target=0.8)
    lax_obs, _ = lax.reset(seed=2_000_000)
    np.testing.assert_array_equal(seen[0][-80:], lax_obs[-80:])
