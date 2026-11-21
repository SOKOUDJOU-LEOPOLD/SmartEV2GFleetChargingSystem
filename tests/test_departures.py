"""Tests for hidden early departures (ADR-0002, envs/make_env.py ProjectEV2Gym)."""

import numpy as np
import pytest

from smartev2g.agents.baselines import RandomPolicy
from smartev2g.envs.departure_models import SyntheticDepartureModel
from smartev2g.envs.make_env import make_env, observation_names
from smartev2g.eval.evaluate import evaluate_policy

DAY = 1_000_000
LAXITY = "V2G_profit_max_laxity"
REWARD = {
    "reward_function": "squared_shortfall",
    "reward_kwargs": {"lambda_dep": 500.0, "lambda_over": 0.1},
}


def play(env, day, action_seed=0):
    """Random actions for a whole day; yields (obs before step, step number) and returns info."""
    obs, _ = env.reset(seed=day)
    rng = np.random.default_rng(action_seed)
    done, history = False, []
    while not done:
        history.append(
            (obs, env.current_step, [cs.evs_connected[0] for cs in env.charging_stations])
        )
        obs, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 20))
        done = terminated or truncated
    return history


def test_no_early_departures_reproduces_the_normal_env():
    """p_early = 0 runs all the new code but must give exactly the same day."""
    plain = make_env(seed=DAY, charging_target=0.8, **REWARD)
    hidden = make_env(seed=DAY, charging_target=0.8, departures={"p_early": 0.0}, **REWARD)
    for (o1, t1, _), (o2, t2, _) in zip(play(plain, DAY), play(hidden, DAY), strict=True):
        np.testing.assert_array_equal(o1, o2)
        assert t1 == t2
    assert hidden.departure_plan == {}


def test_real_departure_never_in_observation_and_car_leaves_exactly_then():
    env = make_env(seed=DAY, state_function=LAXITY, departures={"p_early": 0.5}, **REWARD)
    names = observation_names(env, LAXITY)
    history = play(env, DAY)
    plan = env.departure_plan
    assert len(plan) >= 3  # enough early leavers to test
    seen_early = 0
    for obs, t, cars in history:
        for k, ev in enumerate(cars):
            if ev is None:
                continue
            key = (ev.location, ev.time_of_arrival)
            announced = plan[key][0] if key in plan else ev.time_of_departure
            # The agent is told the ANNOUNCED time, never the real one.
            assert obs[names.index(f"t_dep_{k}")] == announced - t
            if key in plan:
                seen_early += 1
                assert t <= plan[key][1]  # still plugged in only up to its real departure
    assert seen_early > 0
    # Every early car was removed exactly at its real step (EV2Gym records it there).
    for ev in env.EVs:
        key = (ev.location, ev.time_of_arrival)
        if key in plan:
            announced, real = plan[key]
            assert ev.time_of_departure == real
            assert ev.time_of_arrival < real < announced


def test_max_energy_afap_refreshed_for_early_leavers():
    """EV2Gym caches max_energy_AFAP at plug-in from the announced time (ADR-0002)."""
    env = make_env(seed=DAY, departures={"p_early": 0.5}, **REWARD)
    play(env, DAY)
    early = [ev for ev in env.EVs if (ev.location, ev.time_of_arrival) in env.departure_plan]
    assert early
    for ev in early:
        cached = ev.max_energy_AFAP
        ev.calculate_max_energy_with_AFAP(env.charging_stations[ev.location].get_max_power())
        assert ev.max_energy_AFAP == pytest.approx(cached)  # already based on the real time


def test_same_cars_for_every_p_early_and_nested_early_sets():
    """Only departures change with p_early; and a car early at p=0.1 is also early at
    p=0.3 (same random numbers), so worlds differ only by the extra early leavers."""
    low = make_env(seed=DAY, departures={"p_early": 0.1})
    high = make_env(seed=DAY, departures={"p_early": 0.3})
    for day in range(DAY, DAY + 5):
        low.reset(seed=day)
        high.reset(seed=day)
        for a, b in zip(low.EVs_profiles, high.EVs_profiles, strict=True):
            assert (a.time_of_arrival, a.time_of_departure, a.location) == (
                b.time_of_arrival,
                b.time_of_departure,
                b.location,
            )
        assert set(low.departure_plan) <= set(high.departure_plan)
        for key in low.departure_plan:
            assert low.departure_plan[key] == high.departure_plan[key]


def test_share_of_early_leavers_matches_p_early():
    model = SyntheticDepartureModel(p_early=0.3)
    rng = np.random.default_rng(0)
    draws = [model.sample(10, 50, rng) for _ in range(20_000)]
    share = np.mean([d is not None for d in draws])
    assert share == pytest.approx(0.3, abs=0.015)
    early = np.array([d for d in draws if d is not None])
    cut = (50 - early) / 40  # fraction of the announced stay that was cut
    assert cut.min() >= 0.1 - 1e-9 and cut.max() <= 0.5 + 1e-9
    assert ((10 < early) & (early < 50)).all()


def test_departure_penalty_covers_early_leavers():
    """Early leavers are penalised like any other departing car (proposal §9)."""
    kw = {**REWARD, "charging_target": 0.8, "departures": {"p_early": 0.5}}
    episodes, cars = evaluate_policy(RandomPolicy(seed=0), [DAY, DAY + 1], kw)
    for seed, ep in episodes.groupby("seed"):
        day = cars[(cars["seed"] == seed) & (cars["departure_step"] < 112)]
        expected = 500.0 * (day["shortfall"] ** 2).sum()
        assert ep["reward_departure_penalty"].iloc[0] == pytest.approx(expected)
    assert cars["early"].sum() > 0
    assert episodes["n_early"].sum() == cars["early"].sum()


def test_invalid_departure_settings_rejected():
    with pytest.raises(ValueError):
        SyntheticDepartureModel(p_early=1.5)
    with pytest.raises(ValueError):
        SyntheticDepartureModel(p_early=0.1, cut_min=0.6, cut_max=0.5)
