"""Tests for population dynamics (entry/exit), src/model/demography.py + model wiring."""

from __future__ import annotations

import numpy as np
import pytest

from src.estimation import load_ukhls as lk
from src.model.config import PolicyConfig
from src.model.demography import mortality_prob
from src.model.model import SocialServicesModel
from src.model.params import ModelParams
from src.population import synthesize as syn


@pytest.fixture(scope="module")
def setup():
    if not lk.DEFAULT_PANEL_PATH.exists():
        pytest.skip("restricted UKHLS parquet not available")
    return lk.load_panel(), ModelParams.load()


# ---- mortality schedule (pure) ----
def test_mortality_prob_increases_with_age():
    ages = [20, 40, 60, 80, 95]
    qs = [mortality_prob(a) for a in ages]
    assert all(qs[i] < qs[i + 1] for i in range(len(qs) - 1))
    assert all(0.0 <= q <= 1.0 for q in qs)


def test_mortality_multiplier_scales():
    assert mortality_prob(70, multiplier=2.0) == pytest.approx(2 * mortality_prob(70))


# ---- model behaviour ----
def _run(panel, params, *, dynamics, seed=0, n=1500, horizon=10):
    cfg = PolicyConfig(n_agents=n, horizon=horizon, population_dynamics=dynamics)
    pop = syn.build_population(panel, n_agents=n, seed=seed)
    return SocialServicesModel(pop, params, cfg, seed=seed).run(), pop


def test_dynamics_off_is_closed_cohort(setup):
    panel, params = setup
    df, _ = _run(panel, params, dynamics=False, n=1500)
    assert df["agent_id"].nunique() == 1500  # no entrants
    assert df.groupby("tick").size().nunique() == 1


def test_population_stationary_with_dynamics(setup):
    panel, params = setup
    df, _ = _run(panel, params, dynamics=True, n=1500)
    per_tick = df.groupby("tick").size()
    assert per_tick.nunique() == 1 and per_tick.iloc[0] == 1500  # N held constant


def test_turnover_occurs(setup):
    panel, params = setup
    df, _ = _run(panel, params, dynamics=True, n=1500, horizon=10)
    assert df["agent_id"].nunique() > 1500  # entrants joined
    first_tick = set(df[df["tick"] == 0]["agent_id"])
    last_tick = set(df[df["tick"] == df["tick"].max()]["agent_id"])
    assert len(first_tick - last_tick) > 0  # some initial agents exited (died)


def test_dynamics_reproducible(setup):
    panel, params = setup
    a, _ = _run(panel, params, dynamics=True, seed=11)
    b, _ = _run(panel, params, dynamics=True, seed=11)
    assert np.allclose(a["utility"].to_numpy(), b["utility"].to_numpy())


def test_matched_pair_aligns_with_dynamics(setup):
    panel, params = setup
    from src.welfare.runner import compute_welfare, matched_pair

    cfg = PolicyConfig(n_agents=1500, horizon=10, population_dynamics=True, is_awareness=1.5)
    pf, pcf = matched_pair(panel, params, cfg, service="income_support", seed=0)
    res = compute_welfare(pf, pcf, params=params, cfg=cfg, service="income_support")
    assert not res["ev_df"]["EV"].isna().any()  # every matched agent gets a finite EV
    assert res["wevm"][0.0] > 0  # income support has positive value
