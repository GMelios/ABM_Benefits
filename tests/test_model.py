"""Invariant and reproducibility tests for the ABM."""

from __future__ import annotations

import numpy as np
import pytest

from src.estimation import load_ukhls as lk
from src.model.config import PolicyConfig
from src.model.model import SocialServicesModel
from src.model.params import ModelParams
from src.population import synthesize as syn


@pytest.fixture(scope="module")
def setup():
    if not lk.DEFAULT_PANEL_PATH.exists():
        pytest.skip("restricted UKHLS parquet not available")
    panel = lk.load_panel()
    params = ModelParams.load()
    return panel, params


def _run(panel, params, *, seed=0, n=800, **cfg_over):
    cfg = PolicyConfig(n_agents=n, horizon=8, **cfg_over)
    pop = syn.build_population(panel, n_agents=n, seed=seed)
    m = SocialServicesModel(pop, params, cfg, seed=seed)
    return m.run(), m


def test_population_conserved(setup):
    panel, params = setup
    df, _ = _run(panel, params, n=800)
    per_tick = df.groupby("tick").size()
    assert per_tick.nunique() == 1 and per_tick.iloc[0] == 800


def test_state_bounds(setup):
    panel, params = setup
    df, _ = _run(panel, params)
    assert df["health"].between(0.0, 1.0).all()
    assert (df["income"] >= 0).all()
    assert (df["y_market"] >= 0).all() and (df["y_benefit"] >= 0).all()


def test_no_negative_capacity(setup):
    panel, params = setup
    _, m = _run(panel, params)
    assert all(p.capacity >= 0 and p.waiting_list >= 0 for p in m.providers)


def test_reproducible_same_seed(setup):
    panel, params = setup
    df1, _ = _run(panel, params, seed=7)
    df2, _ = _run(panel, params, seed=7)
    # fixed-seed regression: identical trajectories
    assert np.allclose(df1["utility"].to_numpy(), df2["utility"].to_numpy())
    assert np.allclose(df1["income"].to_numpy(), df2["income"].to_numpy())


def test_service_on_changes_outcomes(setup):
    panel, params = setup
    # healthcare ON should raise mean health vs OFF (matched seed/population)
    on, _ = _run(
        panel,
        params,
        seed=3,
        healthcare_on=True,
        hc_awareness=1.0,
        hc_capacity_per_1000=200.0,
        hc_access_health_effect=0.05,
    )
    off, _ = _run(panel, params, seed=3, healthcare_on=False)
    assert on["health"].mean() > off["health"].mean()


def test_income_support_raises_low_incomes(setup):
    panel, params = setup
    on, _ = _run(panel, params, seed=5, income_support_on=True, is_awareness=2.0)
    off, _ = _run(panel, params, seed=5, income_support_on=False)
    # the transfer raises net income; some agents must realise income-support access
    assert on["income"].mean() > off["income"].mean()
    assert on["access_income_support"].mean() > 0.0
