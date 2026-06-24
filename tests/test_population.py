"""Tests for synthetic population synthesis (src/population/synthesize.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.estimation import load_ukhls as lk
from src.population import synthesize as syn


@pytest.fixture(scope="module")
def panel():
    if not lk.DEFAULT_PANEL_PATH.exists():
        pytest.skip("restricted UKHLS parquet not available")
    return lk.load_panel()


def test_population_size_and_columns(panel):
    pop = syn.build_population(panel, n_agents=2000, seed=1)
    assert len(pop) == 2000
    for col in ("agent_id", "y0", "region", "health_index", "employed"):
        assert col in pop.columns
    assert pop["y0"].notna().all()  # baseline income must exist for every agent


def test_population_reproducible(panel):
    a = syn.build_population(panel, n_agents=1000, seed=42)
    b = syn.build_population(panel, n_agents=1000, seed=42)
    pd.testing.assert_frame_equal(a, b)  # one seeded RNG -> identical populations


def test_population_seed_changes_draw(panel):
    a = syn.build_population(panel, n_agents=1000, seed=1)
    b = syn.build_population(panel, n_agents=1000, seed=2)
    assert not a["agent_id"].equals(a["y0"])  # sanity
    assert not np.allclose(a["y0"].to_numpy(), b["y0"].to_numpy())


def test_margins_within_tolerance(panel):
    pop = syn.build_population(panel, n_agents=8000, seed=0)
    rep = syn.validate_margins(pop, panel, tol=0.025)  # raises if any margin off
    assert rep["abs_diff"].max() <= 0.025


def test_ipf_reweight_hits_marginals():
    """IPF capability: a 2x2 seed reweighted to new row/col totals reproduces them."""
    seed = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1], "weight": [1.0, 1.0, 1.0, 1.0]})
    row_tot = pd.Series([30.0, 70.0])  # totals over 'a' = 0,1
    col_tot = pd.Series([40.0, 60.0])  # totals over 'b' = 0,1
    out = syn.ipf_reweight(seed, [(["a"], row_tot), (["b"], col_tot)])
    by_a = out.groupby("a")["weight"].sum().to_numpy()
    by_b = out.groupby("b")["weight"].sum().to_numpy()
    assert np.allclose(by_a, [30, 70], atol=1e-3)
    assert np.allclose(by_b, [40, 60], atol=1e-3)
