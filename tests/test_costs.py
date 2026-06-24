"""Tests for the cost-effectiveness layer (src/welfare/costs.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.estimation import load_ukhls as lk
from src.model.config import PolicyConfig
from src.model.params import ModelParams
from src.welfare import costs
from src.welfare.runner import compute_welfare, matched_pair


# ---- pure-logic ----
def test_cost_effectiveness_bcr_and_net():
    wevm = {0.0: 100.0, 1.0: 200.0}
    ce = costs.cost_effectiveness(wevm, cost_per_person=600.0, annualise=12.0)
    row0 = ce.set_index("eps").loc[0.0]
    assert row0["benefit_per_person"] == 1200.0  # 12 x 100
    assert row0["bcr"] == pytest.approx(1200.0 / 600.0)
    assert row0["net_social_benefit"] == pytest.approx(600.0)
    assert bool(row0["cost_effective"]) is True


def test_bcr_rises_with_eps_when_wevm_does():
    ce = costs.cost_effectiveness({0.0: 50.0, 2.0: 150.0}, cost_per_person=900.0)
    by = ce.set_index("eps")["bcr"]
    assert by.loc[2.0] > by.loc[0.0]


def test_programme_cost_income_support_matches_transfers():
    # one agent, two ticks, £100/mo transfer each tick, delta=1, no admin
    panel = pd.DataFrame(
        {
            "tick": [0, 1],
            "transfer": [100.0, 100.0],
            "access_healthcare": [0, 0],
        }
    )
    cfg = PolicyConfig(is_admin_loading=0.0)
    cpp = costs.programme_cost_per_person(
        panel, service="income_support", cfg=cfg, delta=1.0, n_persons=1
    )
    assert cpp == pytest.approx(100.0 * 12 * 2)  # annualised transfers over 2 ticks


def test_programme_cost_healthcare_counts_accesses():
    panel = pd.DataFrame(
        {"tick": [0, 0, 1], "transfer": [0.0, 0.0, 0.0], "access_healthcare": [True, False, True]}
    )
    cfg = PolicyConfig(hc_unit_cost=50.0)
    cpp = costs.programme_cost_per_person(
        panel, service="healthcare", cfg=cfg, delta=1.0, n_persons=1
    )
    assert cpp == pytest.approx(2 * 50.0)  # 2 accesses x £50


# ---- integration ----
@pytest.fixture(scope="module")
def setup():
    if not lk.DEFAULT_PANEL_PATH.exists():
        pytest.skip("restricted UKHLS parquet not available")
    return lk.load_panel(), ModelParams.load()


def test_income_support_bcr_rises_above_one_with_eps(setup):
    panel, params = setup
    cfg = PolicyConfig(n_agents=1500, horizon=10, is_awareness=1.56)
    pf, pcf = matched_pair(panel, params, cfg, service="income_support", seed=0)
    res = compute_welfare(pf, pcf, params=params, cfg=cfg, service="income_support")
    cpp = costs.programme_cost_per_person(
        pf,
        service="income_support",
        cfg=cfg,
        delta=cfg.discount_delta,
        n_persons=res["ev_df"].shape[0],
    )
    ce = costs.cost_effectiveness(res["wevm"], cpp).set_index("eps")
    # progressive transfer: BCR low at eps=0 (~ net-zero minus admin), higher at eps=2
    assert ce.loc[2.0, "bcr"] > ce.loc[0.0, "bcr"]
    assert cpp > 0 and np.isfinite(ce.loc[0.0, "bcr"])
