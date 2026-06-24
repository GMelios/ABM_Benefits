"""Tests for the health->mortality channel + life-extension valuation."""

from __future__ import annotations

import pytest

from src.estimation import load_ukhls as lk
from src.model.config import PolicyConfig
from src.model.demography import mortality_prob
from src.model.params import ModelParams
from src.welfare import welfare as wf
from src.welfare.runner import compute_welfare, matched_pair


@pytest.fixture(scope="module")
def setup():
    if not lk.DEFAULT_PANEL_PATH.exists():
        pytest.skip("restricted UKHLS parquet not available")
    return lk.load_panel(), ModelParams.load()


# ---- mortality schedule with health (pure) ----
def test_health_lowers_mortality():
    sick = mortality_prob(70, health=0.1, hazard_ratio=4.0)
    healthy = mortality_prob(70, health=0.9, hazard_ratio=4.0)
    assert sick > healthy
    # hazard_ratio=1 recovers the age-only baseline (health ignored)
    assert mortality_prob(70, health=0.1, hazard_ratio=1.0) == mortality_prob(70)


def _cfg(**kw):
    base = dict(
        n_agents=1500,
        horizon=12,
        population_dynamics=True,
        health_affects_mortality=True,
        health_mortality_hr=4.0,
        hc_awareness=1.0,
        hc_capacity_per_1000=300.0,
        hc_access_health_effect=0.05,
    )
    base.update(kw)
    return PolicyConfig(**base)


def test_healthcare_reduces_deaths_monotonically(setup):
    panel, params = setup
    pf, pcf = matched_pair(panel, params, _cfg(), service="healthcare", seed=0)
    cohort = set(pf.loc[pf["tick"] == 0, "agent_id"])
    ft = pf["tick"].max()
    deaths_f = len(cohort - set(pf[(pf["tick"] == ft) & pf["agent_id"].isin(cohort)]["agent_id"]))
    deaths_cf = len(
        cohort - set(pcf[(pcf["tick"] == ft) & pcf["agent_id"].isin(cohort)]["agent_id"])
    )
    # under CRN, factual (with care) survival >= counterfactual, agent-by-agent
    assert deaths_f <= deaths_cf


def test_life_extension_nonnegative_and_present(setup):
    panel, params = setup
    pf, pcf = matched_pair(panel, params, _cfg(), service="healthcare", seed=0)
    le = wf.life_extension(pf, pcf, beta_y=params.beta_y, delta=0.97)
    assert (le["life_years_gained"] >= -1e-9).all()  # monotone: nobody loses life-years
    assert le["life_years_gained"].sum() > 0  # care saves some life-years
    assert (le["life_extension_ev"] >= -1e-9).all()


def test_compute_welfare_adds_mortality_components(setup):
    panel, params = setup
    cfg = _cfg()
    pf, pcf = matched_pair(panel, params, cfg, service="healthcare", seed=0)
    res = compute_welfare(pf, pcf, params=params, cfg=cfg, service="healthcare")
    assert "wevm_life_extension" in res and "life_years_gained_mean" in res
    assert res["wevm_life_extension"][0.0] > 0  # net-benefit life-extension value is positive
