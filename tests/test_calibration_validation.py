"""Tests for ABC posterior calibration and out-of-sample validation."""

from __future__ import annotations

import numpy as np
import pytest

from src.estimation import load_ukhls as lk
from src.experiments import abc_calibrate as abc
from src.experiments import validate as val
from src.model.config import PolicyConfig
from src.model.params import ModelParams


@pytest.fixture(scope="module")
def setup():
    if not lk.DEFAULT_PANEL_PATH.exists():
        pytest.skip("restricted UKHLS parquet not available")
    return lk.load_panel(), ModelParams.load()


# ---- ABC posterior (pure-logic pieces; the SMC run itself is exercised in the runner) ----
def test_posterior_resample_respects_weights():
    post = abc.Posterior(
        param="is_awareness",
        samples=np.array([0.5, 1.5]),
        weights=np.array([0.0, 1.0]),  # all mass on the second particle
        target_rate=0.358,
        posterior_mean=1.5,
        posterior_sd=0.0,
        cred_interval=(1.5, 1.5),
        epsilon=0.01,
        n_populations=3,
    )
    draws = post.resample(200, np.random.default_rng(0))
    assert np.allclose(draws, 1.5)  # zero-weight particle never drawn


def test_weighted_quantiles_monotone():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4)
    qs = abc._weighted_quantiles(v, w, [0.025, 0.5, 0.975])
    assert qs[0] < qs[1] < qs[2]


# ---- validation moments (logic) ----
def test_simulated_moments_keys_match_observed(setup):
    panel, params = setup
    from src.model.model import SocialServicesModel
    from src.population.synthesize import build_population

    cfg = PolicyConfig(n_agents=600, horizon=3)
    pop = build_population(panel, n_agents=600, seed=0)
    sim = SocialServicesModel(pop, params, cfg, seed=0).run()
    sm = val.simulated_moments(sim, tick=2)
    om = val.observed_moments(panel, wave=2)
    assert set(sm) == set(om)
    assert 0.0 <= sm["employment_rate"] <= 1.0
    assert 0.0 <= sm["mean_health"] <= 1.0


def test_forward_validation_runs_and_is_reasonable(setup):
    panel, params = setup
    cfg = PolicyConfig(n_agents=1200, horizon=10, is_awareness=1.56)
    v = val.forward_validation(panel, params, cfg, base_wave=1, target_wave=4, seeds=(0, 1))
    row = v.table.set_index("moment")
    # employment + health should track observed reasonably out-of-sample
    assert row.loc["employment_rate", "rel_diff"] < 0.15
    assert row.loc["mean_health", "rel_diff"] < 0.15


def test_reform_direction_matches_austerity_sign(setup):
    panel, params = setup
    cfg = PolicyConfig(n_agents=1200, horizon=8, is_awareness=1.56)
    r = val.reform_direction_check(panel, params, cfg, eligibility_cut=0.2, seeds=(0, 1))
    # tightening eligibility must reduce simulated receipt (and observed austerity fell too)
    assert r.simulated_change <= 0
    assert r.direction_agrees
