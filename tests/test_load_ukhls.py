"""Tests for the UKHLS panel loader (src/estimation/load_ukhls.py).

These run against the real (git-ignored) parquet when present; if the restricted data is
not available (e.g. CI without the EUL extract), data-dependent tests are skipped while
the pure-logic tests still run on a small synthetic frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.estimation import load_ukhls as lk


def _synthetic_panel(n: int = 200) -> pd.DataFrame:
    """A schema-faithful synthetic panel for logic tests (no microdata)."""
    rng = np.random.default_rng(0)
    waves = rng.integers(1, 12, size=n)
    df = pd.DataFrame(
        {
            "pidp": rng.integers(1, 50, size=n),
            "wave": waves.astype("int8"),
            "hidp": rng.integers(1, 50, size=n),
            "income_net": rng.uniform(0, 5000, n),
            "income_gross": rng.uniform(0, 6000, n),
            "income_labour_net": rng.uniform(0, 4000, n),
            "income_labour_gross": rng.uniform(0, 5000, n),
            "y_benefit": rng.uniform(0, 1000, n),
            "y_market": rng.uniform(0, 4000, n),
            "y_market_is_pretransfer": rng.integers(0, 2, n).astype(bool),
            "health_pcs": rng.uniform(10, 70, n),
            "health_mcs": rng.uniform(10, 70, n),
            "ghq": rng.uniform(0, 36, n),
            "ghq_caseness": rng.uniform(0, 12, n),
            "srh": rng.integers(1, 6, n).astype(float),
            "health_state": rng.integers(0, 3, n).astype(float),
            "disability": rng.integers(0, 2, n).astype(bool),
            "disability_type": rng.integers(0, 4, n).astype("int8"),
            "jbstat": rng.choice([1, 2, 3, 4, 7], n).astype(float),
            "employed": rng.integers(0, 2, n).astype(bool),
            "hours": rng.uniform(0, 60, n),
            "dvage": rng.uniform(16, 90, n),
            "sex": rng.integers(1, 3, n).astype(float),
            "mastat_dv": rng.integers(0, 11, n).astype(float),
            "hiqual_dv": rng.integers(1, 10, n).astype(float),
            "ethn_dv": rng.integers(1, 98, n).astype(float),
            "nkids_dv": np.nan,
            "gor_dv": rng.integers(1, 13, n).astype(float),
            "urban_dv": rng.integers(1, 3, n).astype(float),
            "weight_xsec": rng.uniform(0, 8, n),
            "design_weight": rng.uniform(0, 8, n),
            "weight_long": rng.uniform(0, 5, n),
        }
    )
    return df


# ---- pure-logic tests (always run) ----------------------------------------------------
def test_validate_accepts_synthetic():
    lk.validate_panel(_synthetic_panel())


def test_validate_rejects_sentinel_negatives():
    df = _synthetic_panel()
    df.loc[0, "ghq"] = -9.0  # un-cleaned UKHLS sentinel
    with pytest.raises(ValueError, match="negative"):
        lk.validate_panel(df)


def test_validate_rejects_missing_column():
    df = _synthetic_panel().drop(columns=["income_net"])
    with pytest.raises(ValueError, match="missing expected columns"):
        lk.validate_panel(df)


def test_derived_columns_well_formed():
    df = lk.add_analysis_columns(_synthetic_panel())
    assert df["log_income"].notna().all() and np.isfinite(df["log_income"]).all()
    h = df["health_index"]
    assert ((h >= 0) & (h <= 1)).all()
    assert ((df["wellbeing_ghq"] >= 0) & (df["wellbeing_ghq"] <= lk.GHQ_MAX)).all()
    assert df["region"].notna().all()


# ---- data-dependent tests (skip if restricted parquet absent) -------------------------
@pytest.fixture(scope="module")
def panel():
    if not lk.DEFAULT_PANEL_PATH.exists():
        pytest.skip("restricted UKHLS parquet not available")
    return lk.load_panel()


def test_real_panel_shape(panel):
    assert panel["pidp"].nunique() > 50_000
    assert set(panel["wave"].unique()).issubset(set(lk.WAVE_LETTERS))


def test_income_decomposition_identity(panel):
    # income_net should equal market+benefit for the bulk (extra components only in the tail)
    chk = panel[["income_net", "y_market", "y_benefit"]].dropna()
    diff = (chk["income_net"] - (chk["y_market"] + chk["y_benefit"])).abs()
    assert diff.median() < 1.0
