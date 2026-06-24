"""Shared estimation utilities: lag construction, weighted estimators, param I/O.

All estimation scripts in this package use these helpers so that survey weighting,
clustering, missing-data handling, and parameter provenance are consistent.

PROVENANCE: operates on the UKHLS long panel (waves 1-11, EUL) loaded via
``load_ukhls.load_panel``. Writes only aggregate coefficients to ``params/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

_REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_DIR = _REPO_ROOT / "params"

# UKHLS hiqual_dv -> compact education bands (kept explicit, not magic).
EDUCATION_BANDS: dict[int, str] = {
    1: "degree",
    2: "other_higher",
    3: "a_level",
    4: "gcse",
    5: "other_qual",
    9: "no_qual",
}


# --------------------------------------------------------------------------------------
# Panel shaping
# --------------------------------------------------------------------------------------
def add_lead(df: pd.DataFrame, col: str, *, new: str | None = None) -> pd.DataFrame:
    """Add next-wave value of ``col`` aligned only where the next obs is the consecutive wave.

    The panel is unbalanced; ``shift(-1)`` within person is only a valid t+1 link when the
    following row's wave is exactly ``wave + 1``. Rows without a consecutive follow-up get NaN.
    """
    new = new or f"{col}_next"
    df = df.sort_values(["pidp", "wave"]).copy()
    nxt_val = df.groupby("pidp", observed=True)[col].shift(-1)
    nxt_wave = df.groupby("pidp", observed=True)["wave"].shift(-1)
    consecutive = nxt_wave == df["wave"] + 1
    df[new] = nxt_val.where(consecutive)
    return df


def education_band(df: pd.DataFrame) -> pd.Series:
    """Map ``hiqual_dv`` to compact education bands (categorical)."""
    return df["hiqual_dv"].map(EDUCATION_BANDS).astype("category")


def design_matrix(
    df: pd.DataFrame,
    *,
    outcome: str,
    numeric: list[str],
    categoricals: list[str] | None = None,
    weight: str,
    add_const: bool = True,
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Assemble a listwise-complete (y, X, weights, groups) estimation sample.

    Categoricals are one-hot encoded (first level dropped). Rows with any missing model
    variable or non-positive weight are dropped. ``groups`` is ``pidp`` for clustering.

    Returns
    -------
    (y, X, w, groups)
    """
    categoricals = categoricals or []
    cols = [outcome, *numeric, *categoricals, weight, "pidp"]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[sub[weight] > 0]

    X_parts: list[pd.DataFrame] = [sub[numeric].astype(float)]
    for c in categoricals:
        dummies = pd.get_dummies(sub[c], prefix=c, drop_first=True, dtype=float)
        X_parts.append(dummies)
    X = pd.concat(X_parts, axis=1)
    if add_const:
        X = sm.add_constant(X, has_constant="add")
    y = sub[outcome].astype(float)
    return y, X, sub[weight].astype(float), sub["pidp"]


# --------------------------------------------------------------------------------------
# Weighted estimators with cluster-robust (by person) inference
# --------------------------------------------------------------------------------------
# NOTE on inference: the extract carries individual weights but not PSU/strata, so full
# survey-design SEs are not recoverable. We weight point estimates by the survey weight and
# cluster the covariance by person (pidp), the standard practical approximation. SEs should
# be read as approximate; design effects are not fully propagated. (Caveat surfaced in §9.)


def weighted_ols(y, X, w, groups):
    """Survey-weighted OLS with person-cluster-robust covariance."""
    model = sm.WLS(y, X, weights=w)
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


def weighted_glm(y, X, w, groups, family):
    """Survey-weighted GLM (var_weights) with person-cluster-robust covariance.

    Used for logit (Binomial) employment/take-up models and the fractional-logit health
    model (Papke-Wooldridge: Binomial family with a logit link on a [0,1] outcome).
    """
    model = sm.GLM(y, X, family=family, var_weights=w)
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


# --------------------------------------------------------------------------------------
# Parameter I/O (coefficients are aggregates, safe to track per §4)
# --------------------------------------------------------------------------------------
@dataclass
class ParamSet:
    """A versioned estimated parameter file with provenance."""

    name: str
    spec: str
    outcome: str
    waves: str
    variables: list[str]
    n_obs: int
    n_persons: int
    weight: str
    coefficients: dict[str, float]
    std_errors: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def write(self, params_dir: Path = PARAMS_DIR) -> Path:
        """Write to ``params/<name>.json`` with a generation header."""
        params_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "_generated_by": "src/estimation/estimate.py",
            "_generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "_source": "UKHLS (Understanding Society), EUL, waves 1-11; GOR geography",
            **self.__dict__,
        }
        out = params_dir / f"{self.name}.json"
        out.write_text(json.dumps(payload, indent=2))
        return out


def params_from_fit(
    fit,
    *,
    name: str,
    spec: str,
    outcome: str,
    variables: list[str],
    weight: str,
    n_persons: int,
    diagnostics: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    caveats: list[str] | None = None,
    waves: str = "1-11",
) -> ParamSet:
    """Build a :class:`ParamSet` from a fitted statsmodels result."""
    return ParamSet(
        name=name,
        spec=spec,
        outcome=outcome,
        waves=waves,
        variables=variables,
        n_obs=int(fit.nobs),
        n_persons=n_persons,
        weight=weight,
        coefficients={k: float(v) for k, v in fit.params.items()},
        std_errors={k: float(v) for k, v in fit.bse.items()},
        diagnostics=diagnostics or {},
        notes=notes or [],
        caveats=caveats or [],
    )
