"""Estimation layer (T1.3), UKHLS-estimated behavioural & utility models -> params/.

Run: ``uv run python -m src.estimation.estimate`` (writes params/*.json and prints a coefficient summary).

PROVENANCE
----------
Source : UKHLS (Understanding Society), main adult interview, waves 1-11 (a-k), EUL,
         Government Office Region geography. Loaded via ``load_ukhls.load_panel``.
Outputs: aggregate coefficients only (params/*.json), no microdata leaves data/ (§4).

Models
------
1. Utility/wellbeing  : wellbeing_ghq ~ ln(income) + health + employment + demog + FE.
2. Health dynamics    : fractional-logit h_{t+1} on h_t, age, employment, income.
3. Employment trans.  : logit employed_{t+1} on employed_t, health, education, age, region.
4. Earnings           : ln(labour income | employed) ~ education, age, health, region.
5. Take-up / receipt  : logit P(benefit receipt), BASELINE, free params calibrated later.

Identification caveats are surfaced in each ParamSet.caveats (ODD §9), not hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.estimation import _prep
from src.estimation.load_ukhls import WEIGHTS, load_panel

# Scaled age keeps age^2 coefficients on a readable magnitude.
AGE_SCALE = 10.0


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Attach estimation-time columns: education band, scaled age, transition leads."""
    df = df.copy()
    df["education"] = _prep.education_band(df)
    df["sex"] = df["sex"].astype("category")
    df["age10"] = df["dvage"] / AGE_SCALE
    df["age10_sq"] = df["age10"] ** 2
    df["employed_f"] = df["employed"].astype(float)

    # Pre-transfer (market) income for the take-up model: using total net income would
    # include the benefit itself (mechanical positive gradient). Floor/winsorise to a
    # positive band for ln(); flag zero-earners (the most benefit-relevant group) separately.
    pos_m = df.loc[df["y_market"] > 0, "y_market"]
    m_lo, m_hi = float(pos_m.quantile(0.01)), float(pos_m.quantile(0.995))
    df["log_market"] = np.log(df["y_market"].clip(lower=m_lo, upper=m_hi))
    df["no_market_income"] = (df["y_market"] <= 0).astype(float)

    # transition leads (consecutive-wave aligned)
    df = _prep.add_lead(df, "employed_f", new="employed_next")
    df = _prep.add_lead(df, "health_index", new="health_index_next")
    return df


# --------------------------------------------------------------------------------------
# 1. Utility / wellbeing  (the source of beta_y and the EV money scale)
# --------------------------------------------------------------------------------------
def estimate_utility(df: pd.DataFrame) -> _prep.ParamSet:
    """Wellbeing equation v = beta_y*ln(y) + beta_h*health + phi(demog) (ODD §7.7).

    Anchor = reverse-scored GHQ-12 (life satisfaction absent from extract). Pooled
    survey-weighted OLS with region+wave fixed effects, person-clustered SEs. An
    individual fixed-effects variant of beta_y is computed for robustness (stored in
    diagnostics), within-person income variation typically attenuates beta_y.

    Specification: health enters via the physical component (PCS-based
    health_index), not the MCS-laden composite, so beta_health is not inflated by the
    GHQ<->SF-12-MCS construct overlap. EMPLOYMENT is intentionally EXCLUDED: it is a mediator
    of income (employed -> higher income), so conditioning on it biases beta_y downward (it
    flips negative in this sample). Employment's welfare consequence is captured through the
    income channel; its direct non-pecuniary value is left to a cleaner design (deepen list).
    """
    numeric = ["log_income", "health_index", "age10", "age10_sq"]
    cats = ["region", "wave", "education", "sex"]
    y, X, w, g = _prep.design_matrix(
        df,
        outcome="wellbeing_ghq",
        numeric=numeric,
        categoricals=cats,
        weight=WEIGHTS.cross_sectional,
    )
    fit = _prep.weighted_ols(y, X, w, g)
    beta_y = float(fit.params["log_income"])

    # Individual fixed-effects robustness for beta_y (linearmodels PanelOLS).
    fe_beta_y = _fe_income_coef(df)

    ps = _prep.params_from_fit(
        fit,
        name="utility",
        spec="WLS, region+wave FE, cluster(pidp)",
        outcome="wellbeing_ghq (= 36 - GHQ-12 Likert; higher = better)",
        variables=numeric + cats,
        weight=WEIGHTS.cross_sectional,
        n_persons=int(g.nunique()),
        diagnostics={
            "beta_y": beta_y,
            "beta_y_individual_FE": fe_beta_y,
            "rsquared": float(fit.rsquared),
        },
        notes=[
            "beta_y is the marginal utility of log-income; sets the EV/WEVM money scale.",
            f"beta_y (pooled) = {beta_y:.3f}; beta_y (individual FE) = {fe_beta_y:.3f}.",
        ],
        caveats=[
            "Anchor is GHQ-12 (distress, reverse-scored), not life satisfaction (sclfsato absent).",
            "Health = PHYSICAL (PCS) only; employment excluded as an income mediator (de-overlapped spec).",
            "Cross-sectional beta_y mixes between-person differences; FE variant is the within-person check.",
            "Survey SEs approximate: individual weights + person clustering, no PSU/strata.",
        ],
    )
    return ps


def _fe_income_coef(df: pd.DataFrame) -> float:
    """Within-person (entity+time FE) coefficient on log_income, as a beta_y robustness check."""
    from linearmodels.panel import PanelOLS

    cols = [
        "pidp",
        "wave",
        "wellbeing_ghq",
        "log_income",
        "health_index",
        "age10",
        "age10_sq",
        WEIGHTS.cross_sectional,
    ]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[sub[WEIGHTS.cross_sectional] > 0]
    panel = sub.set_index(["pidp", "wave"])
    exog = sm.add_constant(panel[["log_income", "health_index", "age10", "age10_sq"]])
    mod = PanelOLS(
        panel["wellbeing_ghq"],
        exog,
        entity_effects=True,
        time_effects=True,
        weights=panel[WEIGHTS.cross_sectional],
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return float(res.params["log_income"])


# --------------------------------------------------------------------------------------
# 2. Health dynamics  (fractional logit on h in [0,1])
# --------------------------------------------------------------------------------------
def estimate_health_dynamics(df: pd.DataFrame) -> _prep.ParamSet:
    """h_{t+1} ~ logit(h_t, age, employment, income), Papke-Wooldridge fractional logit.

    Service-access -> health effect is NOT estimable here: UKHLS has no direct measure of
    healthcare service *use/access*. So this model captures autonomous dynamics only; the
    access effect size enters the ABM as a CALIBRATION parameter (see caveats / takeup).
    """
    numeric = ["health_index", "age10", "age10_sq", "employed_f", "log_income"]
    y, X, w, g = _prep.design_matrix(
        df,
        outcome="health_index_next",
        numeric=numeric,
        categoricals=None,
        weight=WEIGHTS.longitudinal,
    )
    fit = _prep.weighted_glm(y, X, w, g, family=sm.families.Binomial())
    ps = _prep.params_from_fit(
        fit,
        name="health_dynamics",
        spec="fractional logit (GLM Binomial/logit), cluster(pidp)",
        outcome="health_index_{t+1} in [0,1] (SF-12 PCS/MCS proxy)",
        variables=numeric,
        weight=WEIGHTS.longitudinal,
        n_persons=int(g.nunique()),
        diagnostics={"persistence_coef_h_t": float(fit.params["health_index"])},
        notes=["Strong positive h_t coefficient => health is highly persistent (AR-like)."],
        caveats=[
            "health_index is an SF-6D proxy (PCS/MCS), not a tariff-based utility.",
            "Service-access to health effect not estimated (no access measure in UKHLS); "
            "it is a calibration parameter in the ABM.",
        ],
    )
    return ps


# --------------------------------------------------------------------------------------
# 3. Employment transition  (logit; health is the indirect channel)
# --------------------------------------------------------------------------------------
def estimate_employment(df: pd.DataFrame) -> _prep.ParamSet:
    """Logit P(employed_{t+1}) on employed_t, health, education, age, region, sex (ODD §7.5)."""
    numeric = ["employed_f", "health_index", "age10", "age10_sq"]
    cats = ["education", "region", "sex"]
    y, X, w, g = _prep.design_matrix(
        df,
        outcome="employed_next",
        numeric=numeric,
        categoricals=cats,
        weight=WEIGHTS.longitudinal,
    )
    fit = _prep.weighted_glm(y, X, w, g, family=sm.families.Binomial())
    ps = _prep.params_from_fit(
        fit,
        name="employment",
        spec="logit (GLM Binomial/logit), cluster(pidp)",
        outcome="employed_{t+1} (bool)",
        variables=numeric + cats,
        weight=WEIGHTS.longitudinal,
        n_persons=int(g.nunique()),
        diagnostics={
            "state_dependence_employed_t": float(fit.params["employed_f"]),
            "health_channel_coef": float(fit.params["health_index"]),
        },
        notes=["health_index coef > 0 is the 'better health -> employment' indirect channel."],
        caveats=["Health is endogenous to employment; coefficient is associational, not causal."],
    )
    return ps


# --------------------------------------------------------------------------------------
# 4. Earnings  (level of labour income when employed)
# --------------------------------------------------------------------------------------
def estimate_earnings(df: pd.DataFrame) -> _prep.ParamSet:
    """ln(net labour income | employed, income>0) ~ education, age, health, region, sex (§7.6)."""
    work = df[(df["employed"]) & (df["income_labour_net"] > 0)].copy()
    work["log_labour"] = np.log(work["income_labour_net"])
    numeric = ["age10", "age10_sq", "health_index"]
    cats = ["education", "region", "sex"]
    y, X, w, g = _prep.design_matrix(
        work,
        outcome="log_labour",
        numeric=numeric,
        categoricals=cats,
        weight=WEIGHTS.cross_sectional,
    )
    fit = _prep.weighted_ols(y, X, w, g)
    ps = _prep.params_from_fit(
        fit,
        name="earnings",
        spec="WLS on ln(labour income | employed), cluster(pidp)",
        outcome="ln(income_labour_net) | employed",
        variables=numeric + cats,
        weight=WEIGHTS.cross_sectional,
        n_persons=int(g.nunique()),
        diagnostics={"rsquared": float(fit.rsquared)},
        notes=["Mincer-style: positive returns to education, concave age profile expected."],
        caveats=["Selection into employment not corrected (estimated on workers only)."],
    )
    return ps


# --------------------------------------------------------------------------------------
# 5. Take-up / benefit receipt  (free parameters calibrated to observed receipt)
# --------------------------------------------------------------------------------------
def estimate_takeup(df: pd.DataFrame) -> _prep.ParamSet:
    """Logit P(benefit receipt | income, need, demog) among working age (ODD §7.2).

    Receipt proxy = y_benefit > 0 among 16-64. This is a BASELINE propensity; the
    awareness/take-up free parameters are calibrated to observed receipt rates.
    """
    wa = df[(df["dvage"] >= 16) & (df["dvage"] <= 64)].copy()
    wa["receives"] = (wa["y_benefit"] > 0).astype(float)
    # Predict on PRE-transfer (market) income, not total net income (which embeds the benefit).
    numeric = ["log_market", "no_market_income", "health_index"]
    cats = ["education", "region", "sex"]
    y, X, w, g = _prep.design_matrix(
        wa,
        outcome="receives",
        numeric=numeric,
        categoricals=cats,
        weight=WEIGHTS.cross_sectional,
    )
    fit = _prep.weighted_glm(y, X, w, g, family=sm.families.Binomial())
    base_rate = float(np.average(y, weights=w))
    ps = _prep.params_from_fit(
        fit,
        name="takeup",
        spec="logit (GLM Binomial/logit), cluster(pidp)",
        outcome="benefit receipt (y_benefit>0), ages 16-64",
        variables=numeric + cats,
        weight=WEIGHTS.cross_sectional,
        n_persons=int(g.nunique()),
        diagnostics={
            "weighted_receipt_rate": base_rate,
            "market_income_gradient": float(fit.params["log_market"]),
            "no_market_income_coef": float(fit.params["no_market_income"]),
        },
        notes=[
            f"Baseline weighted receipt rate = {base_rate:.1%}; negative income gradient expected.",
            "Awareness and take-up multipliers are free parameters calibrated to observed receipt.",
        ],
        caveats=[
            "'Receipt' here is broad (any state benefit incl. pensions/child benefit); "
            "narrow to means-tested working-age benefits for sharper targeting."
        ],
    )
    return ps


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------
def main() -> None:
    """Estimate all models, write params/, and print the coefficient summary."""
    df = _prepare(load_panel())
    estimators = [
        ("Utility (wellbeing)", estimate_utility),
        ("Health dynamics", estimate_health_dynamics),
        ("Employment transition", estimate_employment),
        ("Earnings", estimate_earnings),
        ("Take-up / receipt", estimate_takeup),
    ]
    results: list[_prep.ParamSet] = []
    for label, fn in estimators:
        print(f"... estimating: {label}")
        ps = fn(df)
        ps.write()
        results.append(ps)

    print("\n" + "=" * 78 + "\nESTIMATED COEFFICIENTS\n" + "=" * 78)
    for ps in results:
        print(f"\n### {ps.name}  [{ps.spec}]")
        print(f"    outcome: {ps.outcome}")
        print(f"    N={ps.n_obs:,} obs / {ps.n_persons:,} persons; weight={ps.weight}")
        for k, v in ps.diagnostics.items():
            print(f"    * {k}: {v:.4f}" if isinstance(v, float) else f"    * {k}: {v}")
        for c in ps.caveats:
            print(f"    ! caveat: {c}")
    print(f"\nWrote {len(results)} param files to {_prep.PARAMS_DIR}")


if __name__ == "__main__":
    main()
