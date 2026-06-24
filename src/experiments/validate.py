"""Out-of-sample validation (ODD §8.3), does the model reproduce UKHLS it wasn't fit to.

Two checks:

1. **Forward cohort validation** (primary, rigorous). Initialise the model on the wave-1
   cohort, simulate forward k years, and compare simulated moments at the final tick to the
   SAME cohort's ACTUAL outcomes at wave 1+k in UKHLS, on moments NOT used in calibration
   (employment rate, mean health, income percentiles). Calibration only targeted the wave-1
   receipt LEVEL, so the later-wave trajectory is genuinely held out.

2. **Reform direction check** (secondary, illustrative). Simulate an eligibility tightening
   and compare the SIGN of the implied change in benefit receipt to the observed change
   across a UK austerity window. Explicitly confounded (cycle/composition), a direction
   sanity check, not a clean difference-in-differences.

Caveat carried throughout: the closed-cohort configuration has no population entry/exit, so it tracks a closed
cohort; we compare against the surviving cohort to keep this apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.model.config import PolicyConfig
from src.model.model import SocialServicesModel
from src.model.params import ModelParams
from src.population.synthesize import build_population

WORKING_AGE = (16, 64)


def _wa(df: pd.DataFrame, age_col: str) -> pd.DataFrame:
    return df[(df[age_col] >= WORKING_AGE[0]) & (df[age_col] <= WORKING_AGE[1])]


def observed_moments(panel: pd.DataFrame, wave: int, *, cohort: set[int] | None = None) -> dict:
    """UKHLS cross-sectional moments at ``wave`` (optionally restricted to a cohort)."""
    df = panel[panel["wave"] == wave]
    if cohort is not None:
        df = df[df["pidp"].isin(cohort)]
    wa = _wa(df, "dvage")
    return {
        "employment_rate": float(wa["employed"].mean()),
        "mean_health": float(df["health_index"].mean()),
        "receipt_rate": float((wa["y_benefit"] > 0).mean()),
        "median_income": float(df["income_net"].median()),
        "p90_p10_income": float(
            df["income_net"].quantile(0.9) / max(df["income_net"].quantile(0.1), 1.0)
        ),
    }


def simulated_moments(sim_panel: pd.DataFrame, tick: int) -> dict:
    """Compute the same moments from the simulated panel at a given tick."""
    df = sim_panel[sim_panel["tick"] == tick]
    wa = _wa(df, "age")
    return {
        "employment_rate": float(wa["employed"].mean()),
        "mean_health": float(df["health"].mean()),
        "receipt_rate": float(wa["access_income_support"].mean()),
        "median_income": float(df["income"].median()),
        "p90_p10_income": float(df["income"].quantile(0.9) / max(df["income"].quantile(0.1), 1.0)),
    }


@dataclass
class ValidationResult:
    """Observed-vs-simulated comparison for the forward cohort validation."""

    base_wave: int
    target_wave: int
    table: pd.DataFrame  # moment, observed, simulated, abs_diff, rel_diff
    notes: list[str] = field(default_factory=list)


def forward_validation(
    panel: pd.DataFrame,
    params: ModelParams,
    cfg: PolicyConfig,
    *,
    base_wave: int = 1,
    target_wave: int = 5,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> ValidationResult:
    """Compare the simulated wave-1 cohort, k years on, to its actual UKHLS outcomes.

    Moments are averaged over ``seeds``; the observed side is restricted to the cohort
    present at BOTH ``base_wave`` and ``target_wave`` (the closed cohort the model tracks).
    """
    horizon = target_wave - base_wave
    base_ids = set(panel.loc[panel["wave"] == base_wave, "pidp"])
    target_ids = set(panel.loc[panel["wave"] == target_wave, "pidp"])
    cohort = base_ids & target_ids
    obs = observed_moments(panel, target_wave, cohort=cohort)

    sims: list[dict] = []
    for s in seeds:
        pop = build_population(panel, n_agents=cfg.n_agents, seed=s, base_wave=base_wave)
        sim = SocialServicesModel(pop, params, cfg, seed=s).run(horizon=horizon)
        sims.append(simulated_moments(sim, tick=horizon - 1))
    sim_mean = {k: float(np.mean([m[k] for m in sims])) for k in obs}

    rows = []
    for k in obs:
        o, m = obs[k], sim_mean[k]
        rows.append(
            {
                "moment": k,
                "observed": o,
                "simulated": m,
                "abs_diff": abs(o - m),
                "rel_diff": abs(o - m) / abs(o) if o else np.nan,
            }
        )
    return ValidationResult(
        base_wave=base_wave,
        target_wave=target_wave,
        table=pd.DataFrame(rows),
        notes=[
            f"Cohort present in both wave {base_wave} and {target_wave}: n={len(cohort):,}.",
            "Calibration targeted only the wave-1 receipt LEVEL; later-wave moments are held out.",
            "the closed-cohort configuration has no entry or exit; comparison is on the surviving cohort.",
        ],
    )


@dataclass
class ReformCheck:
    """Illustrative reform direction check."""

    observed_change: float
    simulated_change: float
    direction_agrees: bool
    moment: str
    notes: list[str] = field(default_factory=list)


def reform_direction_check(
    panel: pd.DataFrame,
    params: ModelParams,
    cfg: PolicyConfig,
    *,
    pre_waves: tuple[int, ...] = (2, 3),
    post_waves: tuple[int, ...] = (7, 8),
    eligibility_cut: float = 0.15,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> ReformCheck:
    """Compare the SIGN of a simulated eligibility tightening to an observed austerity window.

    Observed moment: change in working-age benefit-receipt rate between ``pre_waves`` and
    ``post_waves`` (the ~2013-2016 UK benefit freeze/cuts span these UKHLS waves). Simulated:
    tighten the income-support eligibility threshold by ``eligibility_cut`` (the model lever
    that maps to the receipt RATE, generosity changes the transfer size, not who receives).

    ILLUSTRATIVE ONLY: the observed change is confounded by the business cycle and sample
    composition; this is a direction sanity check, not a clean policy-effect estimate.
    """

    def _obs_receipt(waves: tuple[int, ...]) -> float:
        wa = _wa(panel[panel["wave"].isin(waves)], "dvage")
        return float((wa["y_benefit"] > 0).mean())

    observed_change = _obs_receipt(post_waves) - _obs_receipt(pre_waves)

    def _sim_receipt(threshold: float) -> float:
        c = cfg.with_(is_eligibility_income=threshold, income_support_on=True)
        rates = []
        for s in seeds:
            pop = build_population(panel, n_agents=c.n_agents, seed=s, base_wave=cfg.base_wave)
            sim = SocialServicesModel(pop, params, c, seed=s).run()
            rates.append(float(_wa(sim, "age")["access_income_support"].mean()))
        return float(np.mean(rates))

    base_thr = cfg.is_eligibility_income
    simulated_change = _sim_receipt(base_thr * (1 - eligibility_cut)) - _sim_receipt(base_thr)
    agrees = bool(
        np.sign(observed_change) == np.sign(simulated_change) or abs(observed_change) < 1e-3
    )
    return ReformCheck(
        observed_change=observed_change,
        simulated_change=simulated_change,
        direction_agrees=agrees,
        moment="working-age benefit-receipt rate",
        notes=[
            f"Austerity window: waves {pre_waves} -> {post_waves}; eligibility tightened "
            f"{eligibility_cut:.0%}.",
            "ILLUSTRATIVE: observed change is confounded (cycle, composition); direction check only.",
        ],
    )
