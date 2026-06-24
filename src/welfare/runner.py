"""Matched-pair welfare runner (ODD §7.8 operational requirement).

Runs the ABM as factual (service ON) vs counterfactual (service OFF) with identical seed
and initial population, differing only in one service, and aggregates the welfare metrics
over N seeded replications as means with confidence intervals (never a single run).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.model.config import PolicyConfig
from src.model.model import SocialServicesModel
from src.model.params import ModelParams
from src.population.synthesize import build_population
from src.welfare import welfare as wf


def matched_pair(
    panel: pd.DataFrame,
    params: ModelParams,
    cfg: PolicyConfig,
    *,
    service: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the factual and counterfactual arms for one seed (identical population + seed).

    The population is built once and shared; both arms use the same model seed so per-agent
    common-random-number streams align and the only difference is ``service`` on/off.
    """
    pop = build_population(panel, n_agents=cfg.n_agents, seed=seed, base_wave=cfg.base_wave)
    panel_f = SocialServicesModel(pop, params, cfg, seed=seed).run()
    panel_cf = SocialServicesModel(pop, params, cfg.counterfactual(service), seed=seed).run()
    return panel_f, panel_cf


def _coverage_by_region(panel_f: pd.DataFrame, service: str) -> pd.DataFrame:
    """Service coverage and unmet need by region, from the factual panel (ODD §4)."""
    df = panel_f
    if service == "healthcare":
        need = df[df["need_healthcare"] > 0]
        g = need.groupby("region")
        rep = pd.DataFrame(
            {
                "coverage": g["access_healthcare"].mean(),
                "unmet_need_rate": 1.0 - g["access_healthcare"].mean(),
                "n_with_need": g.size(),
            }
        )
    else:  # income_support
        g = df.groupby("region")
        rep = pd.DataFrame(
            {
                "access_rate": g["access_income_support"].mean(),
                "mean_benefit": g["y_benefit"].mean(),
                "n": g.size(),
            }
        )
    return rep.reset_index()


def compute_welfare(
    panel_f: pd.DataFrame,
    panel_cf: pd.DataFrame,
    *,
    params: ModelParams,
    cfg: PolicyConfig,
    service: str,
) -> dict:
    """All welfare metrics for one matched pair: EV, WEVM grid, Atkinson, coverage.

    When ``cfg.health_affects_mortality`` is set, survival diverges across arms, so the
    within-life EV is computed on the INITIAL COHORT (matched ids) and a separate
    life-extension component (linear money-metric) is added.
    """
    pf, pcf = panel_f, panel_cf
    if cfg.health_affects_mortality:
        cohort = set(panel_f.loc[panel_f["tick"] == 0, "agent_id"])
        pf = panel_f[panel_f["agent_id"].isin(cohort)]
        pcf = panel_cf[panel_cf["agent_id"].isin(cohort)]

    ev_df = wf.equivalent_variation(pf, pcf, beta_y=params.beta_y, delta=cfg.discount_delta)
    inc_before = pcf.groupby("agent_id")["income"].mean()  # Atkinson before (counterfactual)
    inc_after = pf.groupby("agent_id")["income"].mean()  # ...after (factual)
    out = {
        "ev_df": ev_df,
        "wevm": wf.wevm_grid(ev_df["EV"], ev_df["y0"]),
        "wevm_linear": wf.wevm_grid(ev_df["EV_linear"], ev_df["y0"]),
        "atkinson_before": {e: wf.atkinson(inc_before, e) for e in wf.EPS_GRID if e > 0},
        "atkinson_after": {e: wf.atkinson(inc_after, e) for e in wf.EPS_GRID if e > 0},
        "coverage": _coverage_by_region(panel_f, service),
    }
    if cfg.health_affects_mortality:
        le = wf.life_extension(panel_f, panel_cf, beta_y=params.beta_y, delta=cfg.discount_delta)
        out["life_extension_df"] = le
        out["wevm_life_extension"] = wf.wevm_grid(le["life_extension_ev"], le["y0"])
        out["life_years_gained_mean"] = float(le["life_years_gained"].mean())
    return out


@dataclass
class WelfareResult:
    """Aggregated welfare across N seeds: means with 95% CIs."""

    service: str
    n_seeds: int
    wevm_summary: pd.DataFrame  # eps, mean, ci_lo, ci_hi (closed-form EV)
    wevm_linear_summary: pd.DataFrame  # eps, mean, ci_lo, ci_hi (linear money-metric)
    wevm_by_seed: pd.DataFrame  # seed x eps (closed form)
    atkinson_summary: pd.DataFrame  # eps, before, after (means over seeds)
    coverage: pd.DataFrame  # region coverage (mean over seeds)
    ev_pooled: pd.DataFrame  # pooled per-agent EV across seeds (for plots)
    subgroup: pd.DataFrame  # priority-mass decomposition at eps=1
    notes: list[str] = field(default_factory=list)


def _ci(values: np.ndarray) -> tuple[float, float, float]:
    """Mean and 95% CI (t-based) of a sample of seed-level estimates."""
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, m, m
    se = stats.sem(v)
    h = se * stats.t.ppf(0.975, len(v) - 1)
    return m, m - h, m + h


def evaluate_service(
    panel: pd.DataFrame,
    params: ModelParams,
    cfg: PolicyConfig,
    *,
    service: str,
    seeds: list[int],
) -> WelfareResult:
    """Evaluate one service over ``seeds`` and aggregate to means + 95% CIs."""
    per_seed_wevm: list[dict] = []
    per_seed_wevm_lin: list[dict] = []
    atk_before: list[dict] = []
    atk_after: list[dict] = []
    coverages: list[pd.DataFrame] = []
    ev_frames: list[pd.DataFrame] = []

    for s in seeds:
        pf, pcf = matched_pair(panel, params, cfg, service=service, seed=s)
        res = compute_welfare(pf, pcf, params=params, cfg=cfg, service=service)
        per_seed_wevm.append({"seed": s, **res["wevm"]})
        per_seed_wevm_lin.append({"seed": s, **res["wevm_linear"]})
        atk_before.append(res["atkinson_before"])
        atk_after.append(res["atkinson_after"])
        coverages.append(res["coverage"])
        ev_frames.append(res["ev_df"].assign(seed=s))

    by_seed = pd.DataFrame(per_seed_wevm)
    by_seed_lin = pd.DataFrame(per_seed_wevm_lin)

    def _summary(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for eps in wf.EPS_GRID:
            m, lo, hi = _ci(frame[eps].to_numpy())
            rows.append({"eps": eps, "wevm_mean": m, "ci_lo": lo, "ci_hi": hi})
        return pd.DataFrame(rows)

    atk = pd.DataFrame(
        {
            "eps": [e for e in wf.EPS_GRID if e > 0],
            "atkinson_before": [np.mean([a[e] for a in atk_before]) for e in wf.EPS_GRID if e > 0],
            "atkinson_after": [np.mean([a[e] for a in atk_after]) for e in wf.EPS_GRID if e > 0],
        }
    )
    coverage = pd.concat(coverages).groupby("region").mean(numeric_only=True).reset_index()
    ev_pooled = pd.concat(ev_frames, ignore_index=True)
    subgroup = wf.subgroup_decomposition(ev_pooled, by="region", eps=1.0)

    return WelfareResult(
        service=service,
        n_seeds=len(seeds),
        wevm_summary=_summary(by_seed),
        wevm_linear_summary=_summary(by_seed_lin),
        wevm_by_seed=by_seed,
        atkinson_summary=atk,
        coverage=coverage,
        ev_pooled=ev_pooled,
        subgroup=subgroup,
        notes=_service_notes(service),
    )


def _service_notes(service: str) -> list[str]:
    """Service-specific reporting caveats."""
    base = ["EV uses the per-period closed form; EV_linear is the money-metric robustness."]
    if service == "healthcare":
        base.append(
            "Healthcare uses the physical-health (PCS) utility with employment "
            "excluded as an income mediator, so the closed-form WEVM is finite. "
            "The access-to-health effect size (hc_access_health_effect) is "
            "set by calibration rather than estimated from UKHLS. "
            "The omitted non-pecuniary value of employment makes the indirect "
            "(health to employment) channel a mild lower bound."
        )
    return base
