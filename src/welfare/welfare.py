"""Valuation layer: equivalent variation + the WEVM (ODD §7.8-7.9).

Given matched factual/counterfactual per-agent panels (same seed + population, differing
only in one service), compute each agent's equivalent variation and aggregate via the
Weighted Equivalent-Variation Measure over the inequality-aversion grid.

EV definition (per-period, then discount-summed, the approved refinement of §6, see the
ev-trajectory note). For agent i, deriving from v_i(y^cf+EV, S_0) = v_i(y^f, S_1) with
v = beta_y·ln(y) + phi(S):

    EV_i = sum_t  delta^t · y^{cf}_{i,t} · (exp(Delta_{i,t}/beta_y) - 1),
    Delta_{i,t} = v^{factual}_{i,t} - v^{counterfactual}_{i,t}   (per-period utility diff)

The multiplier is the *counterfactual* income (the world being compensated). A linear
money-metric (first-order, income-effect-free) is reported alongside as a robustness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Inequality-aversion grid (ODD §7.9). eps=0 is the utilitarian benchmark.
EPS_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)

#: Numerical guard on the exp() argument (rare; mainly the health channel where the
#: GHQ-overlap-inflated beta_health meets the small beta_y). Keeps EV finite.
_EXP_CLIP = 50.0


def equivalent_variation(
    panel_factual: pd.DataFrame,
    panel_counterfactual: pd.DataFrame,
    *,
    beta_y: float,
    delta: float,
) -> pd.DataFrame:
    """Per-agent EV from matched panels (merged on agent_id x tick).

    Returns
    -------
    pd.DataFrame
        One row per agent: ``agent_id``, ``region``, ``y0`` (baseline income, fixed for
        WEVM weighting), ``EV`` (per-period closed form), ``EV_linear`` (money-metric).
    """
    keys = ["agent_id", "tick"]
    f = panel_factual[[*keys, "utility", "income", "y0", "region"]].rename(
        columns={"utility": "u_f", "income": "y_f"}
    )
    cf = panel_counterfactual[[*keys, "utility", "income"]].rename(
        columns={"utility": "u_cf", "income": "y_cf"}
    )
    m = f.merge(cf, on=keys, how="inner")
    m["delta_t"] = m["u_f"] - m["u_cf"]
    disc = delta ** m["tick"].to_numpy()
    y_cf = np.maximum(m["y_cf"].to_numpy(), 1.0)  # counterfactual income, floored positive

    arg = np.clip(m["delta_t"].to_numpy() / beta_y, -_EXP_CLIP, _EXP_CLIP)
    ev_t = disc * y_cf * np.expm1(arg)  # per-period closed-form EV
    ev_lin_t = disc * y_cf * (m["delta_t"].to_numpy() / beta_y)  # linear money-metric

    m["_ev_t"] = ev_t
    m["_ev_lin_t"] = ev_lin_t
    out = (
        m.groupby("agent_id")
        .agg(
            region=("region", "first"),
            y0=("y0", "first"),
            EV=("_ev_t", "sum"),
            EV_linear=("_ev_lin_t", "sum"),
        )
        .reset_index()
    )
    return out


def life_extension(
    panel_factual: pd.DataFrame,
    panel_counterfactual: pd.DataFrame,
    *,
    beta_y: float,
    delta: float,
    dead_utility: float = 0.0,
) -> pd.DataFrame:
    """Per-agent value of life-extension on the INITIAL COHORT (health->mortality channel).

    For ticks where exactly one arm is alive (the service changed survival), the welfare gain
    is the living utility relative to ``dead_utility`` (reference = 0). The full-life-year
    utility jump over the small beta_y would overflow the closed form, so this component is
    valued with the LINEAR money-metric and reported SEPARATELY from the within-life WEVM.

    Restricted to the initial cohort (agents present at tick 0), whose ids match across arms;
    entrants diverge across arms under differential mortality and are excluded.

    Returns
    -------
    pd.DataFrame
        ``agent_id, y0, region, life_years_gained`` (discounted), ``life_extension_ev``,
        one row per cohort agent (0 where survival did not differ).
    """
    cohort = sorted(set(panel_factual.loc[panel_factual["tick"] == 0, "agent_id"]))
    fa = panel_factual[panel_factual["agent_id"].isin(cohort)][
        ["agent_id", "tick", "utility", "y0", "region"]
    ].rename(columns={"utility": "u_f"})
    ca = panel_counterfactual[panel_counterfactual["agent_id"].isin(cohort)][
        ["agent_id", "tick", "utility"]
    ].rename(columns={"utility": "u_cf"})

    m = fa.merge(ca, on=["agent_id", "tick"], how="outer")
    m["f_alive"] = m["u_f"].notna()
    m["cf_alive"] = m["u_cf"].notna()
    diff = m[m["f_alive"] != m["cf_alive"]].copy()  # exactly one arm alive

    y0map = fa.groupby("agent_id")["y0"].first()
    regmap = fa.groupby("agent_id")["region"].first()
    uf = diff["u_f"].fillna(dead_utility).to_numpy()
    ucf = diff["u_cf"].fillna(dead_utility).to_numpy()
    disc = delta ** diff["tick"].to_numpy()
    diff["ev_t"] = disc * diff["agent_id"].map(y0map).to_numpy() * ((uf - ucf) / beta_y)
    diff["ly_t"] = disc * (diff["f_alive"].astype(int) - diff["cf_alive"].astype(int)).to_numpy()

    g = diff.groupby("agent_id").agg(
        life_extension_ev=("ev_t", "sum"), life_years_gained=("ly_t", "sum")
    )
    out = pd.DataFrame({"agent_id": cohort})
    out = out.merge(g, on="agent_id", how="left").fillna(
        {"life_extension_ev": 0.0, "life_years_gained": 0.0}
    )
    out["y0"] = out["agent_id"].map(y0map)
    out["region"] = out["agent_id"].map(regmap)
    return out


def wevm(ev: np.ndarray | pd.Series, y0: np.ndarray | pd.Series, eps: float) -> float:
    """Weighted Equivalent-Variation Measure at inequality aversion ``eps`` (ODD §7.9).

    ``WEVM(eps) = sum_i (y0_i/y*)^(-eps) EV_i / sum_i (y0_i/y*)^(-eps)``, with
    ``y* = mean baseline income`` (fixed benchmark).
    """
    ev = np.asarray(ev, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    y_star = y0.mean()
    w = (np.maximum(y0, 1.0) / y_star) ** (-eps)
    return float(np.sum(w * ev) / np.sum(w))


def wevm_grid(ev, y0, eps_grid: tuple[float, ...] = EPS_GRID) -> dict[float, float]:
    """WEVM across the full eps grid."""
    return {eps: wevm(ev, y0, eps) for eps in eps_grid}


def atkinson(values: np.ndarray | pd.Series, eps: float) -> float:
    """Atkinson inequality index of a positive distribution at aversion ``eps``.

    ``A = 1 - (E[y^(1-eps)])^(1/(1-eps)) / E[y]`` for eps != 1; geometric-mean form at eps=1.
    Used to report how the welfare (income) distribution moves before vs after a service.
    """
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    y = np.maximum(y, 1.0)
    mean = y.mean()
    if abs(eps - 1.0) < 1e-9:
        ede = np.exp(np.mean(np.log(y)))
    else:
        ede = np.mean(y ** (1.0 - eps)) ** (1.0 / (1.0 - eps))
    return float(1.0 - ede / mean)


def subgroup_decomposition(
    ev_df: pd.DataFrame, *, by: str = "region", eps: float = 1.0
) -> pd.DataFrame:
    """Decompose total priority-weighted EV by subgroup using PRIORITY MASS, not pop share.

    Priority mass of agent i = ``(y0_i/y*)^(-eps)`` (the WEVM weight). Each subgroup's
    reported share is its share of total priority mass and of total priority-weighted EV
    (ODD §7.9: decomposition by priority mass, not population share).
    """
    df = ev_df.copy()
    y_star = df["y0"].mean()
    df["priority_mass"] = (np.maximum(df["y0"], 1.0) / y_star) ** (-eps)
    df["weighted_ev"] = df["priority_mass"] * df["EV"]
    g = df.groupby(by)
    rep = pd.DataFrame(
        {
            "n": g.size(),
            "pop_share": g.size() / len(df),
            "priority_mass_share": g["priority_mass"].sum() / df["priority_mass"].sum(),
            "wevm_subgroup": g.apply(lambda s: wevm(s["EV"], s["y0"], eps), include_groups=False),
            "weighted_ev_share": g["weighted_ev"].sum() / df["weighted_ev"].sum(),
        }
    ).reset_index()
    return rep
