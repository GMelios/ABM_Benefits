"""Synthetic population for the ABM (T1.3 initialisation, ODD §5).

Builds the agent population whose initial joint state (income, health, employment,
education, region) is drawn from the UKHLS empirical distribution so correlations are
preserved (ODD §5.1).

Method (DOCUMENTED, important): no external area control totals (ONS LAD/GOR marginals)
were supplied with this extract. Per the build spec, the fallback is to **resample the
survey-weighted UKHLS cross-section**, sampling whole person-records with probability
proportional to the cross-sectional weight reproduces the weighted joint distribution
(all margins AND their correlations) by construction. An IPF reweighter (:func:`ipf_reweight`)
is provided for when real area marginals become available; the ABM space seam is unchanged.

All randomness flows through a single seeded NumPy ``Generator``.
Operates on individual data in memory; emits only the synthetic agent table (not microdata
in the disclosure sense, but treat with the same care, do not persist to tracked paths).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.estimation.load_ukhls import GOR_NAMES, WEIGHTS

# Columns each agent carries as its t=0 state (ODD §2.1).
INIT_STATE_COLS: tuple[str, ...] = (
    "region",
    "dvage",
    "sex",
    "education",
    "hidp",
    "employed",
    "income_net",
    "y_market",
    "y_benefit",
    "health_index",
    "log_income",
)

# Essential vars an agent must have to be initialised (listwise-complete base).
_ESSENTIAL: tuple[str, ...] = (
    "region",
    "dvage",
    "sex",
    "employed",
    "income_net",
    "y_market",
    "y_benefit",
    "health_index",
    WEIGHTS.cross_sectional,
)

AGE_BANDS = [16, 25, 35, 45, 55, 65, 75, 200]
AGE_LABELS = ["16-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]


def base_cross_section(panel: pd.DataFrame, *, wave: int = 1) -> pd.DataFrame:
    """Return the listwise-complete, positively-weighted base cross-section for ``wave``.

    Education band is added if absent. Records missing any essential init variable or with
    a non-positive weight are dropped (count surfaced via the returned frame's attrs).
    """
    df = panel[panel["wave"] == wave].copy()
    if "education" not in df.columns:
        from src.estimation._prep import education_band

        df["education"] = education_band(df)
    n0 = len(df)
    df = df[
        [c for c in {*INIT_STATE_COLS, *_ESSENTIAL, WEIGHTS.cross_sectional} if c in df.columns]
    ].copy()
    df = df.dropna(subset=list(_ESSENTIAL))
    df = df[df[WEIGHTS.cross_sectional] > 0]
    df.attrs["dropped_frac"] = 1.0 - len(df) / n0 if n0 else 0.0
    df.attrs["base_wave"] = wave
    return df.reset_index(drop=True)


def build_population(
    panel: pd.DataFrame,
    *,
    n_agents: int = 5000,
    seed: int = 0,
    base_wave: int = 1,
) -> pd.DataFrame:
    """Build ``n_agents`` synthetic agents by survey-weighted resampling (ODD §5.1).

    Returns
    -------
    pd.DataFrame
        One row per agent with ``agent_id``, ``y0`` (baseline income fixed for WEVM
        weighting, ODD §5.3), an ``age_band``, and the :data:`INIT_STATE_COLS` initial state.
    """
    base = base_cross_section(panel, wave=base_wave)
    if base.empty:
        raise ValueError(f"No usable base cross-section at wave {base_wave}.")
    rng = np.random.default_rng(seed)
    w = base[WEIGHTS.cross_sectional].to_numpy(dtype=float)
    probs = w / w.sum()
    idx = rng.choice(len(base), size=n_agents, replace=True, p=probs)

    pop = base.iloc[idx].reset_index(drop=True).copy()
    pop.insert(0, "agent_id", np.arange(n_agents))
    pop["y0"] = pop["income_net"].astype(float)  # baseline income, held fixed (ODD §5.3)
    pop["age_band"] = pd.cut(pop["dvage"], bins=AGE_BANDS, labels=AGE_LABELS, right=False).astype(
        "category"
    )
    keep = ["agent_id", "y0", "age_band", *INIT_STATE_COLS]
    pop = pop[[c for c in keep if c in pop.columns]]
    pop.attrs["base_wave"] = base_wave
    pop.attrs["base_dropped_frac"] = base.attrs.get("dropped_frac", 0.0)
    return pop


# --------------------------------------------------------------------------------------
# Margin validation (validate the population before trusting output)
# --------------------------------------------------------------------------------------
def _shares(values: pd.Series, weights: pd.Series | None = None) -> pd.Series:
    """Category shares, optionally survey-weighted, summing to 1."""
    if weights is None:
        return values.value_counts(normalize=True)
    s = pd.Series(weights.to_numpy(), index=values.to_numpy()).groupby(level=0).sum()
    return s / s.sum()


def margin_report(pop: pd.DataFrame, panel: pd.DataFrame, *, base_wave: int = 1) -> pd.DataFrame:
    """Compare synthetic-population margins to the survey-weighted UKHLS base margins.

    Returns a tidy frame: dimension, category, target_share (weighted UKHLS), synth_share,
    abs_diff. Covers region, sex, age band, and the employment rate.
    """
    base = base_cross_section(panel, wave=base_wave)
    base["age_band"] = pd.cut(base["dvage"], bins=AGE_BANDS, labels=AGE_LABELS, right=False)
    pop = pop.copy()
    if "age_band" not in pop.columns:
        pop["age_band"] = pd.cut(pop["dvage"], bins=AGE_BANDS, labels=AGE_LABELS, right=False)

    rows = []
    dims = {
        "region": ("region", lambda d: d["region"].astype(str)),
        "sex": ("sex", lambda d: d["sex"].astype(str)),
        "age_band": ("age_band", lambda d: d["age_band"].astype(str)),
    }
    w = base[WEIGHTS.cross_sectional]
    for dim, (_, getter) in dims.items():
        tgt = _shares(getter(base), w)
        syn = _shares(getter(pop))
        for cat in sorted(set(tgt.index) | set(syn.index)):
            t, s = float(tgt.get(cat, 0.0)), float(syn.get(cat, 0.0))
            rows.append((dim, cat, t, s, abs(t - s)))
    # employment rate (single number)
    t_emp = float(np.average(base["employed"].astype(float), weights=w))
    s_emp = float(pop["employed"].astype(float).mean())
    rows.append(("employment_rate", "employed", t_emp, s_emp, abs(t_emp - s_emp)))
    return pd.DataFrame(
        rows, columns=["dimension", "category", "target_share", "synth_share", "abs_diff"]
    )


def validate_margins(
    pop: pd.DataFrame, panel: pd.DataFrame, *, base_wave: int = 1, tol: float = 0.02
) -> pd.DataFrame:
    """Assert every synthetic margin matches the weighted UKHLS target within ``tol``.

    Raises
    ------
    AssertionError
        If any category's absolute share difference exceeds ``tol``.
    """
    rep = margin_report(pop, panel, base_wave=base_wave)
    worst = rep.sort_values("abs_diff", ascending=False).iloc[0]
    if worst["abs_diff"] > tol:
        raise AssertionError(
            f"Population margin off target by {worst['abs_diff']:.3f} > tol={tol} "
            f"at {worst['dimension']}={worst['category']} "
            f"(target {worst['target_share']:.3f}, synth {worst['synth_share']:.3f})."
        )
    return rep


# --------------------------------------------------------------------------------------
# IPF reweighter (capability for when external area marginals are supplied)
# --------------------------------------------------------------------------------------
def ipf_reweight(
    seed: pd.DataFrame,
    marginals: list[tuple[list[str], pd.Series | np.ndarray]],
    *,
    weight_col: str = "weight",
    max_iter: int = 200,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Fit cell weights to target marginals by iterative proportional fitting (``ipfn``).

    Parameters
    ----------
    seed
        Long frame of cells with an initial ``weight_col``.
    marginals
        List of (dimension-columns, target-totals) constraints to fit.

    Returns
    -------
    pd.DataFrame
        ``seed`` with ``weight_col`` adjusted so the marginals are reproduced. Use this when
        real ONS area control totals are available instead of the resampling fallback.
    """
    from ipfn import ipfn

    df = seed.copy()
    if weight_col not in df.columns:
        df[weight_col] = 1.0
    # ipfn's DataFrame API indexes targets by category via .loc, so keep pandas Series
    # (indexed by the dimension's category values); only bare lists are coerced to Series.
    aggregates = [
        m[1] if isinstance(m[1], pd.Series) else pd.Series(np.asarray(m[1], dtype=float))
        for m in marginals
    ]
    dimensions = [list(m[0]) for m in marginals]
    fitted = ipfn.ipfn(
        df,
        aggregates,
        dimensions,
        weight_col=weight_col,
        max_iteration=max_iter,
        convergence_rate=tol,
    ).iteration()
    return fitted


if __name__ == "__main__":  # pragma: no cover - manual inspection
    from src.estimation.load_ukhls import load_panel

    panel = load_panel()
    pop = build_population(panel, n_agents=5000, seed=0)
    print(
        f"Built {len(pop):,} agents (base wave {pop.attrs['base_wave']}, "
        f"dropped {pop.attrs['base_dropped_frac']:.1%} of base for missingness)."
    )
    rep = validate_margins(pop, panel, tol=0.02)
    print("\nMargin recovery (max abs diff = " f"{rep['abs_diff'].max():.4f}):")
    print(rep.to_string(index=False))
    _ = GOR_NAMES  # referenced for region labelling
