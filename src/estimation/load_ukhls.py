"""Canonical loader for the harmonised UKHLS long panel (T1.3 estimation layer).

PROVENANCE
-------------------------
Source      : UK Household Longitudinal Study (Understanding Society, UKHLS),
              main adult interview, waves 1-11 (letters a-k).
Access tier : End User Licence (EUL). Geography = Government Office Region (`gor_dv`).
              Finer geography (LAD) would require Special Licence, NOT used here.
File        : data/ukhls_panel_long.parquet  (GIT-IGNORED restricted microdata)
Form        : Already a long person x wave panel (476,187 rows, 87,905 persons),
              with UKHLS sentinel missing codes (-1,-2,-7,-8,-9) already mapped to NaN
              and source wave-prefixed variables harmonised to the friendly names below.

This module does NOT re-derive the harmonisation; it loads, *validates*, documents, and
adds the minimal analysis columns the estimation/ABM layers need. Heavier estimation-time
transforms live in the per-model estimation scripts.

Restricted-data rule: this module returns individual-level data. Never persist its output
to a tracked path, never log raw rows. Only aggregates/coefficients may leave `data/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL_PATH = _REPO_ROOT / "data" / "ukhls_panel_long.parquet"

# --------------------------------------------------------------------------------------
# Codebook constants (these are metadata, not microdata, safe to keep in tracked code)
# --------------------------------------------------------------------------------------
WAVE_LETTERS: dict[int, str] = {i: chr(ord("a") + i - 1) for i in range(1, 12)}  # 1->a ... 11->k

#: Government Office Region codes (UKHLS `gor_dv`). 12 = Northern Ireland (UKHLS extension).
GOR_NAMES: dict[int, str] = {
    1: "North East",
    2: "North West",
    3: "Yorkshire and the Humber",
    4: "East Midlands",
    5: "West Midlands",
    6: "East of England",
    7: "London",
    8: "South East",
    9: "South West",
    10: "Wales",
    11: "Scotland",
    12: "Northern Ireland",
}

#: `jbstat` current economic activity, kept for reference; `employed` is already derived.
JBSTAT_LABELS: dict[int, str] = {
    1: "self-employed",
    2: "paid employment",
    3: "unemployed",
    4: "retired",
    5: "on maternity leave",
    6: "family care / home",
    7: "full-time student",
    8: "long-term sick/disabled",
    9: "govt training scheme",
    10: "unpaid family business",
    11: "on apprenticeship",
    12: "doing something else",
    13: "not asked / inapplicable",
    97: "other",
}

#: Columns we require to be present; loader fails loudly if the schema drifts.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "pidp",
    "wave",
    "hidp",
    "income_net",
    "income_gross",
    "income_labour_net",
    "income_labour_gross",
    "y_benefit",
    "y_market",
    "y_market_is_pretransfer",
    "health_pcs",
    "health_mcs",
    "ghq",
    "ghq_caseness",
    "srh",
    "health_state",
    "disability",
    "disability_type",
    "jbstat",
    "employed",
    "hours",
    "dvage",
    "sex",
    "mastat_dv",
    "hiqual_dv",
    "ethn_dv",
    "nkids_dv",
    "gor_dv",
    "urban_dv",
    "weight_xsec",
    "design_weight",
    "weight_long",
)

#: Columns known to be fully/severely unusable in THIS extract (surfaced, not hidden).
KNOWN_EMPTY_COLUMNS: tuple[str, ...] = (
    "nkids_dv",  # 100% missing in this extract -> childcare and child counts unavailable (a later extension)
)


@dataclass(frozen=True)
class Weights:
    """Which survey weight applies to which analysis (UKHLS conventions).

    cross_sectional : representativeness of a single wave's cross-section -> use for
                      population margins / initialisation targets (present all waves).
    longitudinal    : representativeness of wave-to-wave transitions -> use for the
                      dynamic transition models (present waves 2-11; wave 1 has none).
    """

    cross_sectional: str = "weight_xsec"
    longitudinal: str = "weight_long"


WEIGHTS = Weights()


# --------------------------------------------------------------------------------------
# Load + validate
# --------------------------------------------------------------------------------------
def load_panel(
    path: str | Path = DEFAULT_PANEL_PATH,
    *,
    validate: bool = True,
    add_derived: bool = True,
) -> pd.DataFrame:
    """Load the harmonised UKHLS long panel.

    Parameters
    ----------
    path
        Parquet path. Defaults to the git-ignored ``data/`` location.
    validate
        Run :func:`validate_panel` (schema, wave coverage, sentinel-negative scan).
    add_derived
        Attach the minimal analysis columns (see :func:`add_analysis_columns`).

    Returns
    -------
    pd.DataFrame
        Long panel sorted by ``(pidp, wave)``. Individual-level, handle per §4.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"UKHLS panel not found at {path}. It is restricted microdata held in the "
            "git-ignored data/ directory and must be obtained under the UK Data Service EUL."
        )
    df = pd.read_parquet(path)
    df = df.sort_values(["pidp", "wave"]).reset_index(drop=True)
    if validate:
        validate_panel(df)
    if add_derived:
        df = add_analysis_columns(df)
    return df


def validate_panel(df: pd.DataFrame) -> None:
    """Assert the panel matches the documented schema before anything trusts it (§3.7).

    Raises
    ------
    ValueError
        If required columns are missing, waves are out of range, or supposedly-clean
        numeric columns still carry UKHLS sentinel negatives.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Panel is missing expected columns: {missing}")

    waves = set(df["wave"].unique().tolist())
    if not waves.issubset(set(WAVE_LETTERS)):
        raise ValueError(f"Unexpected wave codes: {sorted(waves - set(WAVE_LETTERS))}")

    # Sentinel-negative scan: these columns must never be < 0 (incomes can legitimately be 0).
    nonneg = [
        "income_net",
        "income_gross",
        "y_benefit",
        "y_market",
        "health_pcs",
        "health_mcs",
        "ghq",
        "ghq_caseness",
        "srh",
        "health_state",
        "dvage",
        "hours",
    ]
    for c in nonneg:
        neg = int((df[c] < 0).sum())
        if neg:
            raise ValueError(
                f"Column {c!r} has {neg} negative values, UKHLS sentinel missing codes "
                "may not have been cleaned to NaN. Refusing to proceed."
            )

    # GOR must decode.
    bad_gor = set(df["gor_dv"].dropna().unique()) - set(GOR_NAMES)
    if bad_gor:
        raise ValueError(f"Unknown gor_dv codes: {sorted(bad_gor)}")


# --------------------------------------------------------------------------------------
# Derived analysis columns
# --------------------------------------------------------------------------------------
# Tunable, documented constants (no undocumented magic numbers).
INCOME_FLOOR_PCT: float = 1.0  # winsorise income at this lower percentile before ln()
INCOME_WINSOR_TOP_PCT: float = 99.5  # ...and this upper percentile (tames the £723k outlier)
GHQ_MAX: int = 36  # GHQ-12 Likert scale maximum (scghq1_dv ranges 0-36)


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach minimal analysis columns used downstream (non-destructive copy).

    Adds
    ----
    region : categorical region name from ``gor_dv``.
    income_for_utility : ``income_net`` winsorised to a positive range for safe ``ln()``.
    log_income : natural log of ``income_for_utility`` (the utility/EV income term, §7.7).
    wellbeing_ghq : reverse-scored GHQ-12 (``GHQ_MAX - ghq``), higher = better wellbeing.
        The wellbeing anchor uses reverse-scored GHQ-12; life satisfaction (sclfsato) is
        not available in this extract.
    health_index : SF-12 PCS/MCS mapped to [0,1] as an SF-6D *proxy* (items unavailable).
        # Calibrate to a published SF-6D crosswalk to refine this mapping.
    """
    df = df.copy()

    # Region label
    df["region"] = df["gor_dv"].map(GOR_NAMES).astype("category")

    # Income for utility: winsorise to a strictly-positive band, then ln().
    inc = df["income_net"].astype(float)
    pos = inc[inc > 0]
    lo = float(np.percentile(pos, INCOME_FLOOR_PCT))
    hi = float(np.percentile(pos, INCOME_WINSOR_TOP_PCT))
    df["income_for_utility"] = inc.clip(lower=lo, upper=hi)
    df["log_income"] = np.log(df["income_for_utility"])

    # Wellbeing anchor: reverse-scored GHQ so that higher means better.
    df["wellbeing_ghq"] = GHQ_MAX - df["ghq"]

    # Health stock proxy in [0,1]; a tariff-based SF-6D would need item-level SF-12.
    df["health_index"] = health_index_01(df)

    return df


def health_index_01(df: pd.DataFrame) -> pd.Series:
    """Map the SF-12 PHYSICAL component summary (PCS) to a [0,1] health stock.

    This is the model's health stock, what the healthcare service improves and what feeds
    employment/earnings. We use PCS (physical functioning) ONLY, deliberately EXCLUDING the
    mental component (MCS): MCS is near-collinear with GHQ-12 (corr ~0.72), and since GHQ is
    our wellbeing/utility anchor, including MCS would price mental health on both sides of the
    utility equation, inflating beta_health and making the healthcare EV degenerate. PCS keeps
    physical health (the healthcare target) cleanly separate from GHQ mental wellbeing.

    Transparent min-max of PCS onto [0,1] via the theoretical 0-100 SF-12 summary range.
    # TODO: calibrate, replace with an item-level SF-6D tariff if an extract with SF-12
    #        items becomes available; until then this is a documented physical-health proxy.
    """
    pcs = df["health_pcs"].astype(float)
    return (pcs / 100.0).clip(lower=0.0, upper=1.0)


# --------------------------------------------------------------------------------------
# Summary helpers (for the summary tables and tests)
# --------------------------------------------------------------------------------------
def variable_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column coverage table: dtype, % missing, and min/max for numerics."""
    rows = []
    for c in df.columns:
        s = df[c]
        miss = float(s.isna().mean() * 100)
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            vmin, vmax = (float(s.min()), float(s.max())) if s.notna().any() else (np.nan, np.nan)
        else:
            vmin = vmax = np.nan
        rows.append((c, str(s.dtype), round(miss, 1), vmin, vmax))
    return pd.DataFrame(rows, columns=["column", "dtype", "pct_missing", "min", "max"])


def wave_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Rows and unique persons per wave."""
    g = df.groupby("wave")
    return pd.DataFrame({"rows": g.size(), "n_persons": g["pidp"].nunique()}).reset_index()


if __name__ == "__main__":  # pragma: no cover - manual data inspection
    panel = load_panel()
    print(
        f"Loaded {len(panel):,} person-waves; {panel['pidp'].nunique():,} persons; "
        f"waves {panel['wave'].min()}-{panel['wave'].max()}"
    )
    print("\n--- wave coverage ---")
    print(wave_coverage(panel).to_string(index=False))
    print("\n--- variable summary ---")
    print(variable_summary(panel).to_string(index=False))
