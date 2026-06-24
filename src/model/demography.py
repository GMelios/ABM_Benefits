"""Demographic turnover for the ABM: age-based mortality (ODD §2 population dynamics).

Scope: mortality depends on age only, not on health or income. This keeps
deaths identical across matched factual/counterfactual runs (common random numbers), so the
welfare comparison stays clean. The consequence is that services do not extend life under this configuration,
a health->mortality channel (and its life-valuation implications) is a deliberate next step,
not included here.

Entry is handled in the model: each death is replaced by a young entrant drawn from the
UKHLS young-respondent pool (stationary population), so the age structure churns while N is
held constant.
"""

from __future__ import annotations

# Approximate UK annual death probability (qx) by age band. Illustrative values in the right
# ballpark of ONS National Life Tables; NOT the exact series.
# TODO: calibrate, replace with the published ONS National Life Tables qx by single year of age.
MORTALITY_BY_AGE_BAND: dict[tuple[int, int], float] = {
    (16, 24): 0.0004,
    (25, 34): 0.0007,
    (35, 44): 0.0013,
    (45, 54): 0.0030,
    (55, 64): 0.0075,
    (65, 74): 0.0180,
    (75, 84): 0.0500,
    (85, 200): 0.1300,
}


#: Reference health at which the health->mortality factor is 1.0 (population-ish midpoint).
HEALTH_REF: float = 0.5


def _age_qx(age: float) -> float:
    """Baseline annual death probability by age band (age-only)."""
    a = int(age)
    for (lo, hi), q in MORTALITY_BY_AGE_BAND.items():
        if lo <= a <= hi:
            return q
    return MORTALITY_BY_AGE_BAND[(16, 24)] if a < 16 else MORTALITY_BY_AGE_BAND[(85, 200)]


def mortality_prob(
    age: float,
    *,
    multiplier: float = 1.0,
    health: float | None = None,
    hazard_ratio: float = 1.0,
) -> float:
    """Return the annual probability of death (age schedule, scaled by lever + health).

    ``multiplier`` is a scenario lever (1.0 = baseline). When ``health`` is supplied and
    ``hazard_ratio`` > 1, the hazard is scaled by ``hazard_ratio ** (HEALTH_REF - health)``
    so the sickest (h=0) face ``hazard_ratio``x the rate of the healthiest (h=1); at
    ``health == HEALTH_REF`` the factor is 1. This is the health->mortality channel that
    lets healthcare extend life. ``hazard_ratio == 1`` recovers the age-only schedule.
    """
    q = _age_qx(age) * multiplier
    if health is not None and hazard_ratio != 1.0:
        q *= hazard_ratio ** (HEALTH_REF - float(health))
    return min(max(q, 0.0), 1.0)
