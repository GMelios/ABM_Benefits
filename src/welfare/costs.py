"""Programme costs + cost-effectiveness verdict (ODD §7.9: WEVM-based benefit-cost).

Turns the per-person monetary WEVM into a verdict by comparing it to the programme's cost:

    benefit-cost ratio (BCR) = annual benefit per person / annual cost per person
    net social benefit       = annual benefit per person - annual cost per person

BCR > 1  ==>  the service delivers more monetised welfare than it costs (cost-effective).

Units (important): the welfare layer's per-period EV is on a MONTHLY-income basis (income is
£/month, a tick is a year). To compare against real annual-£ unit costs we ANNUALISE the
benefit by ``MONTHS_PER_YEAR`` (= 12). The EV difference Delta is invariant to monthly vs
annual income (the ln scale cancels), so annual EV = 12 x monthly EV; WEVM is linear in EV,
so annual WEVM = 12 x reported (monthly-basis) WEVM.

Reference figures (public, cited; documented constants, not fitted):
- NICE cost-effectiveness threshold: £20k-30k / QALY (-> £25k-35k from Apr 2026).
  https://www.nice.org.uk/news/articles/changes-to-nice-s-cost-effectiveness-thresholds-confirmed
- HM Treasury / DHSC social value of a QALY ~ £70k; 1 WELLBY ~ £13k (range £10-16k, 2021).
  https://whatworkswellbeing.org/blog/converting-the-wellby/
- Healthcare unit cost per contact: PSSRU Unit Costs of Health and Social Care (Jones et al.).
  https://www.pssru.ac.uk/unitcostsreport/  (exact per-contact figure is a documented lever)
"""

from __future__ import annotations

import pandas as pd

MONTHS_PER_YEAR: float = 12.0

# --- cited reference thresholds (for context / health verdicts) ---
NICE_THRESHOLD_PER_QALY: tuple[float, float] = (20_000.0, 30_000.0)  # £/QALY (pre-2026)
QALY_SOCIAL_VALUE: float = 70_000.0  # £ per QALY, HMT/DHSC
WELLBY_VALUE: float = 13_000.0  # £ per WELLBY (life-satisfaction point-year), HMT Green Book


def programme_cost_per_person(
    panel_factual: pd.DataFrame,
    *,
    service: str,
    cfg,
    delta: float,
    n_persons: int,
) -> float:
    """Discounted programme cost per person in real annual £ (incremental cost of the service).

    Income support: annualised transfers paid x (1 + admin loading). Healthcare: realised
    accesses x unit cost. The counterfactual incurs none of this service's cost, so the
    factual cost IS the incremental cost. ``n_persons`` must match the WEVM denominator.
    """
    g = panel_factual.groupby("tick")
    if service == "income_support":
        per_tick = g["transfer"].sum() * MONTHS_PER_YEAR * (1.0 + cfg.is_admin_loading)
    elif service == "healthcare":
        per_tick = g["access_healthcare"].sum() * cfg.hc_unit_cost
    else:
        raise ValueError(f"unknown service {service!r}")
    disc = delta ** per_tick.index.to_numpy(dtype=float)
    total_pv = float((per_tick.to_numpy() * disc).sum())
    return total_pv / n_persons


def cost_effectiveness(
    wevm_grid: dict[float, float],
    cost_per_person: float,
    *,
    annualise: float = MONTHS_PER_YEAR,
) -> pd.DataFrame:
    """Benefit-cost ratio and net social benefit per ε (annual £ per person).

    ``wevm_grid`` is the monthly-basis WEVM(ε); it is annualised by ``annualise`` to match the
    real annual-£ ``cost_per_person``.
    """
    rows = []
    for eps, wevm in wevm_grid.items():
        benefit = annualise * wevm
        bcr = benefit / cost_per_person if cost_per_person > 0 else float("inf")
        rows.append(
            {
                "eps": eps,
                "benefit_per_person": benefit,
                "cost_per_person": cost_per_person,
                "bcr": bcr,
                "net_social_benefit": benefit - cost_per_person,
                "cost_effective": bcr >= 1.0,
            }
        )
    return pd.DataFrame(rows)


def cost_per_life_year(
    panel_factual: pd.DataFrame,
    life_years_gained_total: float,
    *,
    cfg,
    delta: float,
) -> float:
    """Discounted healthcare cost per discounted life-year gained (health->mortality channel).

    Compare to the QALY social value (£70k) or NICE threshold, noting a life-year is not a
    full QALY (a life-year at health h<1 is worth a fraction of a QALY).
    """
    g = panel_factual.groupby("tick")
    per_tick = g["access_healthcare"].sum() * cfg.hc_unit_cost
    disc = delta ** per_tick.index.to_numpy(dtype=float)
    total_pv = float((per_tick.to_numpy() * disc).sum())
    return total_pv / life_years_gained_total if life_years_gained_total > 0 else float("inf")
