"""Scenario configuration: policy levers + service-effect sizes (the things we vary).

These are not estimated behavioural parameters (those live in ``params/`` via
:mod:`src.model.params`). They are policy choices and a small number of calibration
targets. Anything not grounded in UKHLS or an external source is flagged ``TODO: calibrate``
and surfaced here rather than buried in agent code.

Changing a :class:`PolicyConfig` *is* running a policy scenario (ODD §2.1 Policymaker).
The welfare layer produces matched factual/counterfactual pairs by toggling exactly one
service's ``*_on`` flag while holding seed + population fixed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PolicyConfig:
    """Policy levers for the two services and run controls.

    Income support (cash transfer)
    ------------------------------
    income_support_on : service present (factual) vs absent (counterfactual).
    is_eligibility_income : monthly market-income threshold for eligibility (£/month).
        Policy lever, a documented design choice, not estimated.
    is_min_income : guaranteed monthly net income; transfer tops eligible takers up to it.
    is_awareness : multiplier on the UKHLS-estimated baseline receipt probability
        (take-up). Free calibration parameter; 1.0 = estimated baseline.
    is_budget : total monthly transfer budget across all agents (£). If binding, transfers
        are rationed lowest-income-first (priority rule, ODD §7.3). None = uncapped.

    Healthcare (capacity-constrained access)
    ----------------------------------------
    healthcare_on : service present vs absent.
    hc_need_threshold : health index below which an agent needs care (need = threshold - h).
    hc_awareness : probability an agent with need seeks care. FREE calibration param.
    hc_capacity_per_1000 : provider slots per 1,000 population per tick (supply lever).
    hc_access_health_effect : health-index gain from a realised access this tick.
        Not estimable from UKHLS (no access measure); a calibration parameter set
        against external evidence.
    """

    # run controls
    n_agents: int = 5000
    horizon: int = 10  # ticks (years); ODD §2.3 policy horizon 10-20y
    base_wave: int = 1
    discount_delta: float = 0.97  # from build spec (decisions already made)

    # population dynamics (ODD §2), default OFF (closed cohort) to preserve baseline results.
    # When ON: age-based mortality removes agents and an equal number of young entrants join
    # (stationary N). Mortality is age-only UNLESS health_affects_mortality is also set
    # (below), which lets services change survival. See src/model/demography.py.
    population_dynamics: bool = False
    mortality_multiplier: float = 1.0  # scenario lever scaling the mortality schedule
    entry_max_age: int = 21  # entrants drawn from UKHLS respondents at/below this age

    # health -> mortality channel (lets healthcare extend life). Default OFF.
    # When ON, the age-mortality hazard is scaled by health: the sickest (h=0) face
    # `health_mortality_hr`x the death rate of the healthiest (h=1). NOT UKHLS-estimable
    # (no mortality in the panel) -> a documented calibration lever (illustrative).
    # Requires population_dynamics=True. Welfare then restricts to the initial cohort and
    # reports a separate life-extension component (see welfare.life_extension).
    health_affects_mortality: bool = False
    health_mortality_hr: float = (
        4.0  # hazard ratio, sickest vs healthiest (illustrative; TODO calibrate)
    )

    # spatial layer (ODD §2.2), default OFF (region index). When ON, agents are scattered
    # around their region centroid (real GOR geometry) and healthcare take-up decays with
    # distance to the region provider -> within-region spatial inequality in coverage. Needs
    # the `spatial` extra (geopandas/mesa-geo). See src/model/space.py.
    spatial: bool = False
    agent_scatter_km: float = 40.0  # sd of agent dispersion around the region centroid
    distance_decay_km: float = 50.0  # take-up decays as exp(-distance / this); behavioural lever

    # income-support service
    income_support_on: bool = True
    is_eligibility_income: float = 1000.0  # £/mo market income; policy lever
    is_min_income: float = 1200.0  # £/mo guaranteed net; policy lever
    is_awareness: float = 1.0  # take-up multiplier; FREE (calibrate)
    is_budget: float | None = None  # £/mo total; None = uncapped

    # healthcare service
    healthcare_on: bool = True
    hc_need_threshold: float = 0.50  # health index; policy/design
    hc_awareness: float = 0.7  # seek-care prob; FREE (calibrate)
    hc_capacity_per_1000: float = 80.0  # slots/1000/tick; supply lever
    hc_access_health_effect: float = 0.02  # h gain per access; CALIBRATION (TODO)

    # programme costs (for the cost-effectiveness verdict; see src/welfare/costs.py)
    is_admin_loading: float = 0.03  # benefit admin as a fraction of transfers (DWP-ish; documented)
    hc_unit_cost: float = 50.0  # £ per healthcare access (PSSRU Unit Costs-based; documented)

    def with_(self, **overrides: Any) -> PolicyConfig:
        """Return a copy with overrides applied (e.g. to build a counterfactual)."""
        d = asdict(self)
        d.update(overrides)
        return PolicyConfig(**d)

    def counterfactual(self, service: str) -> PolicyConfig:
        """Return the matched counterfactual: this config with ``service`` switched OFF."""
        flag = {"income_support": "income_support_on", "healthcare": "healthcare_on"}[service]
        return self.with_(**{flag: False})

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicyConfig:
        """Load a scenario from a YAML file (experiments/configs/*.yaml)."""
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        fields = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)


#: Free parameters eligible for calibration, with documented search bounds.
CALIBRATION_PARAMS: dict[str, tuple[float, float]] = {
    "is_awareness": (0.0, 2.0),
    "hc_awareness": (0.0, 1.0),
    "hc_access_health_effect": (0.0, 0.10),
}

#: Provenance note for the calibration effect size (kept with the config).
UNCALIBRATED_NOTES: list[str] = [
    "hc_access_health_effect is not estimated from UKHLS (no service-access measure); "
    "it is a calibration parameter that sets the health-service welfare estimate.",
]
