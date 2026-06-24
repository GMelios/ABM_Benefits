"""Mesa agents for the social-services ABM (ODD §2.1).

Individual/Household, ServiceProvider, and Policymaker. Behaviour uses estimated
coefficients (via :class:`src.model.params.ModelParams`) and explicit policy levers
(:class:`src.model.config.PolicyConfig`). All randomness flows through ``self.model.rng``
(a single seeded NumPy Generator), no bare ``random``/``np.random``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from mesa import Agent

if TYPE_CHECKING:
    from src.model.model import SocialServicesModel

AGE_SCALE = 10.0  # must match estimation (src/estimation/estimate.py AGE_SCALE)


class Individual(Agent):
    """A person: the unit of welfare. Carries income/health/employment state + trajectory.

    Income is decomposed as net = market + benefit + other (other held at baseline). The
    income-support service modifies the benefit component; the healthcare service modifies
    the health stock, which feeds employment and earnings (the indirect channels).
    """

    def __init__(self, model: SocialServicesModel, init: dict) -> None:
        super().__init__(model)
        # Stable birth id (= population agent_id for initial agents, a sequential id for
        # entrants) and a per-agent RNG stream derived from it + the single model seed. This
        # gives common random numbers across matched factual/counterfactual runs without a
        # model-level registry, and survives population turnover.
        self.aid: int = int(init["agent_id"])
        # NB: `rng` is a reserved (read-only) Mesa Agent property -> use `prng` for the
        # per-agent common-random-number stream.
        self.prng: np.random.Generator = np.random.default_rng([model._seed, self.aid])
        self.region: str = init["region"]
        self.age: float = float(init["dvage"])
        self.sex: float = float(init["sex"])
        self.education: str = init["education"]
        self.hidp: int = int(init["hidp"])

        self.employed: bool = bool(init["employed"])
        self.y_market: float = float(init["y_market"])
        self.y_benefit: float = float(init["y_benefit"])
        # other (investment/pension/etc.) = net minus the two modelled components, held fixed
        self.y_other: float = max(float(init["income_net"]) - self.y_market - self.y_benefit, 0.0)
        self.health: float = float(init["health_index"])

        self.y0: float = float(init["y0"])  # baseline income, fixed for WEVM (ODD §5.3)
        self.income: float = self.y_market + self.y_benefit + self.y_other

        # per-service flags set during the tick
        self.need: dict[str, float] = {"income_support": 0.0, "healthcare": 0.0}
        self.eligible: dict[str, bool] = {"income_support": False, "healthcare": False}
        self.access: dict[str, bool] = {"income_support": False, "healthcare": False}
        self.transfer: float = 0.0  # this tick's income-support transfer

        self.utility: float = 0.0
        self.utility_trajectory: list[float] = []

        # spatial layer (optional): location + distance-decay access factor, fixed at init from
        # the agent's own seeded stream so it is identical across matched runs (CRN).
        self.distance_km: float = 0.0
        self.distance_access_factor: float = 1.0
        if model.cfg.spatial:
            from src.model import space

            _x, _y, self.distance_km = space.assign_location(
                self.region, self.prng, scatter_km=model.cfg.agent_scatter_km
            )
            self.distance_access_factor = space.distance_decay(
                self.distance_km, scale_km=model.cfg.distance_decay_km
            )

    # -- helpers -------------------------------------------------------------------------
    @property
    def age10(self) -> float:
        """Age on the estimation scale (age/10)."""
        return self.age / AGE_SCALE

    @property
    def log_income(self) -> float:
        """Return ln of current net income (floored, matching the utility income term)."""
        return math.log(max(self.income, 1.0))

    def recompute_income(self) -> None:
        """Refresh net income from its components (incl. this tick's service transfer)."""
        self.income = self.y_market + self.y_benefit + self.y_other + self.transfer

    # -- ODD §3 tick stages (called by the model in order) -------------------------------
    def update_need_eligibility(self) -> None:
        """Step 2: update need[·] and eligible[·] from current state + policy thresholds."""
        cfg = self.model.cfg
        # income support: working-age, low market income
        self.eligible["income_support"] = (
            cfg.income_support_on
            and 16 <= self.age <= 64
            and self.y_market < cfg.is_eligibility_income
        )
        self.need["income_support"] = max(cfg.is_min_income - self.income, 0.0)
        # healthcare: need rises as health falls below threshold
        self.need["healthcare"] = max(cfg.hc_need_threshold - self.health, 0.0)
        self.eligible["healthcare"] = cfg.healthcare_on and self.need["healthcare"] > 0.0
        # reset realised access + last tick's transfer for the new tick
        self.access = {"income_support": False, "healthcare": False}
        self.transfer = 0.0
        self.recompute_income()  # income now excludes any prior-tick transfer

    def wants_income_support(self, u: float) -> bool:
        """Step 3: eligible agent takes up with estimated baseline prob x awareness lever.

        ``u`` is a pre-drawn uniform (common random numbers) so matched factual/counterfactual
        runs consume identical randomness.
        """
        if not self.eligible["income_support"]:
            return False
        p = self.model.params.receipt_prob(
            log_market=math.log(max(self.y_market, 1.0)),
            no_market_income=self.y_market <= 0,
            health=self.health,
            education=self.education,
            region=self.region,
            sex=self.sex,
        )
        p = min(p * self.model.cfg.is_awareness, 1.0)
        return u < p

    def wants_healthcare(self, u: float) -> bool:
        """Step 3: agent with health need seeks care with prob = awareness x distance decay.

        With the spatial layer on, the seek-care probability is scaled by the distance-decay
        factor (agents far from their provider are less likely to access care).
        """
        if not self.eligible["healthcare"]:
            return False
        return u < self.model.cfg.hc_awareness * self.distance_access_factor

    def apply_direct_effects(self) -> None:
        """Step 4: realised access updates income (transfer) and health (access effect)."""
        cfg = self.model.cfg
        if self.access["income_support"]:
            # transient top-up to the minimum income; included in income via recompute,
            # NOT folded into y_benefit (which would persist/compound across ticks)
            self.transfer = max(cfg.is_min_income - self.income, 0.0)
            self.recompute_income()
        if self.access["healthcare"]:
            self.health = min(self.health + cfg.hc_access_health_effect, 1.0)

    def step_dynamics(self, draws: dict) -> None:
        """Step 5: sample next-period health, employment, and update market income.

        ``draws`` carries the pre-drawn uniforms/normals for this agent so that matched
        factual/counterfactual runs consume identical randomness (common random numbers).
        """
        p = self.model.params
        # health transition (fractional logit -> mean; add nothing stochastic for h to keep
        # the bounded mean dynamics; service effect already applied as a direct shift)
        self.health = p.health_next(
            health=self.health,
            age10=self.age10,
            employed=self.employed,
        )
        # employment transition (Bernoulli on estimated prob, common-random-number uniform)
        was_employed = self.employed
        prob_emp = p.employed_prob(
            employed=self.employed,
            health=self.health,
            age10=self.age10,
            education=self.education,
            region=self.region,
            sex=self.sex,
        )
        self.employed = draws["u_emp"] < prob_emp
        # Market income: continuing workers PERSIST their (heterogeneous) income; re-entrants
        # are assigned predicted earnings; the non-employed have zero market income. Persisting
        # avoids collapsing the real income distribution onto a deterministic Mincer mean.
        if self.employed and not was_employed:
            self.y_market = math.exp(
                p.log_earnings(
                    age10=self.age10,
                    health=self.health,
                    education=self.education,
                    region=self.region,
                    sex=self.sex,
                )
            )
        elif not self.employed:
            self.y_market = 0.0
        # (the service transfer is transient and is reset at the start of the next tick)
        self.age += 1.0
        self.recompute_income()

    def compute_utility(self) -> None:
        """Indirect utility for this tick (the welfare step of the annual cycle)."""
        self.utility = self.model.params.utility(income=self.income, health=self.health)
        self.utility_trajectory.append(self.utility)


class ServiceProvider(Agent):
    """Supply-side agent: finite capacity per region per tick, with a waiting list (ODD §7.3)."""

    def __init__(
        self, model: SocialServicesModel, region: str, service_type: str, capacity: int
    ) -> None:
        super().__init__(model)
        self.region = region
        self.service_type = service_type
        self.capacity = int(capacity)
        self.utilisation = 0
        self.waiting_list = 0

    def reset_period(self) -> None:
        """Clear per-tick utilisation/waiting counters before allocation."""
        self.utilisation = 0
        self.waiting_list = 0


class Policymaker(Agent):
    """The scenario lever (ODD §2.1): sets provider capacity and eligibility each tick.

    The allocation rule maps a population-based capacity-per-1000 to per-region
    provider capacity; eligibility thresholds are read directly from the PolicyConfig.
    """

    def __init__(self, model: SocialServicesModel) -> None:
        super().__init__(model)

    def set_capacities(self) -> None:
        """Step 1: set each healthcare provider's capacity from the supply lever."""
        cfg = self.model.cfg
        for prov in self.model.providers:
            if prov.service_type == "healthcare":
                pop = self.model.region_population.get(prov.region, 0)
                prov.capacity = int(round(cfg.hc_capacity_per_1000 * pop / 1000.0))
            prov.reset_period()
