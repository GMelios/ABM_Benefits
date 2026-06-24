"""The social-services ABM (Mesa 3.x), behaviour layer (ODD §3 tick order).

Builds Individuals from the synthetic population, healthcare ServiceProviders per region,
and a Policymaker. Each tick runs the fixed ODD §3 order:

    1 policy -> 2 need/eligibility -> 3 take-up/allocation -> 4 direct effects
      -> 5 dynamics -> 6 welfare -> 7 observation  [-> 5b demography, when enabled]

When ``cfg.population_dynamics`` is set, a demography step runs after observation: age-based
deaths exit and an equal number of young entrants join (stationary N, ODD §2). Mortality is
age-only, so deaths are identical across matched runs (CRN) and services don't change who
dies under this configuration (a health-to-mortality channel is available as an extension).

Reproducibility: one model RNG seeded at construction drives activation
order; each agent additionally has a deterministic per-agent stream (``prng``) derived from
(seed, agent_id) so that matched factual/counterfactual runs consume identical randomness
(common random numbers), isolating the service's causal effect, including under turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mesa import Model

from src.model.agents import Individual, Policymaker, ServiceProvider
from src.model.config import PolicyConfig
from src.model.params import ModelParams


class SocialServicesModel(Model):
    """Mesa model coupling estimated behaviour, policy levers, and welfare observation."""

    def __init__(
        self,
        population: pd.DataFrame,
        params: ModelParams,
        cfg: PolicyConfig,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__(rng=seed)
        self.cfg = cfg
        self.params = params
        self._seed = seed
        self.tick = 0

        # build individuals (whole-record init preserves the empirical joint, ODD §5.1).
        # Each agent self-seeds its own RNG stream from (seed, agent_id), see Individual.
        self.n_agents = len(population)
        for row in population.to_dict("records"):
            Individual(self, row)

        self.individuals = self.agents_by_type[Individual]
        self.region_population = self._region_counts()

        # population dynamics setup (entry/exit), only when enabled. Entrants are drawn from
        # the YOUNG subset of the (already survey-weight-representative) synthetic population,
        # so entry needs no extra data and preserves the empirical joint of young people.
        self._next_birth_id = self.n_agents
        self._entry_pool: pd.DataFrame | None = None
        self._entry_rng: np.random.Generator | None = None
        if cfg.population_dynamics:
            young = population[population["dvage"] <= cfg.entry_max_age].reset_index(drop=True)
            if young.empty:
                raise ValueError(
                    f"No agents <= age {cfg.entry_max_age} to seed the entry pool; "
                    "increase n_agents or entry_max_age."
                )
            self._entry_pool = young
            self._entry_rng = np.random.default_rng([seed, 10_000_019])  # dedicated entry stream

        # healthcare providers: one per region present
        self.providers: list[ServiceProvider] = [
            ServiceProvider(self, region=r, service_type="healthcare", capacity=0)
            for r in sorted(self.region_population)
        ]
        self.policymaker = Policymaker(self)

        # spatial layer (optional): a mesa_geo GeoSpace of provider GeoAgents at region
        # centroids, the geometry seam for polygon catchments. Access uses precomputed
        # per-agent distance-decay (set on each Individual at construction).
        self.geospace = None
        if cfg.spatial:
            from src.model.space import build_provider_geospace

            self.geospace = build_provider_geospace(self, self.providers)

        self._records: list[dict] = []
        self._tick_draws: dict[int, dict[str, float]] = {}

    def _region_counts(self) -> dict[str, int]:
        """Return the current count of individuals per region (for capacity allocation)."""
        counts: dict[str, int] = {}
        for a in self.agents_by_type[Individual]:
            counts[a.region] = counts.get(a.region, 0) + 1
        return counts

    # -- tick ---------------------------------------------------------------------------
    def step(self) -> None:
        """Run one tick in the fixed ODD §3 order (+ demography when enabled)."""
        self.individuals = self.agents_by_type[
            Individual
        ]  # refresh (population may have turned over)
        if self.cfg.population_dynamics:
            self.region_population = self._region_counts()
        self.policymaker.set_capacities()  # 1 policy
        self.individuals.shuffle_do("update_need_eligibility")  # 2 need & eligibility
        self._draw_tick_randomness()
        self._take_up_and_allocate()  # 3 take-up & allocation
        self.individuals.shuffle_do("apply_direct_effects")  # 4 direct effects
        for a in self.individuals:  # 5 dynamics (order-invariant)
            a.step_dynamics(self._tick_draws[a.unique_id])
        self.individuals.do("compute_utility")  # 6 welfare
        self._record()  # 7 observation
        if self.cfg.population_dynamics:
            self._demography()  # 5b demography: deaths exit, young entrants join (next tick)
        self.tick += 1

    def _draw_tick_randomness(self) -> None:
        """Draw a fixed per-agent uniform set (CRN): u_is, u_hc, u_emp (+ u_mort if dynamics)."""
        self._tick_draws = {}
        dynamics = self.cfg.population_dynamics
        for a in self.individuals:
            g = a.prng
            draws = {
                "u_is": float(g.random()),
                "u_hc": float(g.random()),
                "u_emp": float(g.random()),
            }
            if dynamics:
                draws["u_mort"] = float(g.random())
            self._tick_draws[a.unique_id] = draws

    def _take_up_and_allocate(self) -> None:
        """Step 3: take-up decisions then capacity-constrained allocation (ODD §7.2-7.3)."""
        # --- income support: eligible takers, rationed lowest-income-first if budget binds ---
        is_takers = [
            a
            for a in self.individuals
            if a.wants_income_support(self._tick_draws[a.unique_id]["u_is"])
        ]
        is_takers.sort(key=lambda a: a.income)  # priority: lowest income first
        budget = self.cfg.is_budget
        spent = 0.0
        for a in is_takers:
            top_up = max(self.cfg.is_min_income - a.income, 0.0)
            if budget is not None and spent + top_up > budget:
                continue  # budget exhausted; remaining go unserved (unmet need recorded)
            a.access["income_support"] = True
            spent += top_up

        # --- healthcare: per-region, highest-need-first up to provider capacity (queue) ---
        prov_by_region = {p.region: p for p in self.providers}
        hc_takers: dict[str, list[Individual]] = {}
        for a in self.individuals:
            if a.wants_healthcare(self._tick_draws[a.unique_id]["u_hc"]):
                hc_takers.setdefault(a.region, []).append(a)
        for region, takers in hc_takers.items():
            prov = prov_by_region.get(region)
            if prov is None:
                continue
            takers.sort(key=lambda a: a.need["healthcare"], reverse=True)
            served = takers[: max(prov.capacity, 0)]
            for a in served:
                a.access["healthcare"] = True
            prov.utilisation = len(served)
            prov.waiting_list = max(len(takers) - prov.capacity, 0)

    # -- observation --------------------------------------------------------------------
    def _record(self) -> None:
        """Append the per-agent panel row for this tick (ODD §4 Observation)."""
        for a in self.individuals:
            self._records.append(
                {
                    "tick": self.tick,
                    "agent_id": a.aid,  # stable birth id (matches across factual/counterfactual)
                    "region": a.region,
                    "age": a.age,
                    "income": a.income,
                    "y_market": a.y_market,
                    "y_benefit": a.y_benefit,
                    "health": a.health,
                    "employed": a.employed,
                    "utility": a.utility,
                    "y0": a.y0,
                    "transfer": a.transfer,  # income-support top-up paid this tick (£/month)
                    "access_income_support": a.access["income_support"],
                    "access_healthcare": a.access["healthcare"],
                    "need_healthcare": a.need["healthcare"],
                }
            )

    # -- demography (population dynamics, ODD §2) ---------------------------------------
    def _demography(self) -> None:
        """Step 5b: age-based deaths exit; an equal number of young entrants join.

        Deaths use the per-agent ``u_mort`` draw against an age-only mortality schedule, so
        they are IDENTICAL across matched factual/counterfactual runs (CRN), the service
        cannot change who dies under this configuration. Entrants (one per death, stationary N) are drawn from
        the young-population pool via the dedicated entry stream, in a deterministic order, so
        the entry sequence is also identical across arms. New agents act from the next tick.
        """
        from src.model.demography import mortality_prob

        hr = self.cfg.health_mortality_hr if self.cfg.health_affects_mortality else 1.0
        living = sorted(self.agents_by_type[Individual], key=lambda a: a.aid)
        deaths = [
            a
            for a in living
            if self._tick_draws[a.unique_id]["u_mort"]
            < mortality_prob(
                a.age,
                multiplier=self.cfg.mortality_multiplier,
                health=(a.health if self.cfg.health_affects_mortality else None),
                hazard_ratio=hr,
            )
        ]
        for a in deaths:
            a.remove()
        for _ in deaths:
            self._spawn_entrant()

    def _spawn_entrant(self) -> None:
        """Create one young entrant drawn (uniformly) from the young-population pool."""
        pool = self._entry_pool
        i = int(self._entry_rng.integers(len(pool)))
        row = pool.iloc[i].to_dict()
        aid = self._next_birth_id
        self._next_birth_id += 1
        row["agent_id"] = aid
        row["y0"] = float(row["income_net"])  # baseline income recorded at entry (ODD §5.3)
        Individual(self, row)

    def run(self, horizon: int | None = None) -> pd.DataFrame:
        """Run ``horizon`` ticks (default cfg.horizon) and return the per-agent panel."""
        horizon = self.cfg.horizon if horizon is None else horizon
        for _ in range(horizon):
            self.step()
        return self.panel()

    def panel(self) -> pd.DataFrame:
        """Return the recorded per-agent x tick panel as a DataFrame."""
        return pd.DataFrame(self._records)
