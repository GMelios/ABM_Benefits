# ODD Protocol — Social-Services Agent-Based Model (Task T1.3)

*This document follows the ODD (Overview, Design concepts, Details) protocol of Grimm et al. and is the methodological backbone of deliverable D1.2. It is written to be precise enough for Claude Code to implement against. Wherever a parameter is "estimated," it is estimated from the **UK Household Longitudinal Study (Understanding Society, UKHLS)** panel; wherever welfare is computed, it uses the **Weighted Equivalent-Variation Measure (WEVM)** of the accompanying paper "The Value of Social Services: An Axiomatic Approach."*

---

## 0. Architecture (read first)

The model has **three layers** that must not be confused:

1. **Estimation layer (offline, from UKHLS).** Behavioural transition models (health, employment, income, take-up) and a utility/wellbeing equation are estimated from the UKHLS panel. These are *inputs* to the ABM, not part of its runtime.
2. **Behaviour layer (the ABM, Mesa).** Agents act each tick according to the estimated rules + policy levers, producing trajectories of income, health, employment, etc.
3. **Valuation layer (the WEVM).** The ABM is run in paired *factual* (service present) and *counterfactual* (service absent/reduced) configurations. For each agent, the difference in utility is converted to an equivalent variation $EV_i$; the $EV_i$ are aggregated by the WEVM into a single, distribution-sensitive, monetary value-added figure.

**The paper supplies layer 3 and the definition of the individual welfare object; it does *not* supply behavioural rules.** Those come from layer 1.

---

## 1. Purpose and patterns

**Purpose.** Simulate the supply and demand of social services across heterogeneous areas and quantify their impact on individual and social welfare *in monetary terms*, under alternative policy scenarios (funding levels, delivery models, eligibility rules). The model must support: (a) individual/community behavioural response to service availability; (b) ex-ante prediction of policy outcomes; (c) monetary valuation of value-added via the WEVM.

**Patterns used to judge whether the model is realistic** (pattern-oriented modelling — the model must reproduce *several simultaneously*, not just one):
- Take-up / receipt gradients by income and region observed in UKHLS.
- The cross-sectional and within-person health–employment association in UKHLS.
- The empirical distribution of life satisfaction / SF-6D health utility and its income gradient.
- Regional inequality in service access and in welfare.
- If available, the response to a known past reform (out-of-sample validation target).

---

## 2. Entities, state variables, and scales

### 2.1 Agent types and state variables

**Individual / Household agent** (the beneficiary; the unit of welfare)
| Variable | Meaning | UKHLS source / status |
|---|---|---|
| `region` | area of residence | GOR (EUL) or LAD (Special Licence) |
| `age`, `sex`, `hh_type`, `education` | demographics | UKHLS |
| `employed` | employment status | UKHLS `jbstat`-derived |
| `y` | current gross/equivalised income | UKHLS income variables |
| `y0` | **baseline income, fixed for weighting** | period-0 `y` |
| `h` | health stock = SF-6D utility ∈ [0,1] | derived from SF-12 |
| `ls` | wellbeing / utility proxy | life satisfaction, GHQ-12 |
| `need[service]` | need intensity per service | function of demographics/health |
| `eligible[service]` | eligibility flags | from policy rules |
| `access[service]` | access realised this tick | model-generated |
| `utility` | indirect utility (Sec. 7.7) | model-generated |
| `EV` | equivalent variation (counterfactual runs) | model-generated |

**Service Provider agent** (supply side)
`region`/catchment, `service_type`, `capacity` (served per tick), `utilisation`, `waiting_list`, `unit_cost`, `quality`. Initialised from open/administrative supply data, **not** from UKHLS.

**Policymaker agent** (the scenario lever)
`budget`, `allocation_rule` (how budget maps to provider capacity across areas/services), `eligibility_thresholds`, `delivery_model` parameters. Changing these *is* running a policy scenario.

**(Optional) Labour-market entity per region** — `vacancies`, matching parameters; mediates the indirect employment effect. Add only after the core loop validates.

### 2.2 Environment
A set of areas (regions or LADs) with attributes (`population`, `deprivation`, provider set). Represented spatially via **Mesa-Geo** (`GeoSpace` with polygon `GeoAgent`s) when distance/catchment matters, or as a region index otherwise.

### 2.3 Scales
- **Tick = 1 year**, to match UKHLS wave cadence (the natural frequency for the estimated transitions). Horizon: 10–20 years for policy projection.
- **Spatial unit** = Government Office Region by default; Local Authority District under Special Licence.

---

## 3. Process overview and scheduling

Each tick executes in this fixed order:

1. **Policy step.** Policymaker converts `budget` + `allocation_rule` into per-provider `capacity` and sets `eligibility_thresholds` for the period.
2. **Need & eligibility.** Each individual updates `need[·]`; `eligible[·]` is evaluated against thresholds.
3. **Take-up & allocation.** Eligible individuals with need attempt access (Sec. 7.2). Providers allocate scarce capacity by a priority/queue rule (Sec. 7.3); `access[·]` and `waiting_list` are set.
4. **Direct effects.** Realised access updates health/income/skills by estimated effect sizes (Sec. 7.4–7.6).
5. **Dynamics (indirect effects).** Health, employment, and income transitions are sampled from the estimated UKHLS models, conditional on prior state and access. This is where indirect effects propagate (e.g. health → employment → income → future need).
6. **Welfare.** Compute `utility` for every agent (Sec. 7.7). In a counterfactual-paired run, compute `EV` (Sec. 7.8) and the WEVM (Sec. 7.9).
7. **Observation.** Record the per-agent panel and area/aggregate metrics.

**Activation:** individuals are activated in random order each tick (Mesa `AgentSet.shuffle_do`). All stochastic draws use the model's single seeded RNG. Updates within a step are computed from start-of-tick state to avoid order artefacts where feasible.

---

## 4. Design concepts

- **Basic principles.** Welfare economics (money-metric equivalent variation; the WEVM as the social aggregator) for valuation; empirically estimated transition models for behaviour. The normative content lives in one transparent parameter, ε.
- **Emergence.** Aggregate and distributional welfare, regional inequality of access, and monetary value-added are *outputs* that emerge from agent interactions — they are never imposed.
- **Adaptation & objectives.** Individuals are **boundedly rational**: they follow estimated take-up rules, not utility maximisation. The policymaker follows an explicit allocation rule (the scenario lever).
- **Learning.** None by default. An optional extension is awareness diffusion through a neighbour network (raising take-up where neighbours have taken up).
- **Sensing.** Individuals sense local service availability, their own need, and their eligibility. They do **not** sense others' incomes or the global state.
- **Interaction.** Individual ↔ provider (capacity, queues, waiting lists); individual ↔ labour market (matching); optional neighbourhood spillovers in health/awareness.
- **Stochasticity.** Take-up and all state transitions are probabilistic, sampled from the estimated models; fully seeded for reproducibility.
- **Collectives.** Households and regions.
- **Observation.** Per-agent panel (`y, h, employed, utility, EV`), area aggregates, service `coverage` and `unmet_need`, costs, and the welfare outputs: `WEVM(ε)` for a grid of ε, the Atkinson index of welfare before/after, and net benefit (ε=0 benchmark).

---

## 5. Initialization

1. **Synthetic population.** Generate agents by reweighting/IPF of the UKHLS sample to match area marginal totals (e.g. age × sex × region, household type). Initial joint states (income, health, employment, education) are drawn from the UKHLS empirical joint distribution so that correlations are preserved. Validate that recovered margins match targets within tolerance **before** trusting any downstream result.
2. **Providers.** Instantiate from supply-side data (locations, capacities, costs by area).
3. **Baseline incomes `y0`** are recorded at t=0 and **held fixed** for all WEVM weighting, per the paper's fixed-benchmark requirement.
4. Fix all seeds.

---

## 6. Input data

| Input | Source | Role | Notes |
|---|---|---|---|
| Individual/household microdata | **UKHLS** | population synthesis + estimation of all behavioural transitions + utility equation | EUL covers GOR; finer geography needs Special Licence / Secure Lab. **Restricted — see governance rules in CLAUDE.md.** |
| Provider locations & capacity | open/administrative (NHS, local authority, etc.) | supply side | joined to areas spatially |
| Policy parameters | scenario design | levers: budgets, eligibility, delivery model | the things you vary |
| Validation targets | UKHLS aggregates + any reform series | calibration & out-of-sample test | |

---

## 7. Submodels

> Notation: each submodel is an estimated statistical model from UKHLS; the agent rule is "sample the next state from the fitted model." Functional forms below are defaults — Claude Code should confirm specifications against the estimated coefficients.

### 7.1 Need and eligibility
`need[service]` is a function of demographics and health (e.g. childcare need rises with young children; healthcare need rises as `h` falls). `eligible[service]` compares income/circumstance to `eligibility_thresholds`.

### 7.2 Take-up
Probability that an eligible individual with need accesses a service:
`P(access) = f(local availability, distance, awareness, need, income)`, a logit estimated/calibrated to reproduce UKHLS receipt rates. Capacity constraints are applied in 7.3.

### 7.3 Provider capacity allocation
Demand is matched to `capacity` by a priority rule (e.g. highest need first, or FIFO). Excess demand forms a `waiting_list` carried to the next tick; unmet need is recorded.

### 7.4 Health dynamics
Dynamic model of SF-6D utility:
`h_{t+1} = g(h_t, age, access_health, employed, income) + noise`, estimated from UKHLS panel (e.g. dynamic linear / fractional-response model since `h ∈ [0,1]`). Service access enters with the estimated effect size; this is the *direct* health effect.

### 7.5 Employment dynamics
Transition model:
`P(employed_{t+1}) = logit(employed_t, h_t, education, age, region labour conditions)`, estimated from wave-to-wave UKHLS transitions. Health entering this equation is the main channel for the *indirect* "better health → better employment" effect.

### 7.6 Income and transfers
`income = earnings(employed, education, …) + transfers(eligibility, policy)`. Earnings estimated from UKHLS; transfers are computed from the policy rules (this is where income-support-type services act directly).

### 7.7 Utility / wellbeing (the individual welfare function)
Specify indirect utility with income entering log-linearly and other arguments additively (the standard wellbeing-valuation form, estimable directly from UKHLS life satisfaction or SF-6D):

$$ v_i = \beta_y \ln(y_i) + \phi_i(\text{health}, \text{employment}, \text{service environment}, \dots) $$

The coefficients (including the marginal utility of income $\beta_y$) are estimated from a UKHLS life-satisfaction (or SF-6D) regression. **This is the layer the paper requires you to supply; the paper gives the EV and aggregation that sit on top of it.**

### 7.8 Equivalent variation (connecting behaviour to the paper)
For agent $i$, $EV_i$ is the income compensation that makes the agent indifferent between the **counterfactual** world (service absent, environment $S_0$) and the **factual** world (service present, environment $S_1$):

$$ v_i(y_i + EV_i,\, S_0) = v_i(y_i,\, S_1). $$

Let $\Delta_i$ be the utility difference the service produces (factual minus counterfactual), evaluated **over the simulated trajectory** as a discounted sum of per-period differences obtained from paired runs:

$$ \Delta_i = \sum_{t} \delta^{t}\big[v_{i,t}^{\,\text{factual}} - v_{i,t}^{\,\text{counterfactual}}\big]. $$

With the log-income utility of 7.7, EV has a **closed form**:

$$ \boxed{\,EV_i = y_i\left(e^{\Delta_i / \beta_y} - 1\right)\,} $$

(If instead income enters linearly, $EV_i = \Delta_i/\beta_y$ — a pure willingness-to-pay with no income effect. Which form to use is a documented modelling decision.)

**Operational requirement this imposes on the ABM:** the model must be runnable as a *matched pair* — identical seeds and initial population, differing only in the service/policy configuration — so that $v_{i,t}^{\text{factual}}$ and $v_{i,t}^{\text{counterfactual}}$ are comparable per agent.

### 7.9 Social valuation (the WEVM)
Aggregate the individual $EV_i$ using baseline incomes and the inequality-aversion parameter ε:

$$ \mathrm{WEVM}(EV, y^0;\varepsilon) = \frac{\sum_{i} \left(y_i^0/y^*\right)^{-\varepsilon} EV_i}{\sum_{i}\left(y_i^0/y^*\right)^{-\varepsilon}}, \qquad y^* = \text{mean baseline income (fixed)}. $$

Reporting conventions:
- Compute for a grid **ε ∈ {0, 0.5, 1, 1.5, 2}**. ε=0 is the utilitarian/net-benefit benchmark; larger ε is increasingly prioritarian.
- Report the **subgroup decomposition** (by region, income band, household type) using the paper's priority-mass weighting — *not* population shares.
- Report the **Atkinson index** of the welfare distribution before vs after, alongside the WEVM, to show how inequality moves.
- Multiply the per-person WEVM by population (or relate to programme cost) to get aggregate value-added and a WEVM-based benefit–cost comparison.

---

## 8. Calibration, validation, and uncertainty

1. **Sensitivity first (SALib).** Identify which parameters move outputs; fix the rest before calibrating.
2. **Calibrate** the few free/uncertain parameters (e.g. take-up, awareness) with ABC (`pyABC`) or `optuna` to match the UKHLS target patterns of §1 — fit *distributions* over parameters, not point estimates.
3. **Out-of-sample validation.** Replay a known past reform; check the model reproduces the observed change.
4. **Uncertainty.** Every reported number is a **mean over N seeded runs with a confidence interval**, propagated through the calibrated parameter posterior. No single-run claims. Report WEVM(ε) with uncertainty bands for each ε.

---

## 9. Known limitations to state in D1.2
- Behavioural rules are only as good as UKHLS identification allows; service-access effects estimated from observational panel data may be confounded — be explicit about identification assumptions.
- UKHLS has limited direct measurement of specific local service *use*; some services are proxied through outcomes or benefit receipt.
- The utility specification (7.7) is a modelling choice; report WEVM sensitivity to it as well as to ε.
- Monetary value-added is presented as a transparent, assumption-explicit figure, not a point truth — its credibility comes from the ε-grid, the utility-form sensitivity, and the validation evidence.
