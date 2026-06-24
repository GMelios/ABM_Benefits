# Valuing Social Services: An Agent-Based Model

An agent-based model, developed under the BENEFITS project, that simulates the supply and
demand of social services across heterogeneous UK areas and quantifies their welfare impact
**in monetary terms** using the Weighted Equivalent-Variation Measure (WEVM).

The model has three layers (see `docs/ODD_protocol.md` for the full specification):

1. **Estimation** (`src/estimation/`): behavioural transition models and a wellbeing equation
   estimated from UK longitudinal data (UKHLS), written to `params/*.json`.
2. **Behaviour** (`src/model/`): a Mesa agent-based model in which agents act each year on the
   estimated rules and the policy levers in force.
3. **Valuation** (`src/welfare/`): matched factual and counterfactual runs give each agent's
   equivalent variation, aggregated by the WEVM across a grid of inequality-aversion values.

The synthetic population is built in `src/population/`; calibration, sensitivity analysis, the
welfare runs, and the cost-effectiveness verdict are in `src/experiments/`.

## Data access

The model is estimated from the **UK Household Longitudinal Study (UKHLS, Understanding
Society)**, which is restricted under the UK Data Service End User Licence and is **not**
distributed with this repository. The repository ships the estimated coefficients
(`params/*.json`), which the behaviour and valuation layers consume, together with the
estimation scripts that regenerate them. To run the parts that read the survey directly,
obtain the data and place it at `data/ukhls_panel_long.parquet`. See
[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) for how to obtain it, the expected schema, and
the required citation. Without the file, the code imports and the logic-only tests run; the
data-dependent steps and tests skip automatically.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                  # core environment (Python 3.12, pinned via uv.lock)
uv sync --extra abc      # optional: pyABC for posterior calibration
uv sync --extra spatial  # optional: geopandas + mesa-geo for the spatial layer
```

## Running the model

```bash
# Estimate the behavioural and wellbeing models from UKHLS -> params/
uv run python -m src.estimation.estimate

# Inspect the harmonised panel
uv run python -m src.estimation.load_ukhls

# Build and validate the synthetic population
uv run python -m src.population.synthesize

# Welfare run: calibrate, screen sensitivity, and compute WEVM over N seeds -> results/
uv run python -m src.experiments.run

# Posterior calibration (ABC) and out-of-sample validation
uv run python -m src.experiments.run_calibration_validation

# Health-to-mortality channel: healthcare extends life, valued separately
uv run python -m src.experiments.run_mortality

# Spatial layer: distance to providers drives access inequality
uv run python -m src.experiments.run_spatial

# Cost-effectiveness verdict: WEVM against programme cost -> benefit-cost ratio
uv run python -m src.experiments.run_cost_effectiveness
```

Scenarios are configuration-driven (`experiments/configs/*.yaml`):

```python
from src.model.config import PolicyConfig
cfg = PolicyConfig.from_yaml("experiments/configs/baseline.yaml")
```

## Tests and quality

```bash
uv run pytest            # unit, invariant, and fixed-seed regression tests
uv run ruff check .      # lint
uv run black --check .   # format
```

## Outputs

For each service (income support, healthcare), over N seeded matched pairs reported with 95
per cent confidence intervals: the WEVM across the inequality-aversion grid, the distribution
of equivalent variations, coverage and unmet need by region, the priority-mass subgroup
decomposition, and the Atkinson index of income before and after. The cost-effectiveness
verdict expresses these against programme cost as a benefit-cost ratio.

The methodology is documented in `docs/ODD_protocol.md` and the variable reference in
`docs/data_dictionary.md`. The project report is in `reports/`.

## Licence

The code, configuration, estimated coefficients, and documentation are released under the MIT
Licence (see `LICENSE`). The licence does not cover the UKHLS data, which remains governed by
the UK Data Service licence terms (`docs/DATA_ACCESS.md`).
