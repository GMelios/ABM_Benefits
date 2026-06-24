"""Cost-effectiveness verdict: WEVM vs programme cost for each service (ODD §7.9).

Run: ``uv run python -m src.experiments.run_cost_effectiveness``

For each service, over N seeded matched pairs (closed-cohort baseline, the cleanest WEVM):
  - the per-person WEVM(ε) (annualised to real £), and
  - the per-person programme cost (transfers + admin; accesses x PSSRU unit cost),
  giving the benefit-cost ratio and net social benefit per ε. BCR >= 1 => cost-effective.

Writes results/cost_effectiveness_<service>.parquet and prints the verdict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.estimation.load_ukhls import load_panel
from src.model.config import PolicyConfig
from src.model.params import ModelParams
from src.welfare import costs
from src.welfare.runner import compute_welfare, matched_pair

_REPO = Path(__file__).resolve().parents[2]
RESULTS = _REPO / "results"

# calibrated take-up (see run_tier2 / calibrate); used so the income-support scale is realistic.
IS_AWARENESS = 1.56


def _evaluate(panel, params, cfg, *, service, seeds):
    """Mean WEVM(ε) grid and mean per-person programme cost over seeds, for one service."""
    wevm_rows, cost_pp = [], []
    for s in seeds:
        pf, pcf = matched_pair(panel, params, cfg, service=service, seed=s)
        res = compute_welfare(pf, pcf, params=params, cfg=cfg, service=service)
        wevm_rows.append(res["wevm"])
        n = res["ev_df"].shape[0]  # WEVM denominator (agents valued)
        cost_pp.append(
            costs.programme_cost_per_person(
                pf, service=service, cfg=cfg, delta=cfg.discount_delta, n_persons=n
            )
        )
    wevm_mean = {e: float(np.mean([w[e] for w in wevm_rows])) for e in wevm_rows[0]}
    return wevm_mean, float(np.mean(cost_pp))


def main(*, n_agents: int = 2500, horizon: int = 10, n_seeds: int = 10) -> None:
    """Run the cost-effectiveness verdict for both services and print it."""
    RESULTS.mkdir(exist_ok=True)
    panel = load_panel()
    params = ModelParams.load()
    cfg = PolicyConfig(n_agents=n_agents, horizon=horizon, is_awareness=IS_AWARENESS)
    seeds = list(range(n_seeds))

    print(
        "=" * 78
        + "\nCOST-EFFECTIVENESS VERDICT (WEVM vs programme cost, annual £/person)\n"
        + "=" * 78
    )
    print(
        f"config: n_agents={n_agents}, horizon={horizon}y, seeds={n_seeds}, δ={cfg.discount_delta}"
    )
    print(
        "benefit annualised x12 (monthly->annual £); costs: transfers+3% admin / "
        f"accesses x £{cfg.hc_unit_cost:.0f} (PSSRU-based)\n"
    )

    for service in ("income_support", "healthcare"):
        wevm, cost_pp = _evaluate(panel, params, cfg, service=service, seeds=seeds)
        ce = costs.cost_effectiveness(wevm, cost_pp)
        ce.to_parquet(RESULTS / f"cost_effectiveness_{service}.parquet")
        print(f"### {service}   (programme cost ~ £{cost_pp:,.0f}/person PV)")
        for r in ce.itertuples():
            verdict = "cost-effective" if r.cost_effective else "NOT cost-effective"
            print(
                f"   ε={r.eps:<3} benefit £{r.benefit_per_person:>8,.0f}  BCR {r.bcr:>5.2f}  "
                f"net £{r.net_social_benefit:>+8,.0f}  -> {verdict}"
            )
        print()

    print("A pure cash transfer returns ~ its cost at ε=0 (BCR~1 minus admin) and")
    print("becomes cost-effective as ε rises (it targets the poor). Healthcare BCR scales with")
    print("the calibrated access-to-health effect.")
    print("\nThresholds for context: NICE £20-30k/QALY; HMT QALY £70k; WELLBY £13k.")


if __name__ == "__main__":
    main()
