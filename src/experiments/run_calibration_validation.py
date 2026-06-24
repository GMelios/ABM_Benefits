"""Runner for ABC posterior calibration with uncertainty-propagated WEVM and validation.

Run: ``uv run python -m src.experiments.run_tier2``

Produces (to results/):
  - the ABC posterior for the take-up lever (samples + summary),
  - income-support WEVM(eps) bands that include PARAMETER uncertainty (posterior-propagated),
  - the out-of-sample forward cohort validation table,
  - the illustrative reform direction check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.estimation.load_ukhls import load_panel
from src.experiments.abc_calibrate import abc_calibrate_takeup, posterior_wevm
from src.experiments.validate import forward_validation, reform_direction_check
from src.model.config import PolicyConfig
from src.model.params import ModelParams

_REPO = Path(__file__).resolve().parents[2]
RESULTS = _REPO / "results"
OBSERVED_RECEIPT_RATE = 0.358


def main(
    *,
    n_agents: int = 2000,
    horizon: int = 10,
    abc_population: int = 50,
    abc_max_pops: int = 6,
    wevm_draws: int = 15,
    target_wave: int = 5,
) -> None:
    """Run the calibration and validation pipeline and print a summary."""
    RESULTS.mkdir(exist_ok=True)
    panel = load_panel()
    params = ModelParams.load()
    base_cfg = PolicyConfig(n_agents=n_agents, horizon=horizon)

    print("... ABC posterior calibration of take-up (is_awareness)")
    post = abc_calibrate_takeup(
        panel,
        params,
        base_cfg,
        target_rate=OBSERVED_RECEIPT_RATE,
        population_size=abc_population,
        max_populations=abc_max_pops,
    )
    pd.DataFrame({"is_awareness": post.samples, "weight": post.weights}).to_parquet(
        RESULTS / "abc_posterior_is_awareness.parquet"
    )

    print("... propagating posterior into income-support WEVM bands")
    pw = posterior_wevm(panel, params, base_cfg, post, service="income_support", n_draws=wevm_draws)
    pw.to_parquet(RESULTS / "wevm_posterior_income_support.parquet")

    print("... forward out-of-sample cohort validation")
    val = forward_validation(
        panel,
        params,
        base_cfg.with_(is_awareness=post.posterior_mean),
        base_wave=1,
        target_wave=target_wave,
    )
    val.table.to_parquet(RESULTS / "validation_forward.parquet")

    print("... reform direction check")
    reform = reform_direction_check(panel, params, base_cfg.with_(is_awareness=post.posterior_mean))
    (RESULTS / "validation_reform.json").write_text(
        json.dumps(
            {
                "observed_change": reform.observed_change,
                "simulated_change": reform.simulated_change,
                "direction_agrees": reform.direction_agrees,
                "moment": reform.moment,
            },
            indent=2,
        )
    )

    _print_summary(post, pw, val, reform)


def _print_summary(post, pw, val, reform) -> None:
    print("\n" + "=" * 78 + "\nABC CALIBRATION AND VALIDATION\n" + "=" * 78)
    print(
        f"\nABC posterior for is_awareness ({post.n_populations} populations, "
        f"eps={post.epsilon:.4f}):"
    )
    print(
        f"  mean={post.posterior_mean:.3f}  sd={post.posterior_sd:.3f}  "
        f"95% CI=[{post.cred_interval[0]:.3f}, {post.cred_interval[1]:.3f}]  "
        f"(target receipt {post.target_rate:.1%})"
    )

    print("\nIncome-support WEVM(ε) with PARAMETER + seed uncertainty (£/person):")
    for r in pw.itertuples():
        print(f"  ε={r.eps:<3}  £{r.wevm_mean:8,.0f}  [95%: {r.ci_lo:,.0f}, {r.ci_hi:,.0f}]")

    print(
        f"\nForward out-of-sample validation (wave {val.base_wave} cohort -> wave "
        f"{val.target_wave}, surviving cohort):"
    )
    print(val.table.round(3).to_string(index=False))

    print(f"\nReform direction check ({reform.moment}):")
    print(f"  observed (austerity window): {reform.observed_change:+.3f}")
    print(f"  simulated (eligibility tightening): {reform.simulated_change:+.3f}")
    print(f"  direction agrees: {reform.direction_agrees}")
    print(f"\nWrote calibration and validation outputs to {RESULTS}")


if __name__ == "__main__":
    main()
