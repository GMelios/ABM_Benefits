"""Demonstration: the health->mortality channel, healthcare extends life, valued separately.

Run: ``uv run python -m src.experiments.run_mortality``

With ``health_affects_mortality`` on, healthcare lowers the death hazard (sickest face
``health_mortality_hr``x the healthiest), so the service prevents deaths. Welfare is reported
in two parts on the INITIAL COHORT:
  - within-life WEVM (income/health while alive, per-period closed form), and
  - a life-extension value (extra living-utility years vs a dead reference, LINEAR money-metric,
    reported at eps=0 as it is robust; high-eps WEVM is sensitive to near-zero-income weights).

Mortality is age+health only (no UKHLS mortality data) -> illustrative; the magnitude scales
with the calibrated access-to-health effect.
"""

from __future__ import annotations

import numpy as np

from src.estimation.load_ukhls import load_panel
from src.model.config import PolicyConfig
from src.model.params import ModelParams
from src.welfare.runner import compute_welfare, matched_pair


def main(*, n_agents: int = 2500, horizon: int = 20, seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    """Run the health->mortality demonstration over ``seeds`` and print a summary."""
    panel = load_panel()
    params = ModelParams.load()
    cfg = PolicyConfig(
        n_agents=n_agents,
        horizon=horizon,
        population_dynamics=True,
        health_affects_mortality=True,
        health_mortality_hr=4.0,
        hc_awareness=0.8,
        hc_capacity_per_1000=120.0,
        hc_access_health_effect=0.02,
    )
    prevented, lyg, le0, within0, within1 = [], [], [], [], []
    for s in seeds:
        pf, pcf = matched_pair(panel, params, cfg, service="healthcare", seed=s)
        cohort = set(pf.loc[pf["tick"] == 0, "agent_id"])
        ft = pf["tick"].max()
        df_p = len(cohort - set(pf[(pf["tick"] == ft) & pf["agent_id"].isin(cohort)]["agent_id"]))
        cf_p = len(
            cohort - set(pcf[(pcf["tick"] == ft) & pcf["agent_id"].isin(cohort)]["agent_id"])
        )
        prevented.append(cf_p - df_p)
        res = compute_welfare(pf, pcf, params=params, cfg=cfg, service="healthcare")
        lyg.append(res["life_years_gained_mean"])
        le0.append(res["wevm_life_extension"][0.0])
        within0.append(res["wevm"][0.0])
        within1.append(res["wevm"][1.0])

    print(
        "=" * 70 + "\nHEALTH -> MORTALITY CHANNEL (healthcare, baseline access, 20y)\n" + "=" * 70
    )
    print(
        f"seeds={list(seeds)}, n_agents={n_agents}, hazard ratio (sick:healthy)={cfg.health_mortality_hr}"
    )
    print(f"\ncohort deaths PREVENTED by care: {np.mean(prevented):.0f}  (of {n_agents})")
    print(f"mean discounted life-years gained / person: {np.mean(lyg):.4f}")
    print("\nHealthcare welfare, £ per person (mean over seeds):")
    print(
        f"  within-life WEVM     ε=0: £{np.mean(within0):>9,.0f}   ε=1: £{np.mean(within1):>9,.0f}"
    )
    print(f"  life-extension value ε=0: £{np.mean(le0):>9,.0f}   (robust net-benefit value)")
    print(f"  TOTAL (ε=0)              : £{np.mean(within0) + np.mean(le0):>9,.0f}")
    print("\nMagnitudes scale with the hc_access_health_effect; the")
    print("life-extension high-ε WEVM is sensitive to near-zero-income priority weights.")


if __name__ == "__main__":
    main()
