"""First real run: calibrate -> screen -> evaluate both services -> parquet + figures.

Run: ``uv run python -m src.experiments.run`` (prints the results summary).

Pipeline (ODD §8; multi-seed means with confidence intervals):
  1. Light calibration of the take-up lever to the observed UKHLS receipt rate.
  2. SALib Morris screen of the free parameters (which move the WEVM).
  3. Matched-pair welfare for income support AND healthcare over N seeds.
  4. Write parquet outputs to results/ and figures to results/figures/.

Headline metric is the WEVM(eps) grid with confidence intervals. Utility uses the
de-overlapped spec (physical PCS health, employment excluded as an income mediator), so both
services have finite, usable closed-form WEVM; the healthcare magnitude reflects the calibrated access-to-health effect size.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.estimation.load_ukhls import load_panel  # noqa: E402
from src.experiments.calibrate import calibrate_takeup  # noqa: E402
from src.experiments.sensitivity import morris_screen  # noqa: E402
from src.model.config import PolicyConfig  # noqa: E402
from src.model.params import ModelParams  # noqa: E402
from src.welfare.runner import WelfareResult, evaluate_service  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
RESULTS = _REPO / "results"
FIGURES = RESULTS / "figures"

#: Observed UKHLS working-age benefit-receipt rate (from params/takeup.json diagnostics).
OBSERVED_RECEIPT_RATE = 0.358


def _write_results(res: WelfareResult) -> None:
    """Persist a service's welfare outputs as parquet (results/ is git-ignored)."""
    RESULTS.mkdir(exist_ok=True)
    s = res.service
    res.wevm_summary.to_parquet(RESULTS / f"wevm_{s}.parquet")
    res.wevm_linear_summary.to_parquet(RESULTS / f"wevm_linear_{s}.parquet")
    res.wevm_by_seed.to_parquet(RESULTS / f"wevm_by_seed_{s}.parquet")
    res.atkinson_summary.to_parquet(RESULTS / f"atkinson_{s}.parquet")
    res.coverage.to_parquet(RESULTS / f"coverage_{s}.parquet")
    res.subgroup.to_parquet(RESULTS / f"subgroup_{s}.parquet")
    res.ev_pooled.to_parquet(RESULTS / f"ev_pooled_{s}.parquet")


#: Above this magnitude the per-period closed form is treated as numerically degenerate
#: (health-channel exp overflow) and not plotted/reported as a headline number.
_DEGENERATE = 1e9


def _fig_wevm(res: WelfareResult, *, metric: str = "closed") -> bool:
    """Plot WEVM vs eps with 95% CI band. Returns False (and skips) if degenerate."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = res.wevm_summary if metric == "closed" else res.wevm_linear_summary
    if summary["wevm_mean"].abs().max() > _DEGENERATE:
        return False  # closed-form health-channel explosion, not a meaningful figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary["eps"], summary["wevm_mean"], "o-", color="#1f4e79", label="WEVM")
    ax.fill_between(
        summary["eps"],
        summary["ci_lo"],
        summary["ci_hi"],
        alpha=0.25,
        color="#1f4e79",
        label="95% CI",
    )
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("inequality aversion ε")
    ax.set_ylabel("WEVM (£ per person)")
    label = "per-period closed form" if metric == "closed" else "linear money-metric"
    ax.set_title(f"WEVM(ε), {res.service.replace('_', ' ')} ({label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"wevm_{metric}_{res.service}.png", dpi=130)
    plt.close(fig)
    return True


def _fig_ev_distribution(res: WelfareResult) -> None:
    """Histogram of per-agent EV (clipped at the 99th pct for display)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    ev = res.ev_pooled["EV"]
    hi = ev.quantile(0.99)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(ev.clip(upper=hi), bins=60, color="#2e7d32", alpha=0.8)
    ax.axvline(float(ev.mean()), color="#b00020", lw=1.5, label=f"mean £{ev.mean():,.0f}")
    ax.set_xlabel("equivalent variation EV_i (£, clipped at p99)")
    ax.set_ylabel("agents")
    ax.set_title(f"EV distribution, {res.service.replace('_', ' ')}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"ev_distribution_{res.service}.png", dpi=130)
    plt.close(fig)


def _fig_coverage(res: WelfareResult) -> None:
    """Bar chart of coverage / access by region."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    cov = res.coverage.copy()
    col = "coverage" if "coverage" in cov.columns else "access_rate"
    cov = cov.sort_values(col)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(cov["region"], cov[col], color="#6a1b9a", alpha=0.85)
    ax.set_xlabel(col.replace("_", " "))
    ax.set_title(f"{col.replace('_', ' ').title()} by region, {res.service.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(FIGURES / f"coverage_{res.service}.png", dpi=130)
    plt.close(fig)


def main(
    *,
    n_agents: int = 2500,
    horizon: int = 10,
    n_seeds: int = 20,
    calib_trials: int = 25,
    screen_trajectories: int = 5,
) -> None:
    """Execute the full experiment pipeline and print the results summary."""
    panel = load_panel()
    params = ModelParams.load()
    base_cfg = PolicyConfig(n_agents=n_agents, horizon=horizon)

    # 1. light calibration of the take-up lever ----------------------------------------
    print("... calibrating take-up (is_awareness) to observed receipt rate")
    calib = calibrate_takeup(
        panel,
        params,
        base_cfg,
        target_rate=OBSERVED_RECEIPT_RATE,
        n_trials=calib_trials,
    )
    cfg = base_cfg.with_(is_awareness=calib.value)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "calibration.json").write_text(json.dumps(calib.__dict__, indent=2))

    # 2. sensitivity screen (lighter population) ----------------------------------------
    print("... Morris sensitivity screen")
    screen_cfg = cfg.with_(n_agents=1500)
    screen_is = morris_screen(
        panel, params, screen_cfg, service="income_support", n_trajectories=screen_trajectories
    )
    screen_hc = morris_screen(
        panel, params, screen_cfg, service="healthcare", n_trajectories=screen_trajectories
    )
    screen_is.to_parquet(RESULTS / "sensitivity_income_support.parquet")
    screen_hc.to_parquet(RESULTS / "sensitivity_healthcare.parquet")

    # 3. evaluate both services over N seeds --------------------------------------------
    seeds = list(range(n_seeds))
    print(f"... evaluating income support over {n_seeds} seeds")
    res_is = evaluate_service(panel, params, cfg, service="income_support", seeds=seeds)
    print(f"... evaluating healthcare over {n_seeds} seeds")
    res_hc = evaluate_service(panel, params, cfg, service="healthcare", seeds=seeds)

    # 4. write outputs + figures --------------------------------------------------------
    for res in (res_is, res_hc):
        _write_results(res)
        _fig_wevm(res, metric="closed")
        _fig_wevm(res, metric="linear")
        _fig_ev_distribution(res)
        _fig_coverage(res)

    _print_checkin3(calib, screen_is, screen_hc, res_is, res_hc, cfg)


def _print_checkin3(calib, screen_is, screen_hc, res_is, res_hc, cfg) -> None:
    """Print the results summary."""
    print("\n" + "=" * 78 + "\nRESULTS\n" + "=" * 78)
    print(
        f"\nConfig: n_agents={cfg.n_agents}, horizon={cfg.horizon}y, "
        f"seeds={res_is.n_seeds}, delta={cfg.discount_delta}"
    )
    print(
        f"\nCalibration: is_awareness={calib.value:.3f} -> simulated receipt "
        f"{calib.achieved_rate:.1%} vs UKHLS target {calib.target_rate:.1%}"
    )

    print("\nMost influential parameters (Morris mu_star on WEVM @eps=1):")
    print(
        "  income support:",
        ", ".join(f"{r.param}={r.mu_star:,.0f}" for r in screen_is.head(3).itertuples()),
    )
    print(
        "  healthcare    :",
        ", ".join(f"{r.param}={r.mu_star:,.0f}" for r in screen_hc.head(3).itertuples()),
    )

    for res in (res_is, res_hc):
        print(f"\n### WEVM(ε), {res.service}  (£ per person, mean [95% CI])")
        degenerate = res.wevm_summary["wevm_mean"].abs().max() > _DEGENERATE
        if degenerate:
            print(
                "  closed-form (per-period EV): not reported (health-channel exp overflow), "
                "(β_health·Δh ÷ small β_y); use the linear money-metric below."
            )
        else:
            print("  closed-form (per-period EV):")
            for r in res.wevm_summary.itertuples():
                print(f"    ε={r.eps:<3}  £{r.wevm_mean:12,.0f}  [{r.ci_lo:,.0f}, {r.ci_hi:,.0f}]")
        print("  linear money-metric:")
        for r in res.wevm_linear_summary.itertuples():
            print(f"    ε={r.eps:<3}  £{r.wevm_mean:12,.0f}  [{r.ci_lo:,.0f}, {r.ci_hi:,.0f}]")

    print("\nAtkinson index of income (before vs after), income support:")
    for r in res_is.atkinson_summary.itertuples():
        print(f"    ε={r.eps}: {r.atkinson_before:.4f} -> {r.atkinson_after:.4f}")

    for res in (res_is, res_hc):
        print(f"\n! {res.service}: " + " ".join(res.notes[-1:]))
    print(f"\nWrote parquet results to {RESULTS} and figures to {FIGURES}")


if __name__ == "__main__":
    main()
