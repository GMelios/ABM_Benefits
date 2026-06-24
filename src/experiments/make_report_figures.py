"""Generate consistent annual-£ figures for the D1.2 partner report / briefing.

Numbers are the canonical results from ``run_cost_effectiveness`` (10 seeds, n=2500, 10y),
``run_spatial`` (3 seeds), and ``run_tier2`` forward validation, reproduced here as constants
so the report figures are stable and don't require a re-run. Saves to results/figures/report_*.

Run: ``uv run python -m src.experiments.make_report_figures``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parents[2] / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

NAVY, BURGUNDY, TEAL, GREY = "#1f3a5f", "#7b1e3a", "#2a7f7f", "#8a8a8a"

EPS = [0.0, 0.5, 1.0, 1.5, 2.0]
# annual £/person, income support (run_cost_effectiveness, 10 seeds)
IS_BENEFIT = [20632, 49232, 61521, 62257, 62322]
IS_BCR = [0.97, 2.32, 2.89, 2.93, 2.93]
IS_COST = 21259
# spatial coverage by distance band (run_spatial, 3 seeds)
BANDS = ["<20km", "20-40km", "40-60km", "60km+"]
COVERAGE = [0.69, 0.49, 0.36, 0.19]
# forward validation, wave-1 cohort -> wave 5 (run_tier2)
VAL_MOMENTS = ["employment", "mean health", "median income\n(scaled)"]
VAL_OBS = [0.723, 0.490, 1.260]
VAL_SIM = [0.748, 0.497, 1.200]


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGDIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGDIR / name)


def fig_wevm_bcr():
    """Income-support WEVM(ε) (annual £) + benefit-cost ratio vs ε."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar([str(e) for e in EPS], IS_BENEFIT, color=NAVY)
    ax1.axhline(IS_COST, color=BURGUNDY, ls="--", lw=1.5, label=f"programme cost £{IS_COST:,}")
    ax1.set_title("Income support: WEVM by inequality aversion ε")
    ax1.set_xlabel("ε (0 = utilitarian, higher = pro-poor)")
    ax1.set_ylabel("Welfare value, £/person/yr")
    ax1.legend(fontsize=8)
    ax2.plot([str(e) for e in EPS], IS_BCR, "o-", color=TEAL, lw=2)
    ax2.axhline(1.0, color=GREY, ls="--", lw=1.2)
    ax2.set_title("Benefit-cost ratio (BCR >= 1 => cost-effective)")
    ax2.set_xlabel("ε")
    ax2.set_ylabel("BCR")
    for x, y in zip([str(e) for e in EPS], IS_BCR, strict=True):
        ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=8)
    _save(fig, "report_wevm_bcr_income_support.png")


def fig_spatial():
    """Healthcare coverage by distance to provider (within-region inequality)."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(BANDS, COVERAGE, color=[TEAL, NAVY, NAVY, BURGUNDY])
    ax.set_title("Spatial layer: healthcare coverage falls with distance")
    ax.set_xlabel("distance to provider")
    ax.set_ylabel("coverage among agents with need")
    ax.set_ylim(0, 0.8)
    for x, y in zip(BANDS, COVERAGE, strict=True):
        ax.annotate(
            f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 4), fontsize=9, ha="center"
        )
    _save(fig, "report_spatial_coverage.png")


def fig_validation():
    """Out-of-sample forward validation: observed vs simulated."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(VAL_MOMENTS))
    ax.bar([i - 0.2 for i in x], VAL_OBS, width=0.4, label="observed (UKHLS)", color=NAVY)
    ax.bar([i + 0.2 for i in x], VAL_SIM, width=0.4, label="simulated", color=TEAL)
    ax.set_xticks(list(x))
    ax.set_xticklabels(VAL_MOMENTS)
    ax.set_title("Out-of-sample validation (wave-1 cohort -> wave 5)")
    ax.set_ylabel("value (median income ÷ £1,000)")
    ax.legend(fontsize=8)
    _save(fig, "report_validation.png")


def main():
    """Generate all report figures."""
    fig_wevm_bcr()
    fig_spatial()
    fig_validation()


if __name__ == "__main__":
    main()
