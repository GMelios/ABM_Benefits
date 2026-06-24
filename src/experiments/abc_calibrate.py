"""ABC posterior calibration (pyABC) + propagation of parameter uncertainty into the WEVM.

This deepens the point calibration (`calibrate.py`): instead of a single best `is_awareness`,
we fit an **approximate Bayesian posterior** to the observed UKHLS receipt rate and then
propagate posterior draws through the welfare layer so the reported WEVM(ε) bands include
**parameter uncertainty**, not just seed noise (ODD §8.4).

What UKHLS can identify:
- `is_awareness` (income-support take-up multiplier) is identified by the working-age
  benefit-receipt LEVEL, we calibrate it.
- `hc_awareness` / `hc_access_health_effect` are NOT separately identifiable from UKHLS
  (no service-access measure); they are NOT calibrated here, they remain priors/levers,
  and the healthcare magnitude is reported with sensitivity over their range.

Reproducibility note: pyABC draws particles via NumPy's global RNG, so the calibration
harness seeds it once (the ABM evaluations themselves use the model's own seeded Generator
at fixed seeds, so each summary statistic is deterministic given the parameter).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.estimation.load_ukhls import load_panel
from src.experiments.calibrate import simulated_receipt_rate
from src.model.config import CALIBRATION_PARAMS, PolicyConfig
from src.model.params import ModelParams


@dataclass
class Posterior:
    """ABC posterior over the calibrated parameter(s)."""

    param: str
    samples: np.ndarray  # weighted posterior particles
    weights: np.ndarray  # normalised importance weights
    target_rate: float
    posterior_mean: float
    posterior_sd: float
    cred_interval: tuple[float, float]  # 95% credible interval
    epsilon: float  # final ABC acceptance threshold
    n_populations: int

    def resample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` i.i.d. samples from the (weighted) posterior."""
        idx = rng.choice(len(self.samples), size=n, replace=True, p=self.weights)
        return self.samples[idx]


def abc_calibrate_takeup(
    panel: pd.DataFrame,
    params: ModelParams,
    base_cfg: PolicyConfig,
    *,
    target_rate: float,
    population_size: int = 60,
    max_populations: int = 6,
    min_epsilon: float = 0.002,
    eval_seeds: tuple[int, ...] = (0, 1),
    rng_seed: int = 0,
) -> Posterior:
    """Fit an ABC-SMC posterior for ``is_awareness`` to the receipt-rate target via pyABC.

    The summary statistic is the working-age income-support receipt rate, averaged over a
    few fixed evaluation seeds (so it is a deterministic function of the parameter).
    """
    import pyabc

    # pyABC samples particles via NumPy's global RNG; seed it once at the harness level (the
    # ABM evals use the model's own seeded Generator, so summary stats stay deterministic).
    np.random.seed(rng_seed)  # noqa: NPY002 - required by pyABC's global-RNG particle sampling
    lo, hi = CALIBRATION_PARAMS["is_awareness"]

    def model(parameter: dict) -> dict:
        cfg = base_cfg.with_(is_awareness=float(parameter["is_awareness"]), income_support_on=True)
        rate = float(
            np.mean([simulated_receipt_rate(panel, params, cfg, seed=s) for s in eval_seeds])
        )
        return {"receipt_rate": rate}

    prior = pyabc.Distribution(is_awareness=pyabc.RV("uniform", lo, hi - lo))
    abc = pyabc.ABCSMC(
        model,
        prior,
        pyabc.PNormDistance(p=2),
        population_size=population_size,
        sampler=pyabc.sampler.SingleCoreSampler(),  # in-process: reproducible, no pickling
    )
    db = Path(tempfile.gettempdir()) / f"abc_takeup_{rng_seed}.db"
    db.unlink(missing_ok=True)
    abc.new(f"sqlite:///{db}", {"receipt_rate": target_rate})
    history = abc.run(max_nr_populations=max_populations, minimum_epsilon=min_epsilon)

    df, w = history.get_distribution()
    s = df["is_awareness"].to_numpy()
    mean = float(np.average(s, weights=w))
    var = float(np.average((s - mean) ** 2, weights=w))
    lo_ci, hi_ci = _weighted_quantiles(s, w, [0.025, 0.975])
    eps = float(history.get_all_populations()["epsilon"].iloc[-1])
    return Posterior(
        param="is_awareness",
        samples=s,
        weights=np.asarray(w, dtype=float),
        target_rate=target_rate,
        posterior_mean=mean,
        posterior_sd=float(np.sqrt(var)),
        cred_interval=(float(lo_ci), float(hi_ci)),
        epsilon=eps,
        n_populations=int(history.max_t + 1),
    )


def posterior_wevm(
    panel: pd.DataFrame,
    params: ModelParams,
    base_cfg: PolicyConfig,
    posterior: Posterior,
    *,
    service: str = "income_support",
    n_draws: int = 15,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Propagate the ABC posterior into WEVM(eps) bands (parameter + seed uncertainty).

    Draws ``n_draws`` posterior values of the calibrated parameter; for each, runs the
    matched-pair welfare evaluation over ``seeds``; pools all (draw x seed) WEVM values per
    eps and reports the mean and 95% interval, so the bands reflect calibration uncertainty,
    not just Monte-Carlo seed noise (ODD §8.4).
    """
    from src.welfare import welfare as wf
    from src.welfare.runner import evaluate_service

    rng = np.random.default_rng(rng_seed)
    draws = posterior.resample(n_draws, rng)
    pooled: list[dict] = []
    for draw in draws:
        cfg = base_cfg.with_(**{posterior.param: float(draw)})
        res = evaluate_service(panel, params, cfg, service=service, seeds=list(seeds))
        for row in res.wevm_by_seed.to_dict("records"):
            pooled.append({"param_draw": float(draw), **row})

    pool = pd.DataFrame(pooled)
    rows = []
    for eps in wf.EPS_GRID:
        vals = pool[eps].to_numpy()
        rows.append(
            {
                "eps": eps,
                "wevm_mean": float(vals.mean()),
                "ci_lo": float(np.percentile(vals, 2.5)),
                "ci_hi": float(np.percentile(vals, 97.5)),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["service"] = service
    out.attrs["n_draws"] = n_draws
    out.attrs["n_seeds"] = len(seeds)
    out.attrs["uncertainty"] = "parameter (ABC posterior) + seed"
    return out


def _weighted_quantiles(values: np.ndarray, weights, qs: list[float]) -> list[float]:
    """Weighted quantiles of a particle set."""
    order = np.argsort(values)
    v, wq = values[order], np.asarray(weights)[order]
    cum = np.cumsum(wq) - 0.5 * wq
    cum /= np.sum(wq)
    return [float(np.interp(q, cum, v)) for q in qs]


if __name__ == "__main__":  # pragma: no cover - manual run
    panel = load_panel()
    params = ModelParams.load()
    cfg = PolicyConfig(n_agents=1500, horizon=10)
    post = abc_calibrate_takeup(
        panel, params, cfg, target_rate=0.358, population_size=40, max_populations=5
    )
    print(
        f"is_awareness posterior: mean={post.posterior_mean:.3f} "
        f"sd={post.posterior_sd:.3f} 95%CI=[{post.cred_interval[0]:.3f}, "
        f"{post.cred_interval[1]:.3f}] (eps={post.epsilon:.4f}, {post.n_populations} pops)"
    )
