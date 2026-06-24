"""SALib Morris screen: which free parameters move the WEVM? (ODD §8.1, calibrate-the-rest).

Identifies the influential parameters (by mean absolute elementary effect, mu_star) so the
rest can be fixed before calibration. Kept light (few trajectories, modest population), a
screen, not a full variance decomposition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.sample import morris as morris_sample

from src.model.config import PolicyConfig
from src.model.params import ModelParams
from src.welfare.runner import compute_welfare, matched_pair

#: Free parameters screened, with bounds (policy/calibration levers).
SCREEN_BOUNDS: dict[str, tuple[float, float]] = {
    "is_awareness": (0.5, 2.0),
    "is_min_income": (1000.0, 1600.0),
    "hc_awareness": (0.3, 1.0),
    "hc_capacity_per_1000": (40.0, 150.0),
    "hc_access_health_effect": (0.005, 0.05),
}


def morris_screen(
    panel: pd.DataFrame,
    params: ModelParams,
    base_cfg: PolicyConfig,
    *,
    service: str,
    eps: float = 1.0,
    n_trajectories: int = 6,
    seed: int = 0,
) -> pd.DataFrame:
    """Run a Morris elementary-effects screen on WEVM(eps) for ``service``.

    Returns a tidy frame ranked by ``mu_star`` (mean |elementary effect|): the larger,
    the more influential the parameter on the welfare output.
    """
    names = list(SCREEN_BOUNDS)
    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": [list(SCREEN_BOUNDS[n]) for n in names],
    }
    X = morris_sample.sample(problem, N=n_trajectories, seed=seed)
    Y = np.empty(X.shape[0])
    for i, row in enumerate(X):
        overrides = dict(zip(names, row, strict=True))
        cfg = base_cfg.with_(**overrides)
        pf, pcf = matched_pair(panel, params, cfg, service=service, seed=seed)
        res = compute_welfare(pf, pcf, params=params, cfg=cfg, service=service)
        # screen on the linear money-metric: the closed-form WEVM is numerically degenerate
        # for the health channel (large per-period non-income utility jump / small beta_y),
        # which would swamp the elementary effects with overflow artefacts.
        Y[i] = res["wevm_linear"][eps]
    Si = morris_analyze.analyze(problem, X, Y, seed=seed)
    out = (
        pd.DataFrame(
            {
                "param": names,
                "mu_star": Si["mu_star"],
                "mu": Si["mu"],
                "sigma": Si["sigma"],
            }
        )
        .sort_values("mu_star", ascending=False)
        .reset_index(drop=True)
    )
    out.attrs["service"] = service
    out.attrs["eps"] = eps
    out.attrs["n_evals"] = X.shape[0]
    return out
