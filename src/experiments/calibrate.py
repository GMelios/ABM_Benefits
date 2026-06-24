"""Light calibration of the free take-up parameter to the observed UKHLS receipt rate.

Per the build spec this is a LIGHT calibration (a few free params to the most important
UKHLS target), not a full ABC campaign. We tune ``is_awareness`` so the simulated
income-support receipt rate among the working-age matches the UKHLS working-age benefit
receipt rate. Uses Optuna (TPE) on a single factual run per trial (cheap).

The deeper uncertainty quantification (pyABC posterior over all free params) is left to the
"to deepen" list (ODD §8).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.model.config import CALIBRATION_PARAMS, PolicyConfig
from src.model.model import SocialServicesModel
from src.model.params import ModelParams
from src.population.synthesize import build_population


def simulated_receipt_rate(
    panel: pd.DataFrame, params: ModelParams, cfg: PolicyConfig, *, seed: int
) -> float:
    """Mean per-tick income-support receipt rate among working-age (16-64) agents."""
    pop = build_population(panel, n_agents=cfg.n_agents, seed=seed, base_wave=cfg.base_wave)
    df = SocialServicesModel(pop, params, cfg, seed=seed).run()
    wa = df[(df["age"] >= 16) & (df["age"] <= 64)]
    return float(wa["access_income_support"].mean())


@dataclass
class CalibrationResult:
    """Outcome of the light take-up calibration."""

    param: str
    value: float
    target_rate: float
    achieved_rate: float
    n_trials: int


def calibrate_takeup(
    panel: pd.DataFrame,
    params: ModelParams,
    base_cfg: PolicyConfig,
    *,
    target_rate: float,
    n_trials: int = 30,
    seed: int = 0,
) -> CalibrationResult:
    """Tune ``is_awareness`` so the simulated working-age receipt rate ~= ``target_rate``.

    Returns the best value and the achieved rate. Optuna minimises squared error on a single
    seeded run per trial (fast); the chosen value is then used for multi-seed reporting.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lo, hi = CALIBRATION_PARAMS["is_awareness"]

    def objective(trial: optuna.Trial) -> float:
        aware = trial.suggest_float("is_awareness", lo, hi)
        cfg = base_cfg.with_(is_awareness=aware, income_support_on=True)
        rate = simulated_receipt_rate(panel, params, cfg, seed=seed)
        return (rate - target_rate) ** 2

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params["is_awareness"]
    achieved = simulated_receipt_rate(panel, params, base_cfg.with_(is_awareness=best), seed=seed)
    return CalibrationResult(
        param="is_awareness",
        value=best,
        target_rate=target_rate,
        achieved_rate=achieved,
        n_trials=n_trials,
    )
