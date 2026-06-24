"""Wire estimated UKHLS coefficients (params/*.json) into per-agent prediction functions.

Every behavioural parameter the ABM uses comes from here, there are NO hard-coded
behavioural constants in the agent/model code. Policy levers and the
service-effect sizes live separately in :mod:`src.model.config`.

The :class:`Predictor` reconstructs the exact estimation design (one-hot dummies with the
same drop-first base categories: region base = "East Midlands", education base = "a_level",
sex base = male=1.0) so in-sim linear predictors match the fitted models.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_DIR = _REPO_ROOT / "params"


def _expit(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class Predictor:
    """A fitted linear predictor that rebuilds the estimation design per agent.

    Parameters
    ----------
    coefficients
        Coefficient dict from a param file (keys incl. ``const`` and dummy names).
    numeric
        Numeric feature names to read from the supplied feature dict.
    categoricals
        Map of feature-name -> dummy prefix (e.g. ``{"region": "region"}``). The agent's
        value is formatted into ``f"{prefix}_{value}"``; a key absent from ``coefficients``
        is the dropped base category and contributes 0.
    """

    coefficients: dict[str, float]
    numeric: tuple[str, ...]
    categoricals: dict[str, str]

    def linpred(self, feats: dict) -> float:
        """Return the linear predictor x·beta for one agent's feature dict."""
        z = self.coefficients.get("const", 0.0)
        for n in self.numeric:
            z += self.coefficients.get(n, 0.0) * float(feats[n])
        for attr, prefix in self.categoricals.items():
            z += self.coefficients.get(f"{prefix}_{feats[attr]}", 0.0)
        return z


@dataclass
class ModelParams:
    """All estimated models, loaded and exposed as agent-level prediction functions."""

    beta_y: float
    beta_health: float
    health: Predictor
    employment: Predictor
    earnings: Predictor
    takeup: Predictor
    _raw: dict

    # -- utility (welfare-relevant additive terms; invariant demographic/region/wave terms
    #    cancel in the factual-counterfactual difference, so they are omitted) -------------
    def utility(self, *, income: float, health: float) -> float:
        """Welfare-relevant indirect utility v = beta_y·ln(y) + beta_h·h (de-overlapped spec).

        Health is the PHYSICAL (PCS-based) stock. Employment is NOT a utility argument (it was
        excluded from estimation as an income mediator; its welfare effect flows through
        income). Demographic/region/wave terms are constant within an agent across matched
        runs and cancel in Delta_i, so they are intentionally excluded (see ev-trajectory note).
        """
        y = max(income, 1.0)
        return self.beta_y * math.log(y) + self.beta_health * health

    # -- transitions ---------------------------------------------------------------------
    def health_next(self, *, health: float, age10: float, employed: bool) -> float:
        """Predicted next-period health in [0,1] (fractional logit).

        The income->health channel is intentionally OMITTED: its estimated coefficient is
        statistically indistinguishable from zero (~-0.001), but amplified by
        beta_health/beta_y (~544) it would inject a large spurious welfare artifact (e.g.
        making a cash transfer appear to *reduce* health-mediated wellbeing). The estimated
        coefficient is retained in params/health_dynamics.json for the record. The main
        indirect channel (health -> employment -> income) is unaffected.
        """
        return _expit(
            self.health.linpred(
                {
                    "health_index": health,
                    "age10": age10,
                    "age10_sq": age10**2,
                    "employed_f": float(employed),
                }
            )
        )

    def employed_prob(
        self,
        *,
        employed: bool,
        health: float,
        age10: float,
        education: str,
        region: str,
        sex: float,
    ) -> float:
        """Predicted P(employed next period) (logit)."""
        return _expit(
            self.employment.linpred(
                {
                    "employed_f": float(employed),
                    "health_index": health,
                    "age10": age10,
                    "age10_sq": age10**2,
                    "education": education,
                    "region": region,
                    "sex": sex,
                }
            )
        )

    def log_earnings(
        self, *, age10: float, health: float, education: str, region: str, sex: float
    ) -> float:
        """Predicted ln(net labour income) for an employed agent."""
        return self.earnings.linpred(
            {
                "age10": age10,
                "age10_sq": age10**2,
                "health_index": health,
                "education": education,
                "region": region,
                "sex": sex,
            }
        )

    def receipt_prob(
        self,
        *,
        log_market: float,
        no_market_income: bool,
        health: float,
        education: str,
        region: str,
        sex: float,
    ) -> float:
        """Baseline P(benefit receipt) (logit), scaled by an awareness lever in the model."""
        return _expit(
            self.takeup.linpred(
                {
                    "log_market": log_market,
                    "no_market_income": float(no_market_income),
                    "health_index": health,
                    "education": education,
                    "region": region,
                    "sex": sex,
                }
            )
        )

    # -- loader --------------------------------------------------------------------------
    @classmethod
    def load(cls, params_dir: Path = PARAMS_DIR) -> ModelParams:
        """Load all five param files and assemble the prediction functions."""

        def _coef(name: str) -> dict[str, float]:
            return json.loads((params_dir / f"{name}.json").read_text())["coefficients"]

        util = _coef("utility")
        region_cat = {"region": "region"}
        edu_region_sex = {"education": "education", "region": "region", "sex": "sex"}
        return cls(
            beta_y=util["log_income"],
            beta_health=util["health_index"],
            health=Predictor(
                _coef("health_dynamics"),
                # log_income omitted by design (see health_next docstring), its ~0 coefficient
                # would be amplified ~544x into a spurious welfare channel.
                ("health_index", "age10", "age10_sq", "employed_f"),
                {},
            ),
            employment=Predictor(
                _coef("employment"),
                ("employed_f", "health_index", "age10", "age10_sq"),
                edu_region_sex,
            ),
            earnings=Predictor(
                _coef("earnings"), ("age10", "age10_sq", "health_index"), edu_region_sex
            ),
            takeup=Predictor(
                _coef("takeup"), ("log_market", "no_market_income", "health_index"), edu_region_sex
            ),
            _raw={"region_cat": region_cat},
        )
