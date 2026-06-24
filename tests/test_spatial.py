"""Tests for the spatial layer (src/model/space.py + model wiring).

Skipped entirely if the optional `spatial` extra (geopandas/mesa-geo) is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("mesa_geo")

from src.estimation import load_ukhls as lk  # noqa: E402
from src.model import space  # noqa: E402
from src.model.config import PolicyConfig  # noqa: E402
from src.model.model import SocialServicesModel  # noqa: E402
from src.model.params import ModelParams  # noqa: E402
from src.population import synthesize as syn  # noqa: E402


# ---- pure spatial helpers ----
def test_distance_decay_monotone():
    d = [space.distance_decay(x, scale_km=50) for x in (0, 25, 50, 100)]
    assert d[0] == 1.0
    assert all(d[i] > d[i + 1] for i in range(len(d) - 1))
    assert all(0 < x <= 1 for x in d)


def test_assign_location_reproducible():
    a = space.assign_location("London", np.random.default_rng([1, 7]), scatter_km=40)
    b = space.assign_location("London", np.random.default_rng([1, 7]), scatter_km=40)
    assert a == b and a[2] >= 0  # same stream -> same location; distance non-negative


def test_projected_centroids_cover_all_regions():
    cents = space.projected_centroids()
    assert set(cents) == set(space.REGION_CENTROIDS_LONLAT)


# ---- model integration ----
@pytest.fixture(scope="module")
def setup():
    if not lk.DEFAULT_PANEL_PATH.exists():
        pytest.skip("restricted UKHLS parquet not available")
    return lk.load_panel(), ModelParams.load()


def test_geospace_built_with_providers(setup):
    panel, params = setup
    pop = syn.build_population(panel, n_agents=1000, seed=0)
    m = SocialServicesModel(
        pop, params, PolicyConfig(n_agents=1000, horizon=3, spatial=True), seed=0
    )
    assert m.geospace is not None
    assert len(list(m.geospace.agents)) == len(m.providers)


def test_coverage_falls_with_distance(setup):
    panel, params = setup
    cfg = PolicyConfig(
        n_agents=2000,
        horizon=6,
        spatial=True,
        hc_awareness=0.9,
        hc_capacity_per_1000=400,
        distance_decay_km=50,
    )
    pop = syn.build_population(panel, n_agents=2000, seed=0)
    m = SocialServicesModel(pop, params, cfg, seed=0)
    df = m.run()
    dist = {a.aid: a.distance_km for a in m.individuals}
    t0 = df[df["tick"] == 0].copy()
    t0["dist_km"] = t0["agent_id"].map(dist)
    needy = t0[t0["need_healthcare"] > 0]
    near = needy[needy["dist_km"] < 20]["access_healthcare"].mean()
    far = needy[needy["dist_km"] > 60]["access_healthcare"].mean()
    assert near > far  # distance-decay -> nearer agents access more (spatial inequality)


def test_spatial_off_factor_is_one(setup):
    panel, params = setup
    pop = syn.build_population(panel, n_agents=500, seed=0)
    m = SocialServicesModel(
        pop, params, PolicyConfig(n_agents=500, horizon=2, spatial=False), seed=0
    )
    assert all(a.distance_access_factor == 1.0 for a in m.individuals)
