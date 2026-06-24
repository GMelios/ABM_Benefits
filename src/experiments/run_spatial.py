"""Demonstration: the Mesa-Geo spatial layer, distance to provider drives access inequality.

Run: ``uv run python -m src.experiments.run_spatial``  (needs `uv sync --extra spatial`)

With ``spatial`` on, agents are scattered around their real region centroids and healthcare
take-up decays with distance to the region provider. This produces WITHIN-region inequality
in coverage that the region-index model cannot represent. Reported as coverage by distance
band, averaged over seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.estimation.load_ukhls import load_panel
from src.model.config import PolicyConfig
from src.model.model import SocialServicesModel
from src.model.params import ModelParams
from src.population.synthesize import build_population

BANDS = [0, 20, 40, 60, 1000]
LABELS = ["<20km", "20-40km", "40-60km", "60km+"]


def main(*, n_agents: int = 2500, seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    """Run the spatial demonstration and print coverage by distance band."""
    panel = load_panel()
    params = ModelParams.load()
    cfg = PolicyConfig(
        n_agents=n_agents,
        horizon=8,
        spatial=True,
        hc_awareness=0.9,
        hc_capacity_per_1000=400.0,
        distance_decay_km=50.0,
    )
    rows = []
    for s in seeds:
        pop = build_population(panel, n_agents=n_agents, seed=s)
        m = SocialServicesModel(pop, params, cfg, seed=s)
        df = m.run()
        dist = {a.aid: a.distance_km for a in m.individuals}
        t0 = df[df["tick"] == 0].copy()
        t0["dist_km"] = t0["agent_id"].map(dist)
        needy = t0[t0["need_healthcare"] > 0].copy()
        needy["band"] = pd.cut(needy["dist_km"], BANDS, labels=LABELS)
        rows.append(needy.groupby("band", observed=True)["access_healthcare"].mean())

    cov = pd.concat(rows, axis=1).mean(axis=1)
    print("=" * 64 + "\nSPATIAL LAYER, healthcare coverage by distance to provider\n" + "=" * 64)
    print(
        f"seeds={list(seeds)}, n_agents={n_agents}, distance decay scale={cfg.distance_decay_km}km"
    )
    print("\n  distance band   mean coverage (needy agents)")
    for band in LABELS:
        if band in cov.index:
            print(f"  {band:<14s}  {cov[band]:.3f}")
    gap = cov.get(LABELS[0], np.nan) - cov.get(LABELS[-1], np.nan)
    print(f"\nnear-vs-far coverage gap: {gap:.3f}  (within-region spatial inequality)")
    print("Geometry uses approximate GOR centroids (EPSG:27700); `TODO:` real ONS boundaries.")


if __name__ == "__main__":
    main()
