"""Spatial layer (ODD §2.2): distance/catchment affects service access.

The region-index model treats everyone in a region identically. This layer gives each agent
a LOCATION (scattered around its region centroid) and makes healthcare take-up decay with
distance to the region's provider, producing WITHIN-region spatial inequality in coverage.

Geometry uses real (approximate) Government Office Region centroids, projected to the British
National Grid (EPSG:27700) via geopandas so distances are in metres. ``mesa_geo`` is used to
hold providers as GeoAgents in a GeoSpace, the seam for polygon catchments / agent movement.

Imports of geopandas/mesa_geo are LAZY (function-local) so the core model stays importable
without the optional ``spatial`` extra. All location randomness flows through the per-agent
seeded stream, so locations are fixed and identical across matched runs (CRN).

Geometry caveat: centroids are approximate public values and agents are scattered (not placed
on real boundaries). `TODO:` load ONS GOR/LAD boundary polygons for true within-area placement.
"""

from __future__ import annotations

import math
from functools import lru_cache

# Approximate Government Office Region centroids (lon, lat, WGS84). Public geography, not data.
REGION_CENTROIDS_LONLAT: dict[str, tuple[float, float]] = {
    "North East": (-1.7, 54.9),
    "North West": (-2.6, 53.8),
    "Yorkshire and the Humber": (-1.3, 53.9),
    "East Midlands": (-0.8, 52.8),
    "West Midlands": (-2.0, 52.5),
    "East of England": (0.4, 52.2),
    "London": (-0.12, 51.5),
    "South East": (-0.8, 51.2),
    "South West": (-3.5, 50.8),
    "Wales": (-3.7, 52.3),
    "Scotland": (-4.2, 56.5),
    "Northern Ireland": (-6.5, 54.6),
}


@lru_cache(maxsize=1)
def projected_centroids() -> dict[str, tuple[float, float]]:
    """Region centroids projected to EPSG:27700 (metres). Cached; needs the ``spatial`` extra."""
    import geopandas as gpd
    from shapely.geometry import Point

    names = list(REGION_CENTROIDS_LONLAT)
    pts = gpd.GeoSeries(
        [Point(*REGION_CENTROIDS_LONLAT[n]) for n in names], crs="EPSG:4326"
    ).to_crs(27700)
    return {n: (float(p.x), float(p.y)) for n, p in zip(names, pts, strict=True)}


def assign_location(region: str, prng, *, scatter_km: float) -> tuple[float, float, float]:
    """Draw an agent location scattered around its region centroid (the provider site).

    Returns ``(x, y, distance_km)`` in EPSG:27700 metres; ``distance_km`` is the agent's
    distance to the region provider (at the centroid). Uses the agent's seeded ``prng`` so the
    location is deterministic and identical across matched runs.
    """
    cx, cy = projected_centroids()[region]
    dx = float(prng.normal(0.0, scatter_km * 1000.0))
    dy = float(prng.normal(0.0, scatter_km * 1000.0))
    dist_km = math.hypot(dx, dy) / 1000.0
    return cx + dx, cy + dy, dist_km


def distance_decay(distance_km: float, *, scale_km: float) -> float:
    """Distance-decay access factor in (0, 1]: ``exp(-distance/scale)`` (ODD §7.2 distance term)."""
    if scale_km <= 0:
        return 1.0
    return math.exp(-distance_km / scale_km)


def build_provider_geospace(model, providers):
    """Create a ``mesa_geo`` GeoSpace holding one provider GeoAgent per region (the Mesa-Geo seam).

    Providers are placed at region centroids. The GeoSpace is geometry-ready for polygon
    catchments / spatial queries; the access mechanism uses precomputed distances. Returns
    the populated GeoSpace (or None if the spatial extra is unavailable).
    """
    try:
        import mesa_geo as mg
        from shapely.geometry import Point
    except ImportError:  # pragma: no cover - spatial extra not installed
        return None

    cents = projected_centroids()
    geospace = mg.GeoSpace(crs="EPSG:27700", warn_crs_conversion=False)
    geo_providers = []
    for prov in providers:
        if prov.region not in cents:
            continue
        x, y = cents[prov.region]
        ga = ProviderSite(model, Point(x, y), "EPSG:27700", region=prov.region)
        geo_providers.append(ga)
    if geo_providers:
        geospace.add_agents(geo_providers)
    return geospace


def ProviderSite(model, geometry, crs, *, region: str):  # noqa: N802 - factory, not a class
    """Create a provider GeoAgent at a fixed location (factory keeps mesa_geo import lazy)."""
    import mesa_geo as mg

    agent = mg.GeoAgent(model, geometry, crs)
    agent.region = region
    agent.service_type = "healthcare"
    return agent
