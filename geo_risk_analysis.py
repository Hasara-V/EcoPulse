"""
EcoPulse - Geospatial risk-zone module

"""

import argparse
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

METRIC_CRS = 32644  # UTM zone 44N, appropriate for Sri Lanka


def prox_score(distance_m, scale_m):
    """1.0 when distance is 0, fades linearly to 0.0 at scale_m and beyond."""
    return np.clip(1 - distance_m / scale_m, 0, 1)


def compute_risk(geo_dir, out_dir):
    geo_dir = Path(geo_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def shp(name):
        return str(geo_dir / f"gis_osm_{name}.shp")

    print("Loading layers ...")
    places = gpd.read_file(shp("places_free_1"))
    places = places[places["fclass"].isin(
        ["village", "hamlet", "town", "suburb", "city", "farm"]
    )].copy()

    landuse = gpd.read_file(shp("landuse_a_free_1"))
    forest = landuse[landuse["fclass"].isin(["forest", "scrub"])]

    protected = gpd.read_file(shp("protected_areas_a_free_1"))
    railways = gpd.read_file(shp("railways_free_1"))

    print(f"{len(places)} settlements | {len(forest)} forest/scrub polygons | "
          f"{len(protected)} protected areas | {len(railways)} railway segments")

    # Reproject everything to a metric CRS so distances are in meters
    places_m = places.to_crs(METRIC_CRS)
    forest_m = forest.to_crs(METRIC_CRS)
    protected_m = protected.to_crs(METRIC_CRS)
    railways_m = railways.to_crs(METRIC_CRS)

    def nearest_distance(points, targets, col):
        j = gpd.sjoin_nearest(points, targets[["geometry"]], distance_col=col)
        j = j[~j.index.duplicated(keep="first")]
        return j[col]

    places_m["dist_forest_m"] = nearest_distance(places_m, forest_m, "d1")
    places_m["dist_protected_m"] = nearest_distance(places_m, protected_m, "d2")
    places_m["dist_railway_m"] = nearest_distance(places_m, railways_m, "d3")

    # Weighted HEC risk score: proximity to forest matters most (elephants
    # emerge from forest edges), protected areas next (known elephant
    # ranges), railway proximity last (train-strike risk).
    places_m["risk_score"] = (
        0.5 * prox_score(places_m["dist_forest_m"], 3000)
        + 0.3 * prox_score(places_m["dist_protected_m"], 5000)
        + 0.2 * prox_score(places_m["dist_railway_m"], 5000)
    )

    result = places_m.sort_values("risk_score", ascending=False)

    csv_path = out_dir / "settlement_risk_scores.csv"
    result.drop(columns="geometry").to_csv(csv_path, index=False)
    print(f"Saved risk table to {csv_path}")

    geojson_path = out_dir / "settlement_risk_scores.geojson"
    result.to_crs(4326).to_file(geojson_path, driver="GeoJSON")
    print(f"Saved GeoJSON (for QGIS / web maps) to {geojson_path}")

    print("\nTop 10 highest-risk settlements:")
    print(result[["name", "fclass", "dist_forest_m", "dist_railway_m", "risk_score"]].head(10))

    # Static map
    fig, ax = plt.subplots(figsize=(9, 12))
    forest.plot(ax=ax, color="#2e7d32", alpha=0.5, label="Forest/scrub")
    protected.to_crs(4326).plot(ax=ax, color="#1565c0", alpha=0.3, label="Protected area")
    railways.to_crs(4326).plot(ax=ax, color="black", linewidth=0.5, label="Railway")
    result.to_crs(4326).plot(
        ax=ax, column="risk_score", cmap="Reds", markersize=15,
        legend=True, legend_kwds={"label": "HEC risk score"},
    )
    ax.set_title("Human-Elephant Conflict Risk by Settlement (Sri Lanka)")
    ax.set_axis_off()
    fig.tight_layout()
    map_path = out_dir / "hec_risk_map.png"
    fig.savefig(map_path, dpi=150)
    print(f"Saved risk map image to {map_path}")


def get_risk_for_location(risk_csv_path, place_name):
    """Helper used by fusion_alert.py to look up a settlement's risk score."""
    import pandas as pd
    df = pd.read_csv(risk_csv_path)
    row = df[df["name"].str.lower() == place_name.lower()]
    if row.empty:
        return None
    return float(row.iloc[0]["risk_score"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geo_dir",
        default=r"D:\Researches\Data Odyssey 2026\unique\Data Odyssey 2026 Project\data\geo",
    )
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()
    compute_risk(args.geo_dir, args.out_dir)

# ---------------------------------------------------------------------
# OPTIONAL EXTENSION: grid-based heatmap using the buildings layer
# ---------------------------------------------------------------------
# If you want a finer-grained heatmap (and have time/RAM to spare -- the
# buildings shapefile is large):
#
#   1. Build a regular grid of e.g. 5km x 5km cells over Sri Lanka's
#      bounding box using shapely.geometry.box in a loop.
#   2. For each cell, count building centroids inside it
#      (gpd.sjoin(buildings, grid, predicate="within")).
#   3. Combine building density with dist_forest_m / dist_railway_m
#      computed per grid cell centroid, the same way as above.
#   4. Plot with ax.pcolormesh or geopandas' .plot(column=...) on the grid
#      GeoDataFrame instead of points.
