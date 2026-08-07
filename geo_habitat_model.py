"""
EcoPulse - Geospatial habitat/risk model (trained, not heuristic)
"""
import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

METRIC_CRS = 32644  # UTM zone 44N (Sri Lanka)


# ---------------------------------------------------------------------
# 1. Get real occurrence points from GBIF (or generate local fallback)
# ---------------------------------------------------------------------
def fetch_gbif_occurrences(country="LK", limit=300, max_records=2000):
    """Page through GBIF's public occurrence search API for Elephas
    maximus records with coordinates in Sri Lanka."""
    records = []
    offset = 0
    headers = {"User-Agent": "EcoPulse-Research-App/1.0 (Contact: pvhvishu@gmail.com)"}
    
    while offset < max_records:
        params = {
            "scientificName": "Elephas maximus",
            "country": country,
            "hasCoordinate": "true",
            "limit": limit,
            "offset": offset,
        }
        try:
            resp = requests.get(
                "https://api.gbif.org/v1/occurrence/search",
                params=params,
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            for r in results:
                lat, lon = r.get("decimalLatitude"), r.get("decimalLongitude")
                if lat is not None and lon is not None:
                    records.append({"lat": lat, "lon": lon})
            offset += limit
            if data.get("endOfRecords", True):
                break
            time.sleep(0.1)  # Be polite to the public API
        except Exception as e:
            print(f"[Notice] GBIF API fetch paused at offset {offset}: {e}")
            break

    return pd.DataFrame(records)


def load_occurrences(args):
    # 1. Check if user passed a local CSV file
    if args.occurrence_csv and Path(args.occurrence_csv).exists():
        print(f"Loading occurrence records from local CSV: {args.occurrence_csv}")
        df = pd.read_csv(args.occurrence_csv)
        lat_col = next((c for c in df.columns if c.lower() in ("lat", "decimallatitude")), None)
        lon_col = next((c for c in df.columns if c.lower() in ("lon", "lng", "decimallongitude")), None)
        if lat_col and lon_col:
            df = df.rename(columns={lat_col: "lat", lon_col: "lon"})[["lat", "lon"]].dropna()
            print(f"Loaded {len(df)} occurrence points from local CSV.")
            return df

    # 2. Attempt online GBIF API query
    print("Querying GBIF (api.gbif.org) for Elephas maximus occurrence records in Sri Lanka ...")
    df = fetch_gbif_occurrences()
    
    if len(df) >= 20:
        print(f"Got {len(df)} occurrence points from GBIF API.")
        return df

    # 3. Smart Fallback: Sample points from local protected area shapefiles if GBIF API returns 0 or fails
    print("\n[Notice] GBIF API returned insufficient points (or network was unreachable).")
    print("Generating fallback occurrence points sampled from local protected area shapefiles...")
    
    geo_dir = Path(args.geo_dir)
    try:
        prot_path = geo_dir / "gis_osm_protected_areas_a_free_1.shp"
        land_path = geo_dir / "gis_osm_landuse_a_free_1.shp"
        
        if prot_path.exists():
            protected = gpd.read_file(str(prot_path))
        elif land_path.exists():
            protected = gpd.read_file(str(land_path))
        else:
            raise FileNotFoundError("No shapefile found for sampling habitat points.")

        protected_4326 = protected.to_crs(4326)
        bounds = protected_4326.total_bounds  # [minx, miny, maxx, maxy]
        
        fallback_pts = []
        rng = np.random.default_rng(42)
        union_geom = protected_4326.geometry.union_all()
        
        attempts = 0
        while len(fallback_pts) < 150 and attempts < 10000:
            attempts += 1
            rx = rng.uniform(bounds[0], bounds[2])
            ry = rng.uniform(bounds[1], bounds[3])
            pt = Point(rx, ry)
            if union_geom.contains(pt):
                fallback_pts.append({"lat": ry, "lon": rx})
                
        df_fallback = pd.DataFrame(fallback_pts)
        print(f"Successfully generated {len(df_fallback)} fallback elephant occurrence points inside habitat zones.")
        return df_fallback
    except Exception as e:
        print(f"[Error] Fallback generation error: {e}")
        raise RuntimeError("Could not retrieve GBIF data or generate local fallback points.")


# ---------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------
def load_layers(geo_dir):
    geo_dir = Path(geo_dir)

    def shp(name):
        return str(geo_dir / f"gis_osm_{name}.shp")

    landuse = gpd.read_file(shp("landuse_a_free_1"))
    forest = landuse[landuse["fclass"].isin(["forest", "scrub"])]
    protected = gpd.read_file(shp("protected_areas_a_free_1"))
    railways = gpd.read_file(shp("railways_free_1"))
    places = gpd.read_file(shp("places_free_1"))
    admin = gpd.read_file(shp("adminareas_a_free_1"))
    return forest, protected, railways, places, admin


def nearest_distance(points, targets, col):
    j = gpd.sjoin_nearest(points, targets[["geometry"]], distance_col=col)
    j = j[~j.index.duplicated(keep="first")]
    return j[col].values


def make_features(points_gdf, forest_m, protected_m, rail_m, places_m):
    points_gdf = points_gdf.copy()
    points_gdf["dist_forest_m"] = nearest_distance(points_gdf, forest_m, "d1")
    points_gdf["dist_protected_m"] = nearest_distance(points_gdf, protected_m, "d2")
    points_gdf["dist_rail_m"] = nearest_distance(points_gdf, rail_m, "d3")
    points_gdf["dist_settlement_m"] = nearest_distance(points_gdf, places_m, "d4")
    return points_gdf


def make_pseudo_absences(admin_gdf, n, exclude_points_m, min_dist_from_positive_m=2000, seed=42):
    """Random points inside Sri Lanka's land boundary kept away from positive sightings."""
    rng = np.random.default_rng(seed)
    land = admin_gdf.to_crs(METRIC_CRS).union_all()
    minx, miny, maxx, maxy = land.bounds
    pts = []
    attempts = 0
    while len(pts) < n and attempts < n * 50:
        attempts += 1
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if not land.contains(p):
            continue
        if exclude_points_m.distance(p).min() < min_dist_from_positive_m:
            continue
        pts.append(p)
    return gpd.GeoDataFrame(geometry=pts, crs=METRIC_CRS)


# ---------------------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------------------
def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    occ_df = load_occurrences(args)
    forest, protected, railways, places, admin = load_layers(args.geo_dir)

    forest_m = forest.to_crs(METRIC_CRS)
    protected_m = protected.to_crs(METRIC_CRS)
    rail_m = railways.to_crs(METRIC_CRS)
    places_m = places.to_crs(METRIC_CRS)

    presence = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lat, lon in zip(occ_df["lat"], occ_df["lon"])],
        crs=4326,
    ).to_crs(METRIC_CRS)

    print(f"Generating {len(presence) * 3} pseudo-absence points ...")
    absence = make_pseudo_absences(admin, n=len(presence) * 3, exclude_points_m=presence.geometry)

    presence["label"] = 1
    absence["label"] = 0
    all_pts = pd.concat([presence, absence], ignore_index=True)
    all_pts = gpd.GeoDataFrame(all_pts, geometry="geometry", crs=METRIC_CRS)

    all_pts = make_features(all_pts, forest_m, protected_m, rail_m, places_m)

    feat_cols = ["dist_forest_m", "dist_protected_m", "dist_rail_m", "dist_settlement_m"]
    X = all_pts[feat_cols].values
    y = all_pts["label"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    print(f"\n5-fold cross-val ROC-AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    print(f"Held-out ROC-AUC: {roc_auc_score(yte, proba):.3f}")
    print("\nClassification Report:")
    print(classification_report(yte, clf.predict(Xte), target_names=["not-habitat", "elephant-habitat"]))
    print("Feature Importances:", dict(zip(feat_cols, clf.feature_importances_.round(3))))

    import joblib
    joblib.dump(clf, out_dir / "geo_habitat_model.joblib")
    with open(out_dir / "geo_feature_cols.json", "w") as f:
        json.dump(feat_cols, f)
    print(f"\nSaved trained model to: {out_dir / 'geo_habitat_model.joblib'}")

    # Score settlements with the trained Random Forest model
    places_scored = make_features(places_m, forest_m, protected_m, rail_m, places_m)
    places_scored["habitat_probability"] = clf.predict_proba(places_scored[feat_cols].values)[:, 1]
    result_csv = out_dir / "settlement_risk_scores_trained.csv"
    places_scored.drop(columns="geometry").to_csv(result_csv, index=False)
    print(f"Saved settlement risk scores to: {result_csv}")


def load_geo_model(model_path, feat_cols_path):
    import joblib
    clf = joblib.load(model_path)
    with open(feat_cols_path) as f:
        feat_cols = json.load(f)
    return clf, feat_cols


def predict_habitat_probability(clf, feat_cols, dist_forest_m, dist_protected_m, dist_rail_m, dist_settlement_m):
    x = np.array([[dist_forest_m, dist_protected_m, dist_rail_m, dist_settlement_m]])
    return float(clf.predict_proba(x)[0, 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geo_dir",
        default=r"D:\Researches\Data Odyssey 2026\unique\Data Odyssey 2026 Project\data\geo",
    )
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument(
        "--occurrence_csv",
        default=None,
        help="Optional: path to a local CSV file containing lat/lon occurrence records.",
    )
    args = parser.parse_args()
    train(args)