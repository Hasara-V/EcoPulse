"""
EcoPulse - Data Fusion & Alert simulation
--------------------------------------------
Combines the three modality outputs into one decision, exactly as
described in the proposal's "Data Fusion Layer": a heuristic weighting
across acoustic, visual and geospatial confidence scores, which triggers
a simulated SMS/railway alert when the combined score crosses a
threshold.

Each of the three inputs now comes from a genuinely trained model:
  - Audio : audio_classifier.py trained with --background_dir (real
            elephant-vs-not classifier, not just elephant-vs-elephant)
  - Vision: image_detector.py using pretrained OR (better) the
            fine-tuned weights from image_finetune.py
  - Geo   : geo_habitat_model.py's trained RandomForest habitat model
            if present, otherwise falls back to the heuristic
            geo_risk_analysis.py score

Run this AFTER you have trained what you can:
  1. python audio_classifier.py --background_dir ...   -> outputs/audio_model_best.pt
  2. (optional) python image_finetune.py ...            -> outputs/image_model_best.pt
  3. python geo_risk_analysis.py ...                     -> outputs/settlement_risk_scores.csv
     and/or python geo_habitat_model.py ...              -> outputs/geo_habitat_model.joblib

USAGE:
    python fusion_alert.py --audio_clip "path/to/some_clip.wav" \
                            --image "path/to/some_photo.jpg" \
                            --settlement "Anuradhapura"
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from audio_classifier import elephant_confidence, load_inference_model, predict_clip
from geo_risk_analysis import get_risk_for_location
from image_detector import predict_single, resolve_weights

# Fusion weights -- tune these based on which sensor you trust most in
# practice (e.g. increase acoustic weight at night when cameras are less
# reliable).
WEIGHTS = {"audio": 0.4, "image": 0.35, "geo": 0.25}
ALERT_THRESHOLD = 0.55


def fuse(audio_conf, image_conf, geo_risk):
    score = (
        WEIGHTS["audio"] * audio_conf
        + WEIGHTS["image"] * image_conf
        + WEIGHTS["geo"] * geo_risk
    )
    return score


def get_geo_score(out_dir, settlement, geo_dir=None):
    """Prefer the trained habitat model (geo_habitat_model.py) if it
    exists; otherwise fall back to the heuristic risk score from
    geo_risk_analysis.py."""
    model_path = out_dir / "geo_habitat_model.joblib"
    feat_path = out_dir / "geo_feature_cols.json"
    if model_path.exists() and feat_path.exists() and geo_dir:
        try:
            from geo_habitat_model import load_geo_model, load_layers as load_geo_layers, \
                make_features as make_geo_features

            clf, feat_cols = load_geo_model(model_path, feat_path)
            forest, protected, railways, places, admin = load_geo_layers(geo_dir)
            row = places[places["name"].str.lower() == settlement.lower()]
            if row.empty:
                print(f"[Geo]    settlement '{settlement}' not found in places layer; using heuristic score instead.")
            else:
                METRIC_CRS = 32644
                pt = row.to_crs(METRIC_CRS)
                forest_m, protected_m, rail_m, places_m = (
                    forest.to_crs(METRIC_CRS), protected.to_crs(METRIC_CRS),
                    railways.to_crs(METRIC_CRS), places.to_crs(METRIC_CRS),
                )
                feats = make_geo_features(pt, forest_m, protected_m, rail_m, places_m)
                proba = clf.predict_proba(feats[feat_cols].values)[0, 1]
                print(f"[Geo]    settlement '{settlement}' trained habitat-model probability: {proba:.2f}")
                return float(proba)
        except Exception as e:
            print(f"[Geo]    trained habitat model unavailable ({e}); falling back to heuristic score.")

    risk_csv = out_dir / "settlement_risk_scores.csv"
    if risk_csv.exists():
        val = get_risk_for_location(risk_csv, settlement)
        if val is not None:
            print(f"[Geo]    settlement '{settlement}' heuristic risk score: {val:.2f}")
            return float(val)
    print(f"[Geo]    no geo score available for '{settlement}' -- run geo_risk_analysis.py first.")
    return 0.0


def run(args):
    out_dir = Path(args.out_dir)

    audio_conf = 0.0
    if args.audio_clip:
        model, classes = load_inference_model(
            out_dir / "audio_model_best.pt", out_dir / "audio_classes.json"
        )
        label, top_conf, probs = predict_clip(model, classes, args.audio_clip)
        audio_conf = elephant_confidence(classes, probs)
        print(f"[Audio]  top match '{label}' ({top_conf:.2f}) | P(any elephant call): {audio_conf:.2f}")

    image_conf = 0.0
    if args.image:
        weights = resolve_weights(out_dir, args.weights)
        yolo_model = YOLO(weights)
        present, image_conf = predict_single(yolo_model, args.image)
        print(f"[Vision] elephant present: {present}, confidence {image_conf:.2f}")

    geo_risk = 0.0
    if args.settlement:
        geo_risk = get_geo_score(out_dir, args.settlement, geo_dir=args.geo_dir)

    fused_score = fuse(audio_conf, image_conf, geo_risk)
    print(f"\nFused confidence score: {fused_score:.2f}  (threshold: {ALERT_THRESHOLD})")

    if fused_score >= ALERT_THRESHOLD:
        print("\n*** ALERT TRIGGERED ***")
        print(f"SMS -> Farming communities near {args.settlement or 'the sensor node'}: "
              f"'Elephant activity likely detected nearby. Please take precaution.'")
        print(f"Signal -> Railway control: 'Reduce speed through sensor zone "
              f"near {args.settlement or 'this location'}.'")
    else:
        print("\nNo alert triggered - confidence below threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_clip", default=None,
                         help="Path to an audio clip to test (.wav/.flac/.ogg/.mp3/.m4a/.aac/.wma)")
    parser.add_argument("--image", default=None,
                         help="Path to an image to test (.jpg/.png/.bmp/.tiff/.webp)")
    parser.add_argument("--settlement", default=None, help="Nearest settlement name, e.g. 'Anuradhapura'")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--geo_dir", default=r"D:\Researches\Data Odyssey 2026\unique\Data Odyssey 2026 Project\data\geo",
                         help="Needed only if using the trained geo habitat model")
    args = parser.parse_args()
    run(args)
