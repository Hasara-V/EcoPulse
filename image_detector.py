"""
EcoPulse - Visual detection module

"""

import argparse
import csv
import glob
import os
from pathlib import Path

from ultralytics import YOLO

# Image formats YOLOv8 / OpenCV can read directly -- no conversion needed
# for any of these, unlike mp3/m4a audio which needs ffmpeg.
SUPPORTED_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]


def find_image_files(data_dir):
    """Case-insensitive glob across every supported image extension."""
    files = []
    for ext in SUPPORTED_IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(data_dir, f"*{ext.upper()}")))
    return sorted(set(files))


def resolve_weights(out_dir, requested_weights):
    """If image_finetune.py has produced a fine-tuned model, prefer it
    over the plain pretrained weights (unless the user explicitly asked
    for something else via --weights)."""
    finetuned = Path(out_dir) / "image_model_best.pt"
    if requested_weights == "yolov8n.pt" and finetuned.exists():
        print(f"Found fine-tuned weights at {finetuned}, using those instead of the base model.")
        return str(finetuned)
    return requested_weights


def run_detection(args):
    out_dir = Path(args.out_dir)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)

    weights = resolve_weights(out_dir, args.weights)
    model = YOLO(weights)  # auto-downloads yolov8n.pt on first run if using base weights
    elephant_class_ids = [k for k, v in model.names.items() if "elephant" in v.lower()]
    print("Elephant class id(s) in this model:", elephant_class_ids)

    files = find_image_files(args.data_dir)
    print(f"Found {len(files)} images in {args.data_dir} "
          f"(formats: {', '.join(SUPPORTED_IMAGE_EXTS)})")

    rows = []
    batch_size = 16
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        results = model.predict(batch, conf=args.conf, verbose=False)
        for path, r in zip(batch, results):
            boxes = r.boxes
            elephant_boxes = [b for b in boxes if int(b.cls[0]) in elephant_class_ids]
            max_conf = max((float(b.conf[0]) for b in elephant_boxes), default=0.0)
            rows.append({
                "file": os.path.basename(path),
                "num_elephants_detected": len(elephant_boxes),
                "max_confidence": round(max_conf, 3),
                "elephant_present": len(elephant_boxes) > 0,
            })
            if args.save_annotated and len(elephant_boxes) > 0:
                annotated = r.plot()  # numpy array (BGR) with boxes drawn
                import cv2
                cv2.imwrite(str(out_dir / "annotated" / os.path.basename(path)), annotated)
        print(f"Processed {min(i + batch_size, len(files))}/{len(files)}")

    csv_path = out_dir / "image_detections.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n_detected = sum(r["elephant_present"] for r in rows)
    print(f"\nElephant detected in {n_detected}/{len(rows)} images")
    print(f"Saved detection log to {csv_path}")
    if args.save_annotated:
        print(f"Saved annotated images to {out_dir / 'annotated'}")


def predict_single(model, image_path, conf=0.25):
    """Helper used by fusion_alert.py for a single-frame check."""
    elephant_class_ids = [k for k, v in model.names.items() if "elephant" in v.lower()]
    r = model.predict(image_path, conf=conf, verbose=False)[0]
    elephant_boxes = [b for b in r.boxes if int(b.cls[0]) in elephant_class_ids]
    max_conf = max((float(b.conf[0]) for b in elephant_boxes), default=0.0)
    return len(elephant_boxes) > 0, max_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=r"D:\Researches\Data Odyssey 2026\unique\Data Odyssey 2026 Project\data\images",
    )
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save_annotated", action="store_true", default=True)
    args = parser.parse_args()
    run_detection(args)

# ---------------------------------------------------------------------
# OPTIONAL STRETCH GOAL: fine-tuning on your own labeled data
# ---------------------------------------------------------------------
# 1. Label a subset of images (e.g. 100-150) at https://roboflow.com
#    (free tier) or with labelImg, exporting in "YOLOv8" format. You'll
#    get a data.yaml + images/ + labels/ folder structure.
# 2. Then fine-tune like this:
#
#   from ultralytics import YOLO
#   model = YOLO("yolov8n.pt")
#   model.train(data="path/to/data.yaml", epochs=50, imgsz=640)
#
# 3. Use the resulting best.pt as --weights above.
