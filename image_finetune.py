"""
EcoPulse - Fine-tune YOLOv8 on a REAL bounding-box labeled elephant dataset

"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def finetune(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from roboflow import Roboflow
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov8", location=str(out_dir / "roboflow_dataset"))

    data_yaml = Path(dataset.location) / "data.yaml"
    print(f"Downloaded dataset with data.yaml at {data_yaml}")

    model = YOLO(args.base_weights)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        project=str(out_dir / "runs"),
        name="elephant_finetune",
        patience=15,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    final_path = out_dir / "image_model_best.pt"
    shutil.copy(best_pt, final_path)
    print(f"\nFine-tuned weights saved to {final_path}")
    print("image_detector.py and app.py will automatically use this file if it exists "
          "in the outputs/ folder -- no other changes needed.")

    # Quick validation summary
    metrics = model.val(data=str(data_yaml))
    print(f"\nValidation mAP50: {metrics.box.map50:.3f}, mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="Your Roboflow API key")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug, e.g. 'walailak-university-w0yow'")
    parser.add_argument("--project", required=True, help="Roboflow project slug, e.g. 'elephant-model-nbftw'")
    parser.add_argument("--version", type=int, required=True, help="Dataset version number, e.g. 2")
    parser.add_argument("--base_weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()
    finetune(args)
