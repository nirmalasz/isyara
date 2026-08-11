"""Evaluate a YOLO classification model on the preserved TERBISA test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image


FOCUS_CLASSES = {"Halo", "Hati-hati", "Saya", "Makan", "Mau", "Terima-kasih"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/bisindo_words_cls.pt"))
    parser.add_argument("--data", type=Path, default=Path("datasets/terbisa-cls/test"))
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, default=Path("runs/classify/bisindo_words_cls_test_metrics.json"))
    parser.add_argument("--csv", type=Path, default=Path("runs/classify/bisindo_words_cls_test_predictions.csv"))
    return parser.parse_args()


def iter_images(data_root):
    for class_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.glob("*")):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                yield class_dir.name, image_path


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    class_names = [path.name for path in sorted(path for path in args.data.iterdir() if path.is_dir())]
    confusion = {actual: {predicted: 0 for predicted in class_names} for actual in class_names}
    per_class = {name: {"total": 0, "top1_correct": 0, "top5_correct": 0} for name in class_names}
    rows = []

    for actual, image_path in iter_images(args.data):
        image = Image.open(image_path).convert("RGB")
        result = model.predict(image, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        probs = result.probs
        top1_index = int(probs.top1)
        top5_indices = [int(index) for index in probs.top5]
        predicted = model.names[top1_index]
        top5_labels = [model.names[index] for index in top5_indices]
        confidence = float(probs.top1conf)
        per_class[actual]["total"] += 1
        per_class[actual]["top1_correct"] += int(predicted == actual)
        per_class[actual]["top5_correct"] += int(actual in top5_labels)
        confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1
        rows.append(
            {
                "image": str(image_path),
                "actual": actual,
                "predicted": predicted,
                "confidence": confidence,
                "top5": "|".join(top5_labels),
                "focus": actual in FOCUS_CLASSES,
            }
        )

    total = sum(item["total"] for item in per_class.values())
    top1 = sum(item["top1_correct"] for item in per_class.values())
    top5 = sum(item["top5_correct"] for item in per_class.values())
    metrics = {
        "model": str(args.model),
        "data": str(args.data),
        "total": total,
        "top1_accuracy": top1 / total if total else 0,
        "top5_accuracy": top5 / total if total else 0,
        "per_class": {
            name: {
                **values,
                "top1_accuracy": values["top1_correct"] / values["total"] if values["total"] else 0,
                "top5_accuracy": values["top5_correct"] / values["total"] if values["total"] else 0,
            }
            for name, values in per_class.items()
        },
        "confusion_matrix": confusion,
        "focus_classes": sorted(FOCUS_CLASSES),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "actual", "predicted", "confidence", "top5", "focus"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: metrics[key] for key in ["total", "top1_accuracy", "top5_accuracy"]}, indent=2))
    print(f"Metrics: {args.output}")
    print(f"Predictions: {args.csv}")


if __name__ == "__main__":
    main()
