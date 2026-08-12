"""Evaluate bisindo_words_cls_v2.pt on original TERBISA YOLO frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.assessment_service.assessment_service.inference.bisindo_classifier import clean_classifier_label


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("terbisa-3"))
    parser.add_argument("--model", type=Path, default=Path("models/bisindo_words_cls_v2.pt"))
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
    return parser.parse_args()


def clean_dataset_label(label):
    return clean_classifier_label(label.replace(" -BISINDO-", "").replace("Mau-Ingin", "Mau"))


def key(label):
    return "Terima-kasih" if label in {"Terima kasih", "Terima-kasih"} else label


def load_names(dataset):
    data = yaml.safe_load((dataset / "data.yaml").read_text())
    return {index: key(clean_dataset_label(name)) for index, name in enumerate(data["names"])}


def read_label(label_path, names):
    lines = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
    if not lines:
        return None
    return names[int(lines[0][0])]


def iter_split(dataset, split, names):
    for image_path in sorted((dataset / split / "images").iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = dataset / split / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        label = read_label(label_path, names)
        if label:
            yield label, image_path


def summarize(rows, classes):
    summary = {}
    for label in classes:
        subset = [row for row in rows if row["actual"] == label]
        correct = sum(1 for row in subset if row["predicted"] == label)
        confusions = Counter(row["predicted"] for row in subset if row["predicted"] != label)
        summary[label] = {
            "samples": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset) if subset else 0,
            "main_confusion": confusions.most_common(1)[0][0] if confusions else None,
        }
    return summary


def matrix(rows, classes):
    output = {actual: {predicted: 0 for predicted in classes} for actual in classes}
    for row in rows:
        output[row["actual"]][row["predicted"]] = output[row["actual"]].get(row["predicted"], 0) + 1
    return output


def write_matrix(path, data, classes):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual", *classes])
        for actual in classes:
            writer.writerow([actual, *[data[actual].get(predicted, 0) for predicted in classes]])


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    names = load_names(args.dataset)
    classes = [names[index] for index in sorted(names)]
    all_summary = {}
    for split in args.splits:
        rows = []
        for index, (actual, image_path) in enumerate(iter_split(args.dataset, split, names), start=1):
            image = Image.open(image_path).convert("RGB")
            result = model.predict(image, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            probs = result.probs
            top_indices = [int(i) for i in probs.top5]
            predicted = key(clean_classifier_label(model.names[int(probs.top1)]))
            rows.append(
                {
                    "image": str(image_path),
                    "actual": actual,
                    "predicted": predicted,
                    "confidence": float(probs.top1conf),
                    "top3": "|".join(key(clean_classifier_label(model.names[i])) for i in top_indices[:3]),
                }
            )
            if index % 100 == 0:
                print(f"[{split}] {index}")
        summary = summarize(rows, classes)
        conf = matrix(rows, classes)
        all_summary[split] = {
            "samples": len(rows),
            "accuracy": sum(1 for row in rows if row["actual"] == row["predicted"]) / len(rows) if rows else 0,
            "false_membaca": sum(1 for row in rows if row["actual"] != "Membaca" and row["predicted"] == "Membaca"),
            "per_class": summary,
        }
        with (args.output_dir / f"original_frames_{split}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "actual", "predicted", "confidence", "top3"])
            writer.writeheader()
            writer.writerows(rows)
        write_matrix(args.output_dir / f"original_frames_{split}_confusion.csv", conf, classes)
        (args.output_dir / f"original_frames_{split}_summary.json").write_text(json.dumps(all_summary[split], indent=2), encoding="utf-8")
        print(f"\n{split}: accuracy={all_summary[split]['accuracy']:.3f} false_membaca={all_summary[split]['false_membaca']}")
        print("Class | Samples | Accuracy | Main confusion")
        for label in classes:
            item = summary[label]
            print(f"{label} | {item['samples']} | {item['accuracy']:.3f} | {item['main_confusion'] or '-'}")
    (args.output_dir / "original_frames_summary.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
