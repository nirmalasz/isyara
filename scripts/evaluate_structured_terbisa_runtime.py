"""Evaluate the ISYARA structured recognizer on real TERBISA YOLO images.

This script intentionally does not use synthetic probabilities or temporal
stabilization. It measures the per-image recognition decision before runtime
sequence smoothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from statistics import median

import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.assessment_service.assessment_service.inference.bisindo_classifier import BisindoYoloClassifier
from services.assessment_service.assessment_service.inference.calibration import BisindoCalibration
from services.assessment_service.assessment_service.inference.stabilization import PredictionStabilizer
from services.assessment_service.assessment_service.inference.structured_recognition import TerbisaStructureRecognizer


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ROI_PADDING_BY_TYPE = {
    "medium": 0.55,
    "face_context": 1.1,
    "upper_body": 1.35,
    "combined": 0.65,
    "combined_face_context": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("terbisa-3"))
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evaluation"))
    return parser.parse_args()


def clean_label(label: str):
    label = label.replace(" -BISINDO-", "").replace("-BISINDO-", "")
    label = label.replace("Mau-Ingin", "Mau")
    return "Terima-kasih" if label in {"Terima kasih", "Terima-kasih"} else label


def display_label(label: str | None):
    if not label:
        return None
    return "Terima kasih" if label == "Terima-kasih" else label


def canonical_label(label: str | None):
    cleaned = clean_label(str(label or ""))
    return "Terima-kasih" if cleaned in {"Terima kasih", "Terima-kasih"} else cleaned


class StructuredRuntimeEvaluator:
    def __init__(self, dataset: Path, device: str | None = None):
        self.dataset = dataset
        self.names = self._load_names()
        self.classifier = BisindoYoloClassifier()
        if device:
            self.classifier.device = device
        self.classifier.load()
        self.calibration = BisindoCalibration()
        self.structure_recognizer = TerbisaStructureRecognizer()
        self.mp_hands = None
        self.mp_face = None
        self._init_mediapipe()

    def _load_names(self):
        data = yaml.safe_load((self.dataset / "data.yaml").read_text())
        return {index: clean_label(name) for index, name in enumerate(data["names"])}

    def _init_mediapipe(self):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

        hand_model = Path("models/mediapipe/hand_landmarker.task")
        face_model = Path("models/mediapipe/face_detector.tflite")
        self.mp_hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(hand_model), delegate=BaseOptions.Delegate.CPU),
                running_mode=VisionTaskRunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.35,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
        )
        self.mp_face = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(face_model), delegate=BaseOptions.Delegate.CPU),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_detection_confidence=0.35,
            )
        )

    def iter_split(self, split: str):
        image_dir = self.dataset / split / "images"
        label_dir = self.dataset / split / "labels"
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            label = self.read_label(label_path)
            if label:
                yield label, image_path

    def read_label(self, label_path: Path):
        lines = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
        if not lines:
            return None
        class_id = int(lines[0][0])
        return self.names[class_id]

    def evaluate_image(self, true_label: str, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        structure = self.extract_structure(image)
        rois = self.generate_rois(image, structure)
        raw_result = self.predict_image(image)
        raw_top = raw_result["raw_predictions"][0] if raw_result["raw_predictions"] else None
        structured = self.structured_decision(raw_result["raw_predictions"], structure, structure["hands_detected"])
        roi_results = []
        if rois:
            for roi in rois:
                cropped = image.crop((roi["x1"], roi["y1"], roi["x2"], roi["y2"]))
                prediction = self.predict_image(cropped)
                roi_results.append({"roi": roi, **prediction})
            aggregated = self.aggregate_roi_predictions(roi_results, structure)
        else:
            aggregated = structured
        return {
            "image": str(image_path),
            "true": true_label,
            "raw_prediction": self.key(raw_top.get("label")) if raw_top else None,
            "raw_confidence": raw_top.get("confidence") if raw_top else None,
            "raw_top3": raw_result["raw_predictions"][:3],
            "structured_prediction": aggregated["prediction"],
            "structured_confidence": aggregated["confidence"],
            "structured_rejected": aggregated["rejected"],
            "structured_reason": aggregated["reason"],
            "structured_top3": aggregated["top3"],
            "hands_detected": structure["hands_detected"],
            "body_regions": "|".join(hand.get("body_region", "") for hand in structure.get("hands", [])),
            "eligible_classes": "|".join(aggregated["eligible_classes"]),
            "masked_classes": "|".join(aggregated["masked_classes"]),
            "roi_count": len(rois),
            "selected_roi": aggregated.get("selected_roi"),
        }

    def predict_image(self, image: Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        result = self.classifier.predict(buffer.getvalue())
        predictions = result.as_dict().get("raw_predictions", [])
        return {"raw_predictions": predictions, "latency_ms": result.latency_ms}

    def structured_decision(self, raw_predictions, structure, hands_detected):
        calibrated = self.calibration.calibrate_predictions(raw_predictions)
        structured = self.structure_recognizer.apply(calibrated, structure=structure, hands_detected=hands_detected)
        top, reason, _margin = self.calibration.decision(structured["predictions"])
        prediction = self.key(top.get("label")) if top and not reason else None
        return {
            "prediction": prediction,
            "confidence": top.get("calibrated_confidence") if top else None,
            "rejected": bool(reason),
            "reason": reason,
            "top3": structured["predictions"][:3],
            "eligible_classes": structured["eligible_classes"],
            "masked_classes": structured["masked_classes"],
            "selected_roi": "full_image",
        }

    def aggregate_roi_predictions(self, roi_results, structure):
        by_label = defaultdict(list)
        exemplar = {}
        source_roi = {}
        for result in roi_results:
            roi_type = result["roi"]["type"]
            for item in result["raw_predictions"]:
                label = self.key(item.get("label"))
                confidence = float(item.get("confidence") or 0)
                by_label[label].append(confidence)
                if label not in exemplar or confidence > float(exemplar[label].get("confidence") or 0):
                    exemplar[label] = {**item, "label": label}
                    source_roi[label] = roi_type
        aggregated = []
        for label, values in by_label.items():
            mean_score = sum(values) / len(values)
            score = mean_score * 0.7 + median(values) * 0.3
            aggregated.append({**exemplar[label], "confidence": score, "roi_scores": values})
        calibrated = self.calibration.calibrate_predictions(aggregated)
        structured = self.structure_recognizer.apply(calibrated, structure=structure, hands_detected=structure["hands_detected"])
        top, reason, _margin = self.calibration.decision(structured["predictions"])
        prediction = self.key(top.get("label")) if top and not reason else None
        return {
            "prediction": prediction,
            "confidence": top.get("calibrated_confidence") if top else None,
            "rejected": bool(reason),
            "reason": reason,
            "top3": structured["predictions"][:3],
            "eligible_classes": structured["eligible_classes"],
            "masked_classes": structured["masked_classes"],
            "selected_roi": source_roi.get(self.key(top.get("label"))) if top else None,
        }

    def extract_structure(self, image: Image.Image):
        import mediapipe as mp
        import numpy as np

        arr = np.array(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        hand_results = self.mp_hands.detect(mp_image)
        face_results = self.mp_face.detect(mp_image)
        face_box = self.face_box(face_results, image.size)
        landmarks = hand_results.hand_landmarks or []
        handedness = hand_results.handedness or []
        hands = []
        for index, hand in enumerate(landmarks):
            points = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand]
            bounds = self.bounds(points)
            center = {"x": (bounds["x1"] + bounds["x2"]) / 2, "y": (bounds["y1"] + bounds["y2"]) / 2}
            handed = handedness[index][0].category_name if index < len(handedness) and handedness[index] else f"Hand {index + 1}"
            hands.append(
                {
                    "handedness": handed,
                    "center": center,
                    "bounds": bounds,
                    "landmarks": points,
                    "finger_states": self.finger_states(points, handed),
                    "geometry": self.hand_geometry(points),
                    "body_region": self.body_region(center, face_box),
                    "body_distances": self.body_distances(center, face_box),
                }
            )
        return {
            "hands_detected": len(hands),
            "handedness": [hand["handedness"] for hand in hands],
            "face": face_box,
            "hands": hands,
            "two_hand_distance": self.distance(hands[0]["center"], hands[1]["center"]) if len(hands) >= 2 else None,
            "hands_close": self.distance(hands[0]["center"], hands[1]["center"]) < 0.22 if len(hands) >= 2 else False,
            "two_hand_geometry": self.two_hand_geometry(hands),
        }

    def generate_rois(self, image: Image.Image, structure):
        width, height = image.size
        hands = structure.get("hands", [])
        if not hands:
            return []
        rois = []
        for hand in hands:
            bounds = hand["bounds"]
            prefix = hand["handedness"].lower()
            self.add_roi(rois, bounds, f"{prefix}_medium", width, height, "medium")
            if structure.get("face"):
                self.add_roi(rois, self.union_bounds([bounds, structure["face"]]), f"{prefix}_face_context", width, height, "face_context")
            self.add_roi(rois, bounds, f"{prefix}_upper_body", width, height, "upper_body")
        if len(hands) >= 2:
            combined = self.union_bounds([hand["bounds"] for hand in hands])
            self.add_roi(rois, combined, "combined_hands", width, height, "combined")
            if structure.get("face"):
                self.add_roi(rois, self.union_bounds([combined, structure["face"]]), "combined_face_context", width, height, "combined_face_context")
        return self.unique_rois(rois)[:5]

    def add_roi(self, rois, bounds, roi_type, image_width, image_height, scale_type):
        width = bounds["x2"] - bounds["x1"]
        height = bounds["y2"] - bounds["y1"]
        padding = ROI_PADDING_BY_TYPE[scale_type]
        pad_x = max(width * padding, 0.05)
        pad_y = max(height * padding, 0.05)
        upper_bias = 1.65 if scale_type in {"upper_body", "combined_face_context"} else 1
        lower_bias = 0.75 if scale_type in {"upper_body", "combined_face_context"} else 1
        roi = {
            "type": roi_type,
            "x1": max(0, round((bounds["x1"] - pad_x) * image_width)),
            "y1": max(0, round((bounds["y1"] - pad_y * upper_bias) * image_height)),
            "x2": min(image_width, round((bounds["x2"] + pad_x) * image_width)),
            "y2": min(image_height, round((bounds["y2"] + pad_y * lower_bias) * image_height)),
        }
        if roi["x2"] > roi["x1"] and roi["y2"] > roi["y1"]:
            rois.append(roi)

    def face_box(self, face_results, image_size):
        detections = face_results.detections if face_results else None
        if not detections:
            return None
        box = detections[0].bounding_box
        image_width, image_height = image_size
        x1 = max(0, box.origin_x / image_width)
        y1 = max(0, box.origin_y / image_height)
        x2 = min(1, (box.origin_x + box.width) / image_width)
        y2 = min(1, (box.origin_y + box.height) / image_height)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1, "center": {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2}}

    def body_anchors(self, face):
        if not face:
            return {"forehead": {"x": 0.5, "y": 0.24}, "mouth": {"x": 0.5, "y": 0.36}, "chin": {"x": 0.5, "y": 0.42}, "chest": {"x": 0.5, "y": 0.62}, "torso": {"x": 0.5, "y": 0.76}}
        h = face["height"]
        return {
            "forehead": {"x": face["center"]["x"], "y": face["y1"] + h * 0.22},
            "head": face["center"],
            "mouth": {"x": face["center"]["x"], "y": face["y1"] + h * 0.68},
            "chin": {"x": face["center"]["x"], "y": face["y2"]},
            "chest": {"x": face["center"]["x"], "y": min(1, face["y2"] + h * 1.35)},
            "shoulders": {"x": face["center"]["x"], "y": min(1, face["y2"] + h * 0.9)},
            "torso": {"x": face["center"]["x"], "y": min(1, face["y2"] + h * 2.25)},
        }

    def body_distances(self, center, face):
        return {name: round(self.distance(center, point), 4) for name, point in self.body_anchors(face).items()}

    def body_region(self, center, face):
        distances = self.body_distances(center, face)
        return min(distances, key=distances.get)

    def finger_states(self, hand, handedness):
        handed = str(handedness or "").lower()
        return {
            "thumb": hand[4]["x"] > hand[3]["x"] if "left" in handed else hand[4]["x"] < hand[3]["x"],
            "index": hand[8]["y"] < hand[6]["y"],
            "middle": hand[12]["y"] < hand[10]["y"],
            "ring": hand[16]["y"] < hand[14]["y"],
            "pinky": hand[20]["y"] < hand[18]["y"],
        }

    def hand_geometry(self, hand):
        palm_width = self.distance(hand[5], hand[17])
        palm_height = self.distance(hand[0], hand[9])
        spread = self.distance(hand[8], hand[20]) / max(0.001, palm_width)
        return {
            "palm_aspect": round(palm_width / max(0.001, palm_height), 4),
            "openness": round(spread, 4),
            "rotation": round(math.atan2(hand[5]["y"] - hand[17]["y"], hand[5]["x"] - hand[17]["x"]), 4),
            "fingertip_spread": round(spread, 4),
        }

    def two_hand_geometry(self, hands):
        if len(hands) < 2:
            return None
        return {
            "wrist_distance": round(self.distance(hands[0]["landmarks"][0], hands[1]["landmarks"][0]), 4),
            "palm_distance": round(self.distance(hands[0]["center"], hands[1]["center"]), 4),
            "relative_height": round(hands[0]["center"]["y"] - hands[1]["center"]["y"], 4),
            "overlap": self.boxes_overlap(hands[0]["bounds"], hands[1]["bounds"]),
        }

    def bounds(self, points):
        xs = [point["x"] for point in points]
        ys = [point["y"] for point in points]
        return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}

    def union_bounds(self, bounds_list):
        return {"x1": min(item["x1"] for item in bounds_list), "y1": min(item["y1"] for item in bounds_list), "x2": max(item["x2"] for item in bounds_list), "y2": max(item["y2"] for item in bounds_list)}

    def unique_rois(self, rois):
        seen = set()
        output = []
        for roi in rois:
            key = (roi["type"], round(roi["x1"] / 8), round(roi["y1"] / 8), round(roi["x2"] / 8), round(roi["y2"] / 8))
            if key in seen:
                continue
            seen.add(key)
            output.append(roi)
        return output

    def distance(self, a, b):
        return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

    def boxes_overlap(self, a, b):
        return a["x1"] < b["x2"] and a["x2"] > b["x1"] and a["y1"] < b["y2"] and a["y2"] > b["y1"]

    def key(self, label):
        return canonical_label(label)


def summarize(rows, classes):
    per_class = {}
    for label in classes:
        subset = [row for row in rows if row["true"] == label]
        raw_correct = sum(1 for row in subset if row["raw_prediction"] == label)
        structured_correct = sum(1 for row in subset if row["structured_prediction"] == label)
        rejected = sum(1 for row in subset if row["structured_rejected"])
        confusions = Counter(row["structured_prediction"] or "REJECTED" for row in subset if row["structured_prediction"] != label)
        per_class[label] = {
            "samples": len(subset),
            "raw_correct": raw_correct,
            "raw_accuracy": raw_correct / len(subset) if subset else 0,
            "structured_correct": structured_correct,
            "structured_accuracy": structured_correct / len(subset) if subset else 0,
            "rejected": rejected,
            "main_confusion": confusions.most_common(1)[0][0] if confusions else None,
        }
    return per_class


def confusion_matrix(rows, classes, prediction_key):
    matrix = {actual: {predicted: 0 for predicted in [*classes, "REJECTED"]} for actual in classes}
    for row in rows:
        actual = row["true"]
        predicted = row[prediction_key] or "REJECTED"
        matrix[actual][predicted] = matrix[actual].get(predicted, 0) + 1
    return matrix


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "true",
        "raw_prediction",
        "raw_confidence",
        "structured_prediction",
        "structured_confidence",
        "structured_rejected",
        "structured_reason",
        "hands_detected",
        "body_regions",
        "eligible_classes",
        "masked_classes",
        "roi_count",
        "selected_roi",
        "raw_top3",
        "structured_top3",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if key.endswith("top3") else row.get(key) for key in fieldnames})


def write_matrix_csv(path, matrix, classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [*classes, "REJECTED"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual", *columns])
        for actual in classes:
            writer.writerow([actual, *[matrix[actual].get(column, 0) for column in columns]])


def simulate_sequences(rows, classes):
    results = {}
    for label in classes:
        candidates = [row for row in rows if row["true"] == label and row["structured_prediction"]]
        if not candidates:
            results[label] = {"accepted": False, "prediction": None, "reason": "no_structured_prediction"}
            continue
        sample = candidates[0]
        stabilizer = PredictionStabilizer()
        accepted = None
        for index in range(5):
            accepted = stabilizer.evaluate(sample["structured_prediction"], sample["structured_confidence"] or 0, timestamp_ms=100 + index * 150, probabilities={sample["structured_prediction"]: sample["structured_confidence"] or 0})
        results[label] = {"accepted": bool(accepted["accepted"]), "prediction": accepted["stable_label"], "reason": accepted["rejection_reason"]}
    return results


def main():
    args = parse_args()
    evaluator = StructuredRuntimeEvaluator(args.dataset, device=args.device)
    classes = [evaluator.names[index] for index in sorted(evaluator.names)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for split in args.splits:
        rows = []
        seen = Counter()
        for true_label, image_path in evaluator.iter_split(split):
            if args.limit_per_class and seen[true_label] >= args.limit_per_class:
                continue
            seen[true_label] += 1
            rows.append(evaluator.evaluate_image(true_label, image_path))
            if len(rows) % 50 == 0:
                print(f"[{split}] evaluated {len(rows)} images")
        summary = summarize(rows, classes)
        raw_matrix = confusion_matrix(rows, classes, "raw_prediction")
        structured_matrix = confusion_matrix(rows, classes, "structured_prediction")
        sequence = simulate_sequences(rows, classes)
        all_summaries[split] = {
            "samples": len(rows),
            "per_class": summary,
            "membaca_false_raw": sum(1 for row in rows if row["true"] != "Membaca" and row["raw_prediction"] == "Membaca"),
            "membaca_false_structured": sum(1 for row in rows if row["true"] != "Membaca" and row["structured_prediction"] == "Membaca"),
            "sequence_simulation": sequence,
        }
        write_csv(args.output_dir / f"structured_runtime_{split}_predictions.csv", rows)
        write_matrix_csv(args.output_dir / f"classifier_only_{split}_confusion.csv", raw_matrix, classes)
        write_matrix_csv(args.output_dir / f"structured_{split}_confusion.csv", structured_matrix, classes)
        (args.output_dir / f"structured_runtime_{split}_summary.json").write_text(json.dumps(all_summaries[split], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n{split.upper()} summary")
        print("Class | Samples | Raw acc | Structured acc | Rejected | Main confusion")
        for label in classes:
            item = summary[label]
            print(f"{display_label(label)} | {item['samples']} | {item['raw_accuracy']:.3f} | {item['structured_accuracy']:.3f} | {item['rejected']} | {display_label(item['main_confusion']) or '-'}")
        print(f"False Membaca raw={all_summaries[split]['membaca_false_raw']} structured={all_summaries[split]['membaca_false_structured']}")
    (args.output_dir / "structured_runtime_summary.json").write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
