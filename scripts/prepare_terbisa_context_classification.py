"""Build a multi-context TERBISA classification dataset for BISINDO words."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

from PIL import Image, ImageEnhance, ImageOps


SPLITS = {"train": "train", "valid": "val", "test": "test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = [
    "Anda",
    "Apa",
    "Berhenti",
    "Bodoh",
    "Cantik",
    "Halo",
    "Hati-hati",
    "Lelah",
    "Maaf",
    "Makan",
    "Mau",
    "Membaca",
    "Nama",
    "Sama-sama",
    "Saya",
    "Siapa",
    "Sombong",
    "Takut",
    "Terima-kasih",
]
TRAIN_VARIANTS = [
    ("tight", 0.20, "bbox"),
    ("medium", 0.55, "bbox"),
    ("large_context", 1.10, "bbox"),
    ("square_context", 0.85, "square"),
    ("upper_context", 1.35, "upper"),
]
EVAL_VARIANTS = [
    ("medium", 0.55, "bbox"),
    ("large_context", 1.10, "bbox"),
    ("square_context", 0.85, "square"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("terbisa-3"))
    parser.add_argument("--output", type=Path, default=Path("datasets/terbisa-cls-v2"))
    parser.add_argument("--contact-sheet", type=Path, default=Path("datasets/terbisa-cls-v2/contact_sheet.jpg"))
    parser.add_argument("--max-train-variants-per-object", type=int, default=4)
    parser.add_argument("--context-face-padding", type=float, default=0.22)
    return parser.parse_args()


def image_for_label(label_path, images_dir):
    for suffix in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{label_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def parse_yolo_box(line, image_width, image_height):
    parts = line.split()
    class_id = int(parts[0])
    x_center, y_center, box_width, box_height = [float(value) for value in parts[1:5]]
    width = box_width * image_width
    height = box_height * image_height
    x1 = (x_center * image_width) - (width / 2)
    y1 = (y_center * image_height) - (height / 2)
    x2 = (x_center * image_width) + (width / 2)
    y2 = (y_center * image_height) + (height / 2)
    return class_id, (x1, y1, x2, y2), {
        "bbox_width_ratio": box_width,
        "bbox_height_ratio": box_height,
        "bbox_area_ratio": box_width * box_height,
        "bbox_center_x": x_center,
        "bbox_center_y": y_center,
    }


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )


def expand_box(box, image_width, image_height, padding, mode="bbox"):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if mode == "square":
        side = max(width, height) * (1 + padding * 2)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return clamp_box((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), image_width, image_height)
    if mode == "upper":
        pad_x = width * padding
        return clamp_box((x1 - pad_x, y1 - height * padding * 1.65, x2 + pad_x, y2 + height * padding * 0.75), image_width, image_height)
    pad_x = width * padding
    pad_y = height * padding
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), image_width, image_height)


def union_boxes(boxes):
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersects_or_near(a, b, margin):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + margin < bx1 or bx2 + margin < ax1 or ay2 + margin < by1 or by2 + margin < ay1)


def detect_faces(image):
    try:
        import cv2
        import numpy as np
    except Exception:
        return [], "unavailable"

    if not hasattr(detect_faces, "_detector"):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detect_faces._detector = cv2.CascadeClassifier(str(cascade_path))  # type: ignore[attr-defined]
        detect_faces._cascade_available = not detect_faces._detector.empty()  # type: ignore[attr-defined]
    if not detect_faces._cascade_available:  # type: ignore[attr-defined]
        return [], "unavailable"
    array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    detected = detect_faces._detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(35, 35))  # type: ignore[attr-defined]
    height, width = array.shape[:2]
    faces = [clamp_box((x, y, x + w, y + h), width, height) for x, y, w, h in detected]
    return faces, "opencv_haar"


def context_bucket(annotation_box, image_width, image_height, face_boxes, sibling_boxes):
    x1, y1, x2, y2 = annotation_box
    width = x2 - x1
    height = y2 - y1
    center_y = ((y1 + y2) / 2) / image_height
    torso_context = center_y >= 0.46 or y2 / image_height >= 0.58
    face_margin = max(width, height) * 1.25
    face_context = any(intersects_or_near(annotation_box, face, face_margin) for face in face_boxes)
    two_hand = any(intersects_or_near(annotation_box, other, max(width, height) * 0.9) for other in sibling_boxes)
    if face_context and two_hand:
        return "hand_face_two_hand"
    if face_context:
        return "hand_face"
    if two_hand:
        return "two_hand"
    if torso_context:
        return "hand_torso"
    return "hand_dominant"


def deterministic_variants(annotation_box, image_width, image_height, face_boxes, sibling_boxes, split):
    variants = TRAIN_VARIANTS if split == "train" else EVAL_VARIANTS
    boxes = [(name, expand_box(annotation_box, image_width, image_height, padding, mode)) for name, padding, mode in variants]
    face_margin = max(annotation_box[2] - annotation_box[0], annotation_box[3] - annotation_box[1]) * 1.4
    nearby_faces = [face for face in face_boxes if intersects_or_near(annotation_box, face, face_margin)]
    if nearby_faces:
        face_union = expand_box(union_boxes([annotation_box, *nearby_faces]), image_width, image_height, 0.22, "square")
        boxes.append(("face_union", face_union))
    nearby_hands = [box for box in sibling_boxes if intersects_or_near(annotation_box, box, max(annotation_box[2] - annotation_box[0], annotation_box[3] - annotation_box[1]) * 1.0)]
    if nearby_hands:
        hand_union = expand_box(union_boxes([annotation_box, *nearby_hands]), image_width, image_height, 0.45, "square")
        boxes.append(("two_hand_union", hand_union))
    return dedupe_variant_boxes(boxes)


def dedupe_variant_boxes(boxes):
    seen = set()
    unique = []
    for name, box in boxes:
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if box[2] - box[0] < 24 or box[3] - box[1] < 24:
            continue
        key = tuple(round(value / 4) for value in box)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, box))
    return unique


def training_augmentations(image):
    return [
        ("orig", image),
        ("bright", ImageEnhance.Brightness(image).enhance(1.12)),
        ("desat", ImageEnhance.Color(image).enhance(0.82)),
    ]


def average_hash(image, size=8):
    small = ImageOps.grayscale(image).resize((size, size), Image.Resampling.BILINEAR)
    pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel > avg else "0" for pixel in pixels)
    return hashlib.sha1(bits.encode("ascii")).hexdigest()[:16]


def ensure_class_dirs(output):
    for split in SPLITS.values():
        for class_name in CLASS_NAMES:
            (output / split / class_name).mkdir(parents=True, exist_ok=True)


def read_annotations(label_path, width, height, malformed):
    annotations = []
    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    if not lines:
        malformed.append({"label": str(label_path), "reason": "empty_label"})
        return annotations
    for object_index, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            malformed.append({"label": str(label_path), "line": line, "reason": "short_annotation"})
            continue
        try:
            class_id, box, stats = parse_yolo_box(line, width, height)
        except (TypeError, ValueError) as exc:
            malformed.append({"label": str(label_path), "line": line, "reason": f"parse_error:{exc}"})
            continue
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            malformed.append({"label": str(label_path), "line": line, "reason": "unknown_class"})
            continue
        annotations.append({"class_id": class_id, "box": box, "stats": stats, "object_index": object_index})
    return annotations


def convert_dataset(source, output, max_train_variants):
    ensure_class_dirs(output)
    counts = {split: Counter() for split in SPLITS.values()}
    source_counts = {split: Counter() for split in SPLITS.values()}
    variant_counts = defaultdict(Counter)
    context_counts = defaultdict(Counter)
    bbox_stats = defaultdict(lambda: defaultdict(list))
    duplicate_hashes = defaultdict(Counter)
    malformed = []
    examples = []
    face_detector_status = "not_used"

    for source_split, output_split in SPLITS.items():
        images_dir = source / source_split / "images"
        labels_dir = source / source_split / "labels"
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = image_for_label(label_path, images_dir)
            if image_path is None:
                malformed.append({"label": str(label_path), "reason": "missing_image"})
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                malformed.append({"image": str(image_path), "reason": f"invalid_image:{exc}"})
                continue
            width, height = image.size
            face_boxes, face_detector_status = detect_faces(image)
            annotations = read_annotations(label_path, width, height, malformed)
            for annotation in annotations:
                class_name = CLASS_NAMES[annotation["class_id"]]
                source_counts[output_split][class_name] += 1
                sibling_boxes = [item["box"] for item in annotations if item is not annotation]
                context = context_bucket(annotation["box"], width, height, face_boxes, sibling_boxes)
                context_counts[class_name][context] += 1
                for key, value in annotation["stats"].items():
                    bbox_stats[class_name][key].append(value)
                variants = deterministic_variants(annotation["box"], width, height, face_boxes, sibling_boxes, output_split)
                if output_split == "train":
                    variants = variants[:max_train_variants]
                for variant_name, crop_box in variants:
                    crop = image.crop(crop_box)
                    augmentations = training_augmentations(crop) if output_split == "train" else [("eval", crop)]
                    for aug_name, variant_image in augmentations:
                        output_name = f"{image_path.stem}_{annotation['object_index']}_{variant_name}_{aug_name}.jpg"
                        output_path = output / output_split / class_name / output_name
                        variant_image.save(output_path, quality=92)
                        counts[output_split][class_name] += 1
                        variant_counts[class_name][variant_name] += 1
                        if output_split == "train" and aug_name == "orig":
                            duplicate_hashes[class_name][average_hash(variant_image)] += 1
                        if len(examples) < 120 and aug_name in {"orig", "eval"}:
                            examples.append((output_path, f"{class_name}/{variant_name}"))

    return {
        "counts": counts,
        "source_counts": source_counts,
        "variant_counts": variant_counts,
        "context_counts": context_counts,
        "bbox_stats": bbox_stats,
        "duplicate_hashes": duplicate_hashes,
        "malformed": malformed,
        "examples": examples,
        "face_detector_status": face_detector_status,
    }


def summarize(results, output):
    context_report = {}
    for class_name in CLASS_NAMES:
        total = sum(results["context_counts"][class_name].values())
        context_report[class_name] = {
            key: {
                "count": count,
                "percent": round((count / total) * 100, 2) if total else 0,
            }
            for key, count in sorted(results["context_counts"][class_name].items())
        }
    duplicate_report = {}
    for class_name, hashes in results["duplicate_hashes"].items():
        repeated = [count for count in hashes.values() if count > 1]
        duplicate_report[class_name] = {
            "hash_groups": len(hashes),
            "near_duplicate_groups": len(repeated),
            "max_group_size": max(repeated) if repeated else 1,
            "near_duplicate_images": sum(repeated),
        }
    bbox_statistics = {}
    for class_name, values in results["bbox_stats"].items():
        bbox_statistics[class_name] = {
            key: {
                "median": median(items),
                "min": min(items),
                "max": max(items),
            }
            for key, items in values.items()
            if items
        }
    summary = {
        "source": "terbisa-3",
        "class_names": CLASS_NAMES,
        "splits": SPLITS,
        "face_detector_status": results["face_detector_status"],
        "source_counts": {split: dict(results["source_counts"][split]) for split in SPLITS.values()},
        "generated_counts": {split: dict(results["counts"][split]) for split in SPLITS.values()},
        "generated_totals": {split: sum(results["counts"][split].values()) for split in SPLITS.values()},
        "variant_counts": {class_name: dict(results["variant_counts"][class_name]) for class_name in CLASS_NAMES},
        "context_characteristics": context_report,
        "bbox_statistics": bbox_statistics,
        "near_duplicate_findings": duplicate_report,
        "augmentations": [
            "original",
            "brightness +12%",
            "desaturation -18%",
            "trainer HSV h=0.005 s=0.20 v=0.15",
            "trainer translate=0.04 scale=0.12 degrees=4 perspective=0.0005",
        ],
        "background_strategy": "Original contextual crops plus crop-level desaturation variants. Segmentation-based background neutralization was not used because reliable hand/arm/face/torso segmentation is not available in this repo.",
        "validation_policy": "Validation/test splits preserve TERBISA boundaries and use deterministic medium, large, square, face-union, and two-hand-union crops only.",
        "malformed": results["malformed"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "context_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_contact_sheet(examples, output_path):
    if not examples:
        return
    thumb_size = (150, 150)
    columns = 6
    rows = min(20, (len(examples) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + 26)), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(examples[: columns * rows]):
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, thumb_size)
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * (thumb_size[1] + 26)
        sheet.paste(image, (x + (thumb_size[0] - image.width) // 2, y))
        draw.text((x + 4, y + thumb_size[1] + 4), label[:24], fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def main():
    args = parse_args()
    results = convert_dataset(args.source, args.output, args.max_train_variants_per_object)
    summary = summarize(results, args.output)
    write_contact_sheet(results["examples"], args.contact_sheet)
    print(
        json.dumps(
            {
                "generated_totals": summary["generated_totals"],
                "source_counts": summary["source_counts"],
                "face_detector_status": summary["face_detector_status"],
                "malformed_annotations": len(summary["malformed"]),
                "report": str(args.output / "context_report.json"),
                "contact_sheet": str(args.contact_sheet),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
