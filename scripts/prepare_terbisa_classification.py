"""Convert TERBISA YOLO detection annotations into classification crops."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageOps


SPLITS = {"train": "train", "valid": "val", "test": "test"}
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("terbisa-3"))
    parser.add_argument("--output", type=Path, default=Path("datasets/terbisa-cls"))
    parser.add_argument("--padding", type=float, default=0.20, help="Context padding as a fraction of bbox width/height.")
    parser.add_argument("--contact-sheet", type=Path, default=Path("datasets/terbisa-cls/contact_sheet.jpg"))
    return parser.parse_args()


def yolo_to_xyxy(parts, image_width, image_height, padding):
    class_id = int(parts[0])
    x_center, y_center, box_width, box_height = [float(value) for value in parts[1:5]]
    width = box_width * image_width
    height = box_height * image_height
    pad_x = width * padding
    pad_y = height * padding
    x1 = max(0, round((x_center * image_width) - (width / 2) - pad_x))
    y1 = max(0, round((y_center * image_height) - (height / 2) - pad_y))
    x2 = min(image_width, round((x_center * image_width) + (width / 2) + pad_x))
    y2 = min(image_height, round((y_center * image_height) + (height / 2) + pad_y))
    return class_id, (x1, y1, x2, y2), {
        "bbox_width_ratio": box_width,
        "bbox_height_ratio": box_height,
        "bbox_area_ratio": box_width * box_height,
        "bbox_center_x": x_center,
        "bbox_center_y": y_center,
    }


def image_for_label(label_path, images_dir):
    for suffix in [".jpg", ".jpeg", ".png", ".webp"]:
        candidate = images_dir / f"{label_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def ensure_class_dirs(output):
    for split in SPLITS.values():
        for class_name in CLASS_NAMES:
            (output / split / class_name).mkdir(parents=True, exist_ok=True)


def convert_dataset(source, output, padding):
    ensure_class_dirs(output)
    counts = {split: Counter() for split in SPLITS.values()}
    dimensions = []
    bbox_stats = defaultdict(lambda: defaultdict(list))
    malformed = []
    examples = []

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
            lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
            if not lines:
                malformed.append({"label": str(label_path), "reason": "empty_label"})
                continue
            for object_index, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    malformed.append({"label": str(label_path), "line": line, "reason": "short_annotation"})
                    continue
                try:
                    class_id, box, stats = yolo_to_xyxy(parts, width, height, padding)
                except (TypeError, ValueError) as exc:
                    malformed.append({"label": str(label_path), "line": line, "reason": f"parse_error:{exc}"})
                    continue
                if class_id < 0 or class_id >= len(CLASS_NAMES):
                    malformed.append({"label": str(label_path), "line": line, "reason": "unknown_class"})
                    continue
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    malformed.append({"label": str(label_path), "line": line, "reason": "empty_crop"})
                    continue
                class_name = CLASS_NAMES[class_id]
                crop = image.crop(box)
                output_name = f"{image_path.stem}_{object_index}.jpg"
                output_path = output / output_split / class_name / output_name
                crop.save(output_path, quality=92)
                counts[output_split][class_name] += 1
                dimensions.append(crop.size)
                for key, value in stats.items():
                    bbox_stats[class_name][key].append(value)
                if len(examples) < 80:
                    examples.append((output_path, class_name))
    return counts, dimensions, bbox_stats, malformed, examples


def summarize(counts, dimensions, bbox_stats, malformed, padding, output):
    stats = {
        "padding": padding,
        "totals": {split: sum(counter.values()) for split, counter in counts.items()},
        "counts": {split: dict(sorted(counter.items())) for split, counter in counts.items()},
        "malformed": malformed,
        "example_crop_dimensions": [{"width": width, "height": height} for width, height in dimensions[:20]],
        "classes_with_few_samples": {},
        "bbox_statistics": {},
    }
    combined = Counter()
    for counter in counts.values():
        combined.update(counter)
    stats["classes_with_few_samples"] = {name: count for name, count in sorted(combined.items()) if count < 30}
    for class_name, values in bbox_stats.items():
        stats["bbox_statistics"][class_name] = {
            key: {
                "median": median(items),
                "min": min(items),
                "max": max(items),
            }
            for key, items in values.items()
            if items
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def write_contact_sheet(examples, output_path):
    if not examples:
        return
    thumb_size = (160, 160)
    columns = 5
    rows = min(16, (len(examples) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + 24)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (path, class_name) in enumerate(examples[: columns * rows]):
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, thumb_size)
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * (thumb_size[1] + 24)
        sheet.paste(image, (x + (thumb_size[0] - image.width) // 2, y))
        draw.text((x + 4, y + thumb_size[1] + 4), class_name, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def main():
    args = parse_args()
    counts, dimensions, bbox_stats, malformed, examples = convert_dataset(args.source, args.output, args.padding)
    stats = summarize(counts, dimensions, bbox_stats, malformed, args.padding, args.output)
    write_contact_sheet(examples, args.contact_sheet)
    print(json.dumps({key: stats[key] for key in ["padding", "totals", "classes_with_few_samples"]}, indent=2))
    if malformed:
        print(f"Malformed annotations: {len(malformed)}")
    print(f"Stats: {args.output / 'stats.json'}")
    print(f"Contact sheet: {args.contact_sheet}")


if __name__ == "__main__":
    main()
