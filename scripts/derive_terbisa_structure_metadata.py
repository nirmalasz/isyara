from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import cv2
import mediapipe as mp


LABELS = [
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/terbisa-cls-v2")
    parser.add_argument("--output", default="services/assessment_service/assessment_service/inference/terbisa_structure_metadata.json")
    parser.add_argument("--limit-per-class-split", type=int, default=36)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    rows = collect_rows(dataset, args.limit_per_class_split)
    data = {
        "source": {
            "dataset": "TERBISA",
            "classification_dataset": str(dataset),
            "derivation": "MediaPipe Hands + FaceDetection over representative train/val/test images.",
            "limit_per_class_split": args.limit_per_class_split,
            "image_count": len(rows),
            "notes": "Numeric structure profiles are derived from local TERBISA images. Runtime uses them for eligibility and soft reranking, not direct class prediction.",
        },
        "classes": build_profiles(rows),
        "confusion_groups": {
            "A": ["Saya", "Anda", "Apa"],
            "B": ["Nama", "Membaca"],
            "C": ["Takut", "Lelah"],
            "D": ["Siapa", "Cantik", "Bodoh"],
            "E": ["Sama-sama", "Terima-kasih"],
        },
    }
    Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {args.output} from {len(rows)} images")
    for label in LABELS:
        profile = data["classes"][label]
        print(
            f"{label}: samples={profile['samples']} mp_success={profile['mediapipe_success_rate']:.2f} "
            f"hands={profile['hand_count_distribution']} required={profile['required_hands']} "
            f"regions={profile['dominant_body_regions'][:3]}"
        )


def collect_rows(dataset: Path, limit_per_class_split: int):
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection
    rows = []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.35) as hands_model, mp_face.FaceDetection(
        model_selection=0, min_detection_confidence=0.35
    ) as face_model:
        for split in ["train", "val", "test"]:
            for label in LABELS:
                class_dir = dataset / split / label
                if not class_dir.exists():
                    continue
                files = sorted([*class_dir.glob("*.jpg"), *class_dir.glob("*.jpeg"), *class_dir.glob("*.png")])
                files = prefer_context_images(files)[:limit_per_class_split]
                for image_path in files:
                    row = analyze_image(image_path, label, split, hands_model, face_model)
                    rows.append(row)
    return rows


def prefer_context_images(files):
    priority = {"face_union_eval": 0, "large_context_eval": 1, "upper_body_eval": 2, "medium_eval": 3, "square_context_eval": 4}
    return sorted(files, key=lambda path: (priority_for(path.name, priority), path.name))


def priority_for(name, priority):
    for token, order in priority.items():
        if token in name:
            return order
    return 5


def analyze_image(image_path, label, split, hands_model, face_model):
    image = cv2.imread(str(image_path))
    if image is None:
        return {"label": label, "split": split, "path": str(image_path), "hands_detected": 0, "mediapipe_success": False}
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    hand_result = hands_model.process(rgb)
    face_result = face_model.process(rgb)
    face = face_box(face_result, w, h)
    hands = []
    hand_landmarks = hand_result.multi_hand_landmarks or []
    handedness = hand_result.multi_handedness or []
    for index, landmarks in enumerate(hand_landmarks):
        label_name = handedness[index].classification[0].label if index < len(handedness) and handedness[index].classification else f"Hand {index + 1}"
        points = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks.landmark]
        hands.append(hand_features(points, label_name, face))
    return {
        "label": label,
        "split": split,
        "path": str(image_path),
        "hands_detected": len(hands),
        "mediapipe_success": bool(hands),
        "face_detected": face is not None,
        "hands": hands,
        "two_hand_geometry": two_hand_geometry(hands),
    }


def hand_features(points, handedness, face):
    bounds = landmark_bounds(points)
    center = {"x": (bounds["x1"] + bounds["x2"]) / 2, "y": (bounds["y1"] + bounds["y2"]) / 2}
    fingers = finger_states(points, handedness)
    geom = hand_geometry(points)
    distances = body_distances(center, face)
    return {
        "handedness": handedness,
        "center": center,
        "bounds": bounds,
        "finger_states": fingers,
        "geometry": geom,
        "body_region": body_region(center, face),
        "body_distances": distances,
    }


def landmark_bounds(points):
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}


def finger_states(points, handedness):
    left = "left" in str(handedness).lower()
    thumb_open = points[4]["x"] > points[3]["x"] if left else points[4]["x"] < points[3]["x"]
    return {
        "thumb": thumb_open,
        "index": points[8]["y"] < points[6]["y"],
        "middle": points[12]["y"] < points[10]["y"],
        "ring": points[16]["y"] < points[14]["y"],
        "pinky": points[20]["y"] < points[18]["y"],
    }


def hand_geometry(points):
    palm_width = dist(points[5], points[17])
    palm_height = dist(points[0], points[9])
    spread = dist(points[8], points[20]) / max(0.001, palm_width)
    openness = sum(dist(points[idx], points[0]) for idx in [4, 8, 12, 16, 20]) / max(0.001, palm_width)
    return {
        "palm_aspect": palm_width / max(0.001, palm_height),
        "openness": openness,
        "fingertip_spread": spread,
        "rotation": math.atan2(points[5]["y"] - points[17]["y"], points[5]["x"] - points[17]["x"]),
        "index_vector_x": (points[8]["x"] - points[0]["x"]) / max(0.001, palm_width),
        "index_vector_y": (points[8]["y"] - points[0]["y"]) / max(0.001, palm_width),
    }


def two_hand_geometry(hands):
    if len(hands) < 2:
        return None
    left, right = sorted(hands[:2], key=lambda item: item["center"]["x"])
    span = max(hand["bounds"]["x2"] for hand in hands[:2]) - min(hand["bounds"]["x1"] for hand in hands[:2])
    crossing = horizontal_overlap(left["bounds"], right["bounds"]) and abs(left["center"]["y"] - right["center"]["y"]) < 0.18
    return {
        "palm_distance": point_dist(left["center"], right["center"]),
        "relative_height": left["center"]["y"] - right["center"]["y"],
        "overlap": boxes_overlap(left["bounds"], right["bounds"]),
        "hands_touching": point_dist(left["center"], right["center"]) < 0.18,
        "horizontal_crossing": crossing,
        "span": span,
        "symmetry": 1 - min(1, abs(left["center"]["y"] - right["center"]["y"]) + abs(width(left["bounds"]) - width(right["bounds"]))),
    }


def face_box(face_result, width, height):
    if not face_result.detections:
        return None
    box = face_result.detections[0].location_data.relative_bounding_box
    return {
        "x1": box.xmin,
        "y1": box.ymin,
        "x2": box.xmin + box.width,
        "y2": box.ymin + box.height,
        "width": box.width,
        "height": box.height,
        "center": {"x": box.xmin + box.width / 2, "y": box.ymin + box.height / 2},
    }


def body_anchors(face):
    if not face:
        return {
            "forehead": {"x": 0.5, "y": 0.24},
            "mouth": {"x": 0.5, "y": 0.36},
            "chin": {"x": 0.5, "y": 0.42},
            "chest": {"x": 0.5, "y": 0.62},
            "torso": {"x": 0.5, "y": 0.76},
        }
    height = face["height"]
    return {
        "forehead": {"x": face["center"]["x"], "y": face["y1"] + height * 0.22},
        "mouth": {"x": face["center"]["x"], "y": face["y1"] + height * 0.68},
        "chin": {"x": face["center"]["x"], "y": face["y2"]},
        "chest": {"x": face["center"]["x"], "y": min(1, face["y2"] + height * 1.35)},
        "torso": {"x": face["center"]["x"], "y": min(1, face["y2"] + height * 2.25)},
    }


def body_distances(center, face):
    return {name: point_dist(center, point) for name, point in body_anchors(face).items()}


def body_region(center, face):
    distances = body_distances(center, face)
    return min(distances.items(), key=lambda item: item[1])[0]


def build_profiles(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    profiles = {}
    for label in LABELS:
        items = grouped[label]
        success = [row for row in items if row.get("mediapipe_success")]
        hand_counts = Counter(str(min(2, row.get("hands_detected", 0))) for row in items)
        dominant_count = int(hand_counts.most_common(1)[0][0]) if hand_counts else 0
        required_hands = 2 if hand_counts.get("2", 0) / max(1, len(items)) >= 0.62 else 1
        all_hands = [hand for row in success for hand in row.get("hands", [])]
        two_hands = [row["two_hand_geometry"] for row in success if row.get("two_hand_geometry")]
        profiles[label] = {
            "samples": len(items),
            "mediapipe_success": len(success),
            "mediapipe_success_rate": len(success) / max(1, len(items)),
            "hand_count_distribution": dict(hand_counts),
            "dominant_hand_count": dominant_count,
            "required_hands": required_hands,
            "dominant_body_regions": [name for name, _count in Counter(hand.get("body_region") for hand in all_hands).most_common()],
            "body_distance_median": median_dict([hand.get("body_distances", {}) for hand in all_hands]),
            "finger_state_rate": bool_rate_dict([hand.get("finger_states", {}) for hand in all_hands]),
            "hand_geometry_median": median_dict([hand.get("geometry", {}) for hand in all_hands]),
            "two_hand_geometry_median": median_dict(two_hands),
            "two_hand_boolean_rate": bool_rate_dict(two_hands),
            "face_context_rate": sum(1 for row in success if row.get("face_detected")) / max(1, len(success)),
            "profile_source": "derived_from_mediapipe_terbisa_images",
        }
    return profiles


def median_dict(items):
    values = defaultdict(list)
    for item in items:
        for key, value in (item or {}).items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[key].append(float(value))
    return {key: median(vals) for key, vals in values.items() if vals}


def bool_rate_dict(items):
    values = defaultdict(list)
    for item in items:
        for key, value in (item or {}).items():
            if isinstance(value, bool):
                values[key].append(1.0 if value else 0.0)
    return {key: sum(vals) / len(vals) for key, vals in values.items() if vals}


def width(bounds):
    return bounds["x2"] - bounds["x1"]


def horizontal_overlap(a, b):
    return a["x1"] < b["x2"] and a["x2"] > b["x1"]


def boxes_overlap(a, b):
    return horizontal_overlap(a, b) and a["y1"] < b["y2"] and a["y2"] > b["y1"]


def dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def point_dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


if __name__ == "__main__":
    main()
