from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import time
from typing import Any

from PIL import Image

from ..config import (
    BISINDO_YOLO_EDGE_MARGIN,
    BISINDO_YOLO_CONFIDENCE_THRESHOLD,
    BISINDO_YOLO_IMAGE_SIZE,
    BISINDO_YOLO_MAX_BOX_AREA,
    BISINDO_YOLO_MIN_BOX_AREA,
    BISINDO_YOLO_MODEL_PATH,
    BISINDO_YOLO_SETTINGS_DIR,
)


@dataclass(frozen=True)
class DetectionResult:
    status: str
    detected: bool
    class_id: int | None
    raw_label: str | None
    display_label: str | None
    confidence: float | None
    latency_ms: int | None = None
    raw_predictions: list[dict[str, Any]] | None = None
    detections: int = 0
    threshold_detections: int = 0
    valid_detections: int = 0
    bbox: dict[str, float] | None = None
    rejection_reason: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_mode: str | None = None

    @property
    def prediction(self):
        return self.display_label

    @property
    def display_text(self):
        if self.display_label:
            return self.display_label
        if self.status == "model_unavailable":
            return "Model BISINDO belum tersedia."
        if self.status == "invalid_image":
            return "Frame kamera tidak valid."
        return "Gerakan belum dikenali."

    def as_dict(self):
        return {
            "status": self.status,
            "detected": self.detected,
            "class_id": self.class_id,
            "raw_label": self.raw_label,
            "display_label": self.display_label,
            "label": self.display_label,
            "prediction": self.prediction,
            "display_text": self.display_text,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "raw_predictions": self.raw_predictions or [],
            "detections": self.detections,
            "threshold_detections": self.threshold_detections,
            "valid_detections": self.valid_detections,
            "bbox": self.bbox,
            "rejection_reason": self.rejection_reason,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_mode": self.image_mode,
        }


class BisindoYoloDetector:
    def __init__(
        self,
        model_path: Path = BISINDO_YOLO_MODEL_PATH,
        confidence_threshold: float = BISINDO_YOLO_CONFIDENCE_THRESHOLD,
        image_size: int = BISINDO_YOLO_IMAGE_SIZE,
        min_box_area: float = BISINDO_YOLO_MIN_BOX_AREA,
        max_box_area: float = BISINDO_YOLO_MAX_BOX_AREA,
        edge_margin: float = BISINDO_YOLO_EDGE_MARGIN,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.min_box_area = min_box_area
        self.max_box_area = max_box_area
        self.edge_margin = edge_margin
        self.model = None
        self.names: dict[int, str] = {}
        self.display_names: dict[int, str] = {}
        self.device = "cpu"
        self.load_error: str | None = None

    @property
    def model_available(self):
        return self.model is not None

    @property
    def class_count(self):
        return len(self.names)

    def load(self):
        if self.model is not None or self.load_error:
            return self.model
        BISINDO_YOLO_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(BISINDO_YOLO_SETTINGS_DIR))
        os.environ.setdefault("MPLCONFIGDIR", str(BISINDO_YOLO_SETTINGS_DIR / "matplotlib"))
        if not self.model_path.exists():
            self.load_error = f"Model file not found: {self.model_path}"
            print(f"[ISYARA AI] BISINDO YOLO model missing: {self.model_path}")
            return None
        try:
            from ultralytics import YOLO

            self.device = self._preferred_device()
            self.model = YOLO(str(self.model_path))
            self.names = {int(key): str(value) for key, value in self.model.names.items()}
            self.display_names = {class_id: clean_bisindo_label(label) for class_id, label in self.names.items()}
            print("[ISYARA AI] Loaded BISINDO model")
            print(f"[ISYARA AI] Number of classes: {len(self.names)}")
            print(f"[ISYARA AI] Classes: {self.names}")
            print(f"[ISYARA AI] Device: {self.device}")
        except Exception as exc:  # pragma: no cover - defensive startup logging
            self.load_error = str(exc)
            self.model = None
            self.names = {}
            self.display_names = {}
            print(f"[ISYARA AI] Failed to load BISINDO YOLO model: {exc}")
        return self.model

    def model_info(self):
        self.load()
        return {
            "status": "ready" if self.model_available else "model_unavailable",
            "model": self.model_path.name,
            "task": "BISINDO word detection",
            "model_available": self.model_available,
            "model_path": str(self.model_path),
            "class_count": self.class_count,
            "classes": [self.names[index] for index in sorted(self.names)],
            "display_classes": [self.display_names[index] for index in sorted(self.display_names)],
            "model_names": self.names,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "image_size": self.image_size,
            "min_box_area": self.min_box_area,
            "max_box_area": self.max_box_area,
            "edge_margin": self.edge_margin,
            "error": self.load_error,
        }

    def predict(self, image_bytes: bytes) -> DetectionResult:
        self.load()
        if not self.model_available:
            return DetectionResult(
                status="model_unavailable",
                detected=False,
                class_id=None,
                raw_label=None,
                display_label=None,
                confidence=None,
            )
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return DetectionResult(
                status="invalid_image",
                detected=False,
                class_id=None,
                raw_label=None,
                display_label=None,
                confidence=None,
            )

        image_width, image_height = image.size
        started_at = time.perf_counter()
        results = self.model.predict(
            image,
            conf=0.001,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        boxes = results[0].boxes if results else None
        raw_predictions = self._raw_predictions(boxes)
        detection_count = len(boxes) if boxes else 0
        threshold_detections = sum(1 for item in raw_predictions if item["confidence"] >= self.confidence_threshold)
        valid_candidates = [item for item in self._all_predictions(boxes, image_width, image_height) if item["valid"]]
        if not boxes or len(boxes) == 0:
            return DetectionResult(
                status="no_detection",
                detected=False,
                class_id=None,
                raw_label=None,
                display_label=None,
                confidence=None,
                latency_ms=latency_ms,
                raw_predictions=[],
                detections=0,
                threshold_detections=0,
                valid_detections=0,
                rejection_reason="no_detections",
                image_width=image_width,
                image_height=image_height,
                image_mode=image.mode,
            )

        if not valid_candidates:
            best_raw = raw_predictions[0] if raw_predictions else {}
            return DetectionResult(
                status="invalid_bbox",
                detected=False,
                class_id=None,
                raw_label=None,
                display_label=None,
                confidence=best_raw.get("confidence"),
                latency_ms=latency_ms,
                raw_predictions=raw_predictions,
                detections=detection_count,
                threshold_detections=threshold_detections,
                valid_detections=0,
                rejection_reason=best_raw.get("rejection_reason", "no_valid_bbox"),
                image_width=image_width,
                image_height=image_height,
                image_mode=image.mode,
            )

        best = valid_candidates[0]
        confidence = best["confidence"]
        if confidence < self.confidence_threshold:
            return DetectionResult(
                status="low_confidence",
                detected=False,
                class_id=None,
                raw_label=None,
                display_label=None,
                confidence=confidence,
                latency_ms=latency_ms,
                raw_predictions=raw_predictions,
                detections=detection_count,
                threshold_detections=threshold_detections,
                valid_detections=len(valid_candidates),
                rejection_reason="below_confidence_threshold",
                image_width=image_width,
                image_height=image_height,
                image_mode=image.mode,
            )
        class_id = best["class_id"]
        raw_label = best["raw_label"]
        display_label = best["label"]
        return DetectionResult(
            status="ok",
            detected=True,
            class_id=class_id,
            raw_label=raw_label,
            display_label=display_label,
            confidence=confidence,
            latency_ms=latency_ms,
            raw_predictions=raw_predictions,
            detections=detection_count,
            threshold_detections=threshold_detections,
            valid_detections=len(valid_candidates),
            bbox=best["bbox"],
            image_width=image_width,
            image_height=image_height,
            image_mode=image.mode,
        )

    def _raw_predictions(self, boxes, limit=3):
        if not boxes or len(boxes) == 0:
            return []
        items = []
        confidences = boxes.conf.tolist()
        classes = boxes.cls.tolist()
        for class_id, confidence in zip(classes, confidences, strict=False):
            class_id = int(class_id)
            raw_label = self.names.get(class_id, str(class_id))
            items.append(
                {
                    "class_id": class_id,
                    "raw_label": raw_label,
                    "label": self.display_names.get(class_id, clean_bisindo_label(raw_label)),
                    "confidence": float(confidence),
                }
            )
        return sorted(items, key=lambda item: item["confidence"], reverse=True)[:limit]

    def _all_predictions(self, boxes, image_width, image_height):
        if not boxes or len(boxes) == 0:
            return []
        items = []
        for class_id, confidence, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=False):
            class_id = int(class_id)
            raw_label = self.names.get(class_id, str(class_id))
            bbox = {
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
            }
            valid, reason = self._bbox_is_plausible(bbox, image_width, image_height)
            items.append(
                {
                    "class_id": class_id,
                    "raw_label": raw_label,
                    "label": self.display_names.get(class_id, clean_bisindo_label(raw_label)),
                    "confidence": float(confidence),
                    "bbox": bbox,
                    "valid": valid,
                    "rejection_reason": reason,
                }
            )
        return sorted(items, key=lambda item: item["confidence"], reverse=True)

    def _bbox_is_plausible(self, bbox, image_width, image_height):
        box_width = max(0, bbox["x2"] - bbox["x1"])
        box_height = max(0, bbox["y2"] - bbox["y1"])
        area_ratio = (box_width * box_height) / max(1, image_width * image_height)
        if area_ratio < self.min_box_area:
            return False, "bbox_too_small"
        if area_ratio > self.max_box_area:
            return False, "bbox_too_large"
        margin_x = image_width * self.edge_margin
        margin_y = image_height * self.edge_margin
        clipped_edges = sum(
            [
                bbox["x1"] <= margin_x,
                bbox["y1"] <= margin_y,
                bbox["x2"] >= image_width - margin_x,
                bbox["y2"] >= image_height - margin_y,
            ]
        )
        if clipped_edges >= 2:
            return False, "bbox_heavily_clipped"
        return True, None

    def _preferred_device(self):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"


def clean_bisindo_label(raw_label: str):
    label = raw_label.replace("-BISINDO-", "").strip()
    label = " ".join(label.split())
    if label == "Mau-Ingin":
        return "Mau"
    return label
