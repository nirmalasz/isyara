from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import re
import time

from PIL import Image, ImageOps

from ..config import (
    BISINDO_CLASSIFICATION_CONFIDENCE_THRESHOLD,
    BISINDO_CLASSIFICATION_IMAGE_SIZE,
    BISINDO_CLASSIFIER_MODEL_PATH,
    BISINDO_YOLO_SETTINGS_DIR,
)
from .bisindo_detector import DetectionResult


class BisindoYoloClassifier:
    def __init__(
        self,
        model_path: Path = BISINDO_CLASSIFIER_MODEL_PATH,
        confidence_threshold: float = BISINDO_CLASSIFICATION_CONFIDENCE_THRESHOLD,
        image_size: int = BISINDO_CLASSIFICATION_IMAGE_SIZE,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.model = None
        self.names: dict[int, str] = {}
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
            print(f"[ISYARA AI] BISINDO classifier missing: {self.model_path}")
            return None
        try:
            from ultralytics import YOLO

            self.device = self._preferred_device()
            self.model = YOLO(str(self.model_path))
            self.names = {int(key): str(value) for key, value in self.model.names.items()}
            print("[ISYARA AI] Loaded BISINDO classifier")
            print(f"[ISYARA AI] Classifier classes: {self.names}")
            print(f"[ISYARA AI] Classifier device: {self.device}")
        except Exception as exc:  # pragma: no cover - defensive startup logging
            self.load_error = str(exc)
            self.model = None
            self.names = {}
            print(f"[ISYARA AI] Failed to load BISINDO classifier: {exc}")
        return self.model

    def model_info(self):
        self.load()
        return {
            "status": "ready" if self.model_available else "model_unavailable",
            "model": self.model_path.name,
            "task": "BISINDO word classification",
            "model_available": self.model_available,
            "model_path": str(self.model_path),
            "class_count": self.class_count,
            "classes": [self.names[index] for index in sorted(self.names)],
            "display_classes": [self.names[index].replace("Terima-kasih", "Terima kasih") for index in sorted(self.names)],
            "model_names": self.names,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "image_size": self.image_size,
            "error": self.load_error,
        }

    def predict(self, image_bytes: bytes, debug_context: dict | None = None) -> DetectionResult:
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
            decoded = Image.open(BytesIO(image_bytes))
            image = ImageOps.exif_transpose(decoded).convert("RGB")
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
        debug_payload = self._save_debug_input(image_bytes, image, debug_context)
        started_at = time.perf_counter()
        results = self.model.predict(image, imgsz=self.image_size, device=self.device, verbose=False)
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        probs = results[0].probs if results else None
        if probs is None:
            return DetectionResult(
                status="no_detection",
                detected=False,
                class_id=None,
                raw_label=None,
                display_label=None,
                confidence=None,
                latency_ms=latency_ms,
                raw_predictions=[],
                rejection_reason="no_probabilities",
                image_width=image_width,
                image_height=image_height,
                image_mode=image.mode,
                debug=debug_payload,
            )
        top_indices = sorted(range(len(self.names)), key=lambda index: float(probs.data[index]), reverse=True)
        raw_predictions = [
            {
                "class_id": index,
                "raw_label": self.names.get(index, str(index)),
                "label": clean_classifier_label(self.names.get(index, str(index))),
                "confidence": float(probs.data[index]),
            }
            for index in top_indices
        ]
        if debug_payload is not None:
            debug_payload.update(
                {
                    "source_size": {"width": image_width, "height": image_height},
                    "image_mode": image.mode,
                    "classifier_image_size": self.image_size,
                    "preprocessing_note": "PIL RGB image is passed directly to Ultralytics YOLO classify with imgsz=224.",
                    "top5_original": raw_predictions[:5],
                    "top5_flipped": self._predict_topk(ImageOps.mirror(image), topk=5),
                }
            )
            self._write_debug_metadata(debug_payload)
        class_id = int(probs.top1)
        confidence = float(probs.top1conf)
        raw_label = self.names.get(class_id, str(class_id))
        display_label = clean_classifier_label(raw_label)
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
                detections=1,
                threshold_detections=0,
                valid_detections=1,
                rejection_reason="below_confidence_threshold",
                image_width=image_width,
                image_height=image_height,
                image_mode=image.mode,
                debug=debug_payload,
            )
        return DetectionResult(
            status="ok",
            detected=True,
            class_id=class_id,
            raw_label=raw_label,
            display_label=display_label,
            confidence=confidence,
            latency_ms=latency_ms,
            raw_predictions=raw_predictions,
            detections=1,
            threshold_detections=1,
            valid_detections=1,
            image_width=image_width,
            image_height=image_height,
            image_mode=image.mode,
            debug=debug_payload,
        )

    def _predict_topk(self, image: Image.Image, topk=5):
        results = self.model.predict(image, imgsz=self.image_size, device=self.device, verbose=False)
        probs = results[0].probs if results else None
        if probs is None:
            return []
        top_indices = sorted(range(len(self.names)), key=lambda index: float(probs.data[index]), reverse=True)[:topk]
        return [
            {
                "class_id": index,
                "raw_label": self.names.get(index, str(index)),
                "label": clean_classifier_label(self.names.get(index, str(index))),
                "confidence": float(probs.data[index]),
            }
            for index in top_indices
        ]

    def _save_debug_input(self, image_bytes: bytes, image: Image.Image, debug_context: dict | None):
        if not debug_context:
            return None
        debug_dir = Path(debug_context.get("debug_dir") or "runs/live_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = str(debug_context.get("timestamp") or int(time.time() * 1000))
        roi_type = self._safe_name(str(debug_context.get("roi_type") or "single"))
        frame_id = self._safe_name(str(debug_context.get("frame_id") or "frame"))
        base = debug_dir / f"{timestamp}_{frame_id}_{roi_type}"
        exact_path = base.with_suffix(self._extension_for_content_type(str(debug_context.get("content_type") or "")))
        exact_path.write_bytes(image_bytes)
        decoded_path = debug_dir / f"{timestamp}_{frame_id}_{roi_type}_decoded.jpg"
        image.save(decoded_path, format="JPEG", quality=95)
        payload = {
            "exact_input_path": str(exact_path),
            "decoded_rgb_path": str(decoded_path),
            "roi_type": debug_context.get("roi_type"),
            "frame_id": debug_context.get("frame_id"),
            "full_frame_dimensions": debug_context.get("full_frame_dimensions"),
            "roi": debug_context.get("roi"),
            "hands_detected": debug_context.get("hands_detected"),
            "handedness": debug_context.get("handedness"),
            "mirrored": debug_context.get("mirrored"),
            "expected_display_orientation": debug_context.get("expected_display_orientation"),
            "content_type": debug_context.get("content_type"),
            "input_bytes": len(image_bytes),
        }
        return payload

    def _write_debug_metadata(self, payload):
        metadata_path = Path(payload["exact_input_path"]).with_suffix(".json")
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _safe_name(self, value: str):
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "item"

    def _extension_for_content_type(self, content_type: str):
        if "png" in content_type:
            return ".png"
        if "webp" in content_type:
            return ".webp"
        return ".jpg"

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


def clean_classifier_label(label: str):
    return label.replace("Terima-kasih", "Terima kasih")
