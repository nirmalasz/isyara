from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any


CALIBRATION_PATH = Path(__file__).with_name("bisindo_calibration.json")


class BisindoCalibration:
    def __init__(self, path: Path = CALIBRATION_PATH):
        self.path = path
        self.data = self._load()
        self.global_threshold = float(self.data.get("global_threshold", 0.75))
        self.min_margin = float(self.data.get("min_margin", 0.12))
        self.classes = self.data.get("classes", {})

    def threshold_for(self, label: str):
        return float(self.classes.get(self._key(label), {}).get("threshold", self.global_threshold))

    def prior_for(self, label: str):
        return float(self.classes.get(self._key(label), {}).get("prior_correction", 1.0))

    def calibrate_predictions(self, raw_predictions: list[dict[str, Any]]):
        calibrated = []
        for item in raw_predictions:
            label = item.get("label") or item.get("raw_label")
            confidence = float(item.get("confidence") or 0)
            prior = self.prior_for(label)
            calibrated_score = min(1.0, confidence * prior)
            calibrated.append(
                {
                    **item,
                    "confidence": confidence,
                    "calibrated_confidence": calibrated_score,
                    "class_threshold": self.threshold_for(label),
                    "prior_correction": prior,
                }
            )
        calibrated.sort(key=lambda item: item["calibrated_confidence"], reverse=True)
        return calibrated

    def aggregate_candidate_predictions(self, candidate_payloads: list[dict[str, Any]]):
        by_label: dict[str, list[float]] = {}
        exemplar: dict[str, dict[str, Any]] = {}
        for payload in candidate_payloads:
            for item in payload.get("raw_predictions", []):
                label = item.get("label") or item.get("raw_label")
                if not label:
                    continue
                by_label.setdefault(label, []).append(float(item.get("confidence") or 0))
                if label not in exemplar or float(item.get("confidence") or 0) > float(exemplar[label].get("confidence") or 0):
                    exemplar[label] = item
        aggregated = []
        for label, values in by_label.items():
            mean_score = sum(values) / len(values)
            robust_score = (mean_score * 0.7) + (median(values) * 0.3)
            base = exemplar[label]
            aggregated.append({**base, "confidence": robust_score, "roi_scores": values})
        return self.calibrate_predictions(aggregated)

    def decision(self, calibrated_predictions: list[dict[str, Any]]):
        top = calibrated_predictions[0] if calibrated_predictions else None
        second = calibrated_predictions[1] if len(calibrated_predictions) > 1 else None
        if not top:
            return None, "no_valid_prediction", 0.0
        margin = top["calibrated_confidence"] - (second["calibrated_confidence"] if second else 0.0)
        if top["calibrated_confidence"] < top["class_threshold"]:
            return top, "below_class_threshold", margin
        if margin < self.min_margin:
            return top, "low_margin", margin
        return top, None, margin

    def report(self):
        return self.data

    def _load(self):
        if not self.path.exists():
            return {"global_threshold": 0.75, "min_margin": 0.12, "classes": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _key(self, label: str):
        normalized = str(label or "").strip()
        normalized = normalized.replace(" -BISINDO-", "")
        normalized = normalized.replace("-BISINDO-", "")
        if normalized in {"Terima kasih", "Terima-kasih"}:
            return "Terima-kasih"
        return normalized
