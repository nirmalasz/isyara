from __future__ import annotations

import json
from pathlib import Path
from typing import Any


METADATA_PATH = Path(__file__).with_name("terbisa_structure_metadata.json")


class TerbisaStructureRecognizer:
    def __init__(self, path: Path = METADATA_PATH):
        self.path = path
        self.data = self._load()
        self.classes = self.data.get("classes", {})

    def report(self):
        return self.data

    def class_metadata(self, label: str):
        return self.classes.get(self._key(label), {})

    def apply(self, predictions: list[dict[str, Any]], structure: dict[str, Any] | None = None, hands_detected: int | None = None):
        structure = structure or {}
        observed_hands = self._observed_hands(structure, hands_detected)
        hand_features = structure.get("hands") if isinstance(structure.get("hands"), list) else []
        region_votes = self._region_votes(hand_features)
        filtered = []
        masked = []
        for item in predictions:
            label = item.get("label") or item.get("raw_label")
            meta = self.class_metadata(str(label))
            compatibility, reason = self.compatibility(meta, observed_hands, region_votes)
            score = float(item.get("calibrated_confidence", item.get("confidence", 0)) or 0)
            fused_score = score * compatibility
            enriched = {
                **item,
                "image_calibrated_confidence": score,
                "calibrated_confidence": fused_score,
                "structural_compatibility": compatibility,
                "fused_confidence": fused_score,
                "structural_rejection_reason": reason,
                "required_hands": meta.get("required_hands"),
                "recommended_roi": meta.get("recommended_roi", []),
            }
            if compatibility <= 0:
                masked.append(self._key(str(label)))
            filtered.append(enriched)
        filtered.sort(key=lambda item: item.get("fused_confidence", 0), reverse=True)
        return {
            "predictions": filtered,
            "masked_classes": sorted(set(masked)),
            "eligible_classes": self.eligible_classes(observed_hands),
            "observed_hands": observed_hands,
            "region_votes": region_votes,
        }

    def compatibility(self, meta: dict[str, Any], observed_hands: int | None, region_votes: set[str]):
        if not meta:
            return 1.0, None
        required = meta.get("required_hands")
        if observed_hands == 0:
            return 0.0, "no_hand"
        if isinstance(required, int) and observed_hands is not None and observed_hands < required:
            return 0.0, "insufficient_hands"
        regions = set(meta.get("body_regions", []))
        if regions and region_votes and regions.isdisjoint(region_votes):
            return 0.82, "weak_body_region_match"
        return 1.0, None

    def eligible_classes(self, observed_hands: int | None):
        if observed_hands is None:
            return sorted(self.classes.keys())
        if observed_hands <= 0:
            return []
        eligible = []
        for label, meta in self.classes.items():
            required = meta.get("required_hands")
            if not isinstance(required, int) or observed_hands >= required:
                eligible.append(label)
        return sorted(eligible)

    def _observed_hands(self, structure: dict[str, Any], hands_detected: int | None):
        if hands_detected is not None:
            return int(hands_detected)
        value = structure.get("hands_detected")
        if value is not None:
            return int(value)
        hands = structure.get("hands")
        if isinstance(hands, list):
            return len(hands)
        return None

    def _region_votes(self, hands: list[dict[str, Any]]):
        votes: set[str] = set()
        for hand in hands:
            region = hand.get("body_region")
            if region:
                votes.add(str(region))
            for region_name, distance in (hand.get("body_distances") or {}).items():
                try:
                    if float(distance) < 0.22:
                        votes.add(str(region_name).replace("to_", ""))
                except (TypeError, ValueError):
                    continue
        return votes

    def _load(self):
        if not self.path.exists():
            return {"source": {}, "classes": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _key(self, label: str):
        normalized = str(label or "").strip()
        normalized = normalized.replace(" -BISINDO-", "")
        normalized = normalized.replace("-BISINDO-", "")
        if normalized in {"Terima kasih", "Terima-kasih"}:
            return "Terima-kasih"
        return normalized
