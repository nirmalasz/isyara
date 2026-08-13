from __future__ import annotations

import json
import math
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
        two_hand_geometry = structure.get("two_hand_geometry") if isinstance(structure.get("two_hand_geometry"), dict) else {}
        region_votes = self._region_votes(hand_features)
        filtered = []
        masked = []
        for item in predictions:
            label = item.get("label") or item.get("raw_label")
            meta = self.class_metadata(str(label))
            compatibility_payload = self.compatibility(meta, observed_hands, region_votes, hand_features, two_hand_geometry)
            compatibility = compatibility_payload["compatibility"]
            reason = compatibility_payload["reason"]
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
                "geometry_compatibility": compatibility_payload,
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

    def compatibility(
        self,
        meta: dict[str, Any],
        observed_hands: int | None,
        region_votes: set[str],
        hands: list[dict[str, Any]] | None = None,
        two_hand_geometry: dict[str, Any] | None = None,
    ):
        if not meta:
            return self._compatibility_payload(1.0, None)
        required = meta.get("required_hands")
        if observed_hands == 0:
            return self._compatibility_payload(0.0, "no_hand")
        if isinstance(required, int) and observed_hands is not None and observed_hands < required:
            return self._compatibility_payload(0.0, "insufficient_hands")
        hands = hands or []
        two_hand_geometry = two_hand_geometry or {}
        handshape_score = self._handshape_compatibility(meta, hands)
        body_score = self._body_compatibility(meta, hands, region_votes)
        two_hand_score = self._two_hand_compatibility(meta, observed_hands, two_hand_geometry)
        compatibility = handshape_score * body_score * two_hand_score
        reason = None
        if compatibility < 0.45:
            reason = "weak_geometry_match"
        elif body_score < 0.72:
            reason = "weak_body_region_match"
        elif two_hand_score < 0.72:
            reason = "weak_two_hand_geometry_match"
        elif handshape_score < 0.72:
            reason = "weak_handshape_match"
        return self._compatibility_payload(compatibility, reason, handshape_score, body_score, two_hand_score)

    def _compatibility_payload(self, compatibility, reason, handshape=1.0, body=1.0, two_hand=1.0):
        return {
            "compatibility": round(float(compatibility), 4),
            "reason": reason,
            "handshape": round(float(handshape), 4),
            "body_location": round(float(body), 4),
            "two_hand_geometry": round(float(two_hand), 4),
        }

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

    def _body_compatibility(self, meta, hands, region_votes):
        if not hands:
            return 1.0
        derived = meta.get("body_distance_median") if isinstance(meta.get("body_distance_median"), dict) else {}
        if derived:
            best = 0.45
            for hand in hands:
                distances = hand.get("body_distances") or {}
                score_parts = []
                for key, expected in derived.items():
                    if key in distances:
                        score_parts.append(self._numeric_score(float(distances[key]), float(expected), tolerance=0.18))
                if score_parts:
                    best = max(best, sum(score_parts) / len(score_parts))
            return best
        regions = set(meta.get("body_regions", []))
        if not regions or not region_votes:
            return 1.0
        if regions & region_votes:
            return 1.0
        nearby_groups = [
            {"forehead", "head", "head_side"},
            {"mouth", "chin", "face"},
            {"chest", "shoulders", "torso", "forward_space"},
        ]
        for group in nearby_groups:
            if regions & group and region_votes & group:
                return 0.78
        return 0.55

    def _handshape_compatibility(self, meta, hands):
        if not hands:
            return 1.0
        profile_fingers = meta.get("finger_state_rate") if isinstance(meta.get("finger_state_rate"), dict) else {}
        profile_geometry = meta.get("hand_geometry_median") if isinstance(meta.get("hand_geometry_median"), dict) else {}
        if not profile_fingers and not profile_geometry:
            return self._metadata_handshape_compatibility(meta, hands)
        best = 0.45
        for hand in hands:
            scores = []
            fingers = hand.get("finger_states") or {}
            for key, expected in profile_fingers.items():
                if key in fingers:
                    observed = 1.0 if fingers[key] else 0.0
                    scores.append(1.0 - min(0.55, abs(observed - float(expected)) * 0.75))
            geometry = hand.get("geometry") or {}
            for key, expected in profile_geometry.items():
                observed = self._nested_number(geometry, key)
                if observed is not None:
                    scores.append(self._numeric_score(observed, float(expected), tolerance=0.45))
            if scores:
                best = max(best, sum(scores) / len(scores))
        return best

    def _metadata_handshape_compatibility(self, meta, hands):
        important = {str(item).lower() for item in meta.get("important_fingers", [])}
        text = f"{meta.get('handshape', '')} {meta.get('palm_orientation', '')}".lower()
        scores = []
        for hand in hands:
            fingers = hand.get("finger_states") or {}
            geometry = hand.get("geometry") or {}
            if important:
                expected = [fingers.get(name) for name in important if name in fingers]
                if expected:
                    scores.append(sum(1 for value in expected if value) / len(expected))
            open_count = sum(1 for value in fingers.values() if value)
            openness = float(geometry.get("openness", 0) or 0)
            if "open" in text or "flat" in text or "all" in important:
                if fingers:
                    scores.append(max(open_count / 5, min(1.0, openness / 6.0)))
            if "fist" in text or "closed" in text:
                if fingers:
                    scores.append(1.0 - min(0.65, open_count / 5))
            if "point" in text or "index" in important:
                if "index" in fingers:
                    scores.append(1.0 if fingers.get("index") else 0.45)
        if not scores:
            return 1.0
        return max(0.35, sum(scores) / len(scores))

    def _two_hand_compatibility(self, meta, observed_hands, two_hand_geometry):
        required = meta.get("required_hands")
        if required != 2:
            return 1.0
        if observed_hands is not None and observed_hands < 2:
            return 0.0
        if not two_hand_geometry:
            return 1.0
        profile = meta.get("two_hand_geometry_median") if isinstance(meta.get("two_hand_geometry_median"), dict) else {}
        booleans = meta.get("two_hand_boolean_rate") if isinstance(meta.get("two_hand_boolean_rate"), dict) else {}
        scores = []
        for key, expected in profile.items():
            observed = self._nested_number(two_hand_geometry, key)
            if observed is not None:
                tolerance = 0.22 if key in {"palm_distance", "span"} else 0.35
                scores.append(self._numeric_score(observed, float(expected), tolerance=tolerance))
        for key, expected in booleans.items():
            if key in two_hand_geometry:
                observed = 1.0 if two_hand_geometry[key] else 0.0
                scores.append(1.0 - min(0.65, abs(observed - float(expected)) * 0.9))
        if scores:
            return max(0.2, sum(scores) / len(scores))
        descriptor = f"{meta.get('two_hand_distance', '')} {meta.get('handshape', '')}".lower()
        fallback_scores = []
        interacts = bool(meta.get("both_hands_interact"))
        if "cross" in descriptor or "x-like" in descriptor:
            fallback_scores.append(1.0 if (two_hand_geometry.get("horizontal_crossing") or two_hand_geometry.get("handed_order_crossed")) else 0.35)
            fallback_scores.append(1.0 if two_hand_geometry.get("overlap") else 0.55)
        if "separated" in descriptor or "symmetric" in descriptor:
            fallback_scores.append(0.45 if two_hand_geometry.get("horizontal_crossing") else 1.0)
            fallback_scores.append(0.60 if two_hand_geometry.get("overlap") else 1.0)
            fallback_scores.append(self._numeric_floor(two_hand_geometry.get("palm_distance"), floor=0.20))
        if "close" in descriptor or "touch" in descriptor or "stack" in descriptor:
            fallback_scores.append(1.0 if (two_hand_geometry.get("hands_touching") or two_hand_geometry.get("overlap")) else 0.62)
        if interacts and not (two_hand_geometry.get("overlap") or two_hand_geometry.get("hands_touching") or two_hand_geometry.get("horizontal_crossing")):
            fallback_scores.append(0.72)
        if fallback_scores:
            return max(0.2, sum(fallback_scores) / len(fallback_scores))
        return 1.0

    def _numeric_floor(self, value, floor):
        try:
            return 1.0 if float(value) >= floor else 0.55
        except (TypeError, ValueError):
            return 0.8

    def _numeric_score(self, observed, expected, tolerance):
        if not math.isfinite(observed) or not math.isfinite(expected):
            return 1.0
        return max(0.2, 1.0 - min(0.8, abs(observed - expected) / max(0.001, tolerance)))

    def _nested_number(self, payload, key):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        for nested_key in ["finger_angles", "fingertip_distances", "index_vector", "middle_vector", "thumb_vector"]:
            nested = payload.get(nested_key)
            if isinstance(nested, dict) and isinstance(nested.get(key), (int, float)):
                return float(nested[key])
        return None

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
