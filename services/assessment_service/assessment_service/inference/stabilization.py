from __future__ import annotations

from collections import Counter, defaultdict


class PredictionStabilizer:
    def __init__(
        self,
        window=4,
        stable_count=4,
        release_window=3,
        min_average_confidence=0.25,
        min_stable_duration_ms=0,
        high_confidence_threshold=1.1,
        high_confidence_window=5,
        high_confidence_count=3,
        high_confidence_stable_ms=300,
        normal_confidence_window=4,
        normal_confidence_count=4,
        normal_confidence_stable_ms=0,
    ):
        self.window = window
        self.stable_count = stable_count
        self.release_window = release_window
        self.min_average_confidence = min_average_confidence
        self.min_stable_duration_ms = min_stable_duration_ms
        self.high_confidence_threshold = high_confidence_threshold
        self.high_confidence_window = high_confidence_window
        self.high_confidence_count = high_confidence_count
        self.high_confidence_stable_ms = high_confidence_stable_ms
        self.normal_confidence_window = normal_confidence_window
        self.normal_confidence_count = normal_confidence_count
        self.normal_confidence_stable_ms = normal_confidence_stable_ms
        self.recent = []
        self.locked_label = None
        self.release_misses = 0
        self.state = "IDLE"
        self.candidate_label = None
        self.candidate_since_ms = None

    def evaluate(self, label, confidence, timestamp_ms=None, probabilities=None):
        timestamp_ms = int(timestamp_ms or 0)
        if not label:
            self.recent = []
            self.candidate_label = None
            self.candidate_since_ms = None
            if self.locked_label:
                self.state = "WAITING_FOR_RELEASE"
                self.release_misses += 1
                if self.release_misses >= self.release_window:
                    self.locked_label = None
                    self.release_misses = 0
                    self.state = "IDLE"
            else:
                self.state = "IDLE"
            return self._state(False, False, False, None, None, rejection_reason="no_hand")

        appended_under_lock = False
        if self.locked_label:
            if label != self.locked_label:
                if self.candidate_label != label:
                    self.candidate_label = label
                    self.candidate_since_ms = timestamp_ms
                self.release_misses += 1
                self.recent.append({"label": label, "confidence": confidence, "probabilities": probabilities or {}})
                self.recent = self.recent[-self.window :]
                appended_under_lock = True
                if self.release_misses < max(2, self.release_window - 1):
                    self.state = "WAITING_FOR_RELEASE"
                    return self._state(False, False, True, label, confidence, rejection_reason="waiting_for_release")
                released_label = self.locked_label
                self.recent = [item for item in self.recent if item["label"] != released_label]
                self.locked_label = None
                self.release_misses = 0
                self.state = "IDLE"
            else:
                self.release_misses = 0
                self.state = "WAITING_FOR_RELEASE"
                return self._state(True, False, True, label, confidence, rejection_reason="locked_same_word")

        if not appended_under_lock:
            self.recent.append({"label": label, "confidence": confidence, "probabilities": probabilities or {}})
            self.recent = self.recent[-self.window :]
        stable_label, average_confidence, rule = self._stable_label(timestamp_ms=timestamp_ms)
        if not stable_label:
            self.state = "CANDIDATE"
            if self.candidate_label != label:
                self.candidate_label = label
                self.candidate_since_ms = timestamp_ms
            fallback_rule = self._rule_for(label, confidence, timestamp_ms=timestamp_ms)
            return self._state(
                False,
                False,
                False,
                self.candidate_label,
                confidence,
                rejection_reason=self._history_rejection_reason(label, fallback_rule["window"]),
                agreement_count=self._agreement_count(label, fallback_rule["window"]),
                required_count=fallback_rule["count"],
                required_window=fallback_rule["window"],
                required_duration_ms=fallback_rule["stable_ms"],
                stable_duration_ms=0,
            )

        if average_confidence < self.min_average_confidence:
            self.state = "CANDIDATE"
            return self._state(
                False,
                False,
                False,
                stable_label,
                average_confidence,
                rejection_reason="below_threshold",
                agreement_count=rule["agreement_count"],
                required_count=rule["count"],
                required_window=rule["window"],
                required_duration_ms=rule["stable_ms"],
                stable_duration_ms=0,
            )

        if self.candidate_label != stable_label:
            self.candidate_label = stable_label
            self.candidate_since_ms = timestamp_ms
            self.state = "CANDIDATE"
            if timestamp_ms:
                return self._state(
                    True,
                    False,
                    False,
                    stable_label,
                    average_confidence,
                    rejection_reason="insufficient_duration",
                    agreement_count=rule["agreement_count"],
                    required_count=rule["count"],
                    required_window=rule["window"],
                    required_duration_ms=rule["stable_ms"],
                    stable_duration_ms=0,
                )

        stable_duration = timestamp_ms - (self.candidate_since_ms or timestamp_ms)
        self.state = "CANDIDATE"
        if timestamp_ms and stable_duration < rule["stable_ms"]:
            return self._state(
                True,
                False,
                False,
                stable_label,
                average_confidence,
                rejection_reason="insufficient_duration",
                agreement_count=rule["agreement_count"],
                required_count=rule["count"],
                required_window=rule["window"],
                required_duration_ms=rule["stable_ms"],
                stable_duration_ms=stable_duration,
            )

        self.locked_label = stable_label
        self.release_misses = 0
        self.state = "ACCEPTED"
        return self._state(
            True,
            True,
            False,
            stable_label,
            average_confidence,
            rejection_reason=None,
            agreement_count=rule["agreement_count"],
            required_count=rule["count"],
            required_window=rule["window"],
            required_duration_ms=rule["stable_ms"],
            stable_duration_ms=stable_duration,
        )

    def accept(self, label, confidence, timestamp_ms=None):
        return self.evaluate(label, confidence, timestamp_ms=timestamp_ms)["accepted"]

    def _stable_label(self, timestamp_ms=None):
        if not self.recent:
            return None, None, self._base_rule(timestamp_ms=timestamp_ms)
        label_counts = Counter(item["label"] for item in self.recent)
        candidates = [label for label, _count in label_counts.most_common()]
        fallback_rule = self._base_rule(timestamp_ms=timestamp_ms)
        for label in candidates:
            confidence = self._average_confidence(label, self.window)
            rule = self._rule_for(label, confidence, timestamp_ms=timestamp_ms)
            agreement_count = self._agreement_count(label, rule["window"])
            rule = {**rule, "agreement_count": agreement_count}
            if agreement_count < rule["count"]:
                fallback_rule = rule
                continue
            average_confidence = self._average_confidence(label, rule["window"])
            return label, average_confidence, rule
        return None, None, fallback_rule

    def _average_confidence(self, label, window):
        scores = defaultdict(list)
        for item in self.recent[-window:]:
            probabilities = item.get("probabilities", {})
            if label in probabilities:
                scores[label].append(float(probabilities[label]))
            elif item["label"] == label:
                scores[label].append(float(item["confidence"]))
        values = scores.get(label) or []
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _agreement_count(self, label, window):
        return len([item for item in self.recent[-window:] if item["label"] == label])

    def _history_rejection_reason(self, label, window):
        labels = [item["label"] for item in self.recent[-window:]]
        if len(set(labels)) > 1 and self._agreement_count(label, window) < self.stable_count:
            return "conflicting_history"
        return "insufficient_agreement"

    def _base_rule(self, timestamp_ms=None):
        if not timestamp_ms:
            return {"window": self.window, "count": self.stable_count, "stable_ms": self.min_stable_duration_ms, "mode": "static"}
        return {"window": self.normal_confidence_window, "count": self.normal_confidence_count, "stable_ms": self.normal_confidence_stable_ms, "mode": "normal"}

    def _rule_for(self, label, confidence, timestamp_ms=None):
        if not timestamp_ms:
            return self._base_rule(timestamp_ms=timestamp_ms)
        if confidence >= self.high_confidence_threshold:
            return {
                "window": self.high_confidence_window,
                "count": self.high_confidence_count,
                "stable_ms": self.high_confidence_stable_ms,
                "mode": "high",
            }
        return self._base_rule(timestamp_ms=timestamp_ms)

    def _state(
        self,
        stable,
        accepted,
        suppressed,
        stable_label,
        average_confidence=None,
        rejection_reason=None,
        agreement_count=0,
        required_count=None,
        required_window=None,
        required_duration_ms=None,
        stable_duration_ms=0,
    ):
        return {
            "stable": stable,
            "accepted": accepted,
            "suppressed": suppressed,
            "stable_label": stable_label,
            "average_confidence": average_confidence,
            "locked_label": self.locked_label,
            "release_misses": self.release_misses,
            "state": self.state,
            "rejection_reason": rejection_reason,
            "agreement_count": agreement_count,
            "required_count": required_count or self.stable_count,
            "required_window": required_window or self.window,
            "required_duration_ms": required_duration_ms or self.min_stable_duration_ms,
            "stable_duration_ms": stable_duration_ms,
            "history": [
                {"label": item["label"], "confidence": item["confidence"]}
                for item in self.recent[-(required_window or self.window) :]
            ],
        }
