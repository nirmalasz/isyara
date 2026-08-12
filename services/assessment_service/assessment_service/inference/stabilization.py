from __future__ import annotations


class PredictionStabilizer:
    def __init__(self, window=5, stable_count=4, release_window=6, min_average_confidence=0.75):
        self.window = window
        self.stable_count = stable_count
        self.release_window = release_window
        self.min_average_confidence = min_average_confidence
        self.recent = []
        self.locked_label = None
        self.release_misses = 0

    def evaluate(self, label, confidence):
        if not label:
            self.recent = []
            if self.locked_label:
                self.release_misses += 1
                if self.release_misses >= self.release_window:
                    self.locked_label = None
                    self.release_misses = 0
            return {
                "stable": False,
                "accepted": False,
                "suppressed": False,
                "stable_label": None,
                "average_confidence": None,
                "locked_label": self.locked_label,
                "release_misses": self.release_misses,
            }

        self.release_misses = 0
        self.recent.append({"label": label, "confidence": confidence})
        self.recent = self.recent[-self.window :]
        matches = [item for item in self.recent if item["label"] == label]
        if len(matches) < self.stable_count:
            return self._state(False, False, False, None)

        average_confidence = sum(item["confidence"] for item in matches) / len(matches)
        if average_confidence < self.min_average_confidence:
            return self._state(False, False, False, None, average_confidence)

        if self.locked_label == label:
            return self._state(True, False, True, label, average_confidence)

        self.locked_label = label
        return self._state(True, True, False, label, average_confidence)

    def accept(self, label, confidence, timestamp_ms=None):
        return self.evaluate(label, confidence)["accepted"]

    def _state(self, stable, accepted, suppressed, stable_label, average_confidence=None):
        return {
            "stable": stable,
            "accepted": accepted,
            "suppressed": suppressed,
            "stable_label": stable_label,
            "average_confidence": average_confidence,
            "locked_label": self.locked_label,
            "release_misses": self.release_misses,
        }
