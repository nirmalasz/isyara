from __future__ import annotations


class TemporalProbabilitySmoother:
    def __init__(self, beta=0.25, switch_margin=0.10, switch_confirm_frames=2, stationary_beta_scale=0.65):
        self.beta = beta
        self.switch_margin = switch_margin
        self.switch_confirm_frames = switch_confirm_frames
        self.stationary_beta_scale = stationary_beta_scale
        self.probabilities: dict[str, float] = {}
        self.current_label: str | None = None
        self.switch_candidate: str | None = None
        self.switch_confirm_count = 0

    def reset(self):
        self.probabilities = {}
        self.current_label = None
        self.switch_candidate = None
        self.switch_confirm_count = 0

    def update(self, predictions: list[dict], pose_state: str | None = None):
        current = self._prediction_vector(predictions)
        raw_top1 = self._top1(current)
        if not current:
            self.reset()
            return self._payload(raw_top1=raw_top1, smoothed_top1=None)

        if not self.probabilities:
            self.probabilities = dict(current)
        else:
            beta = self.beta
            if pose_state == "stationary" and self.current_label:
                beta = self.beta * self.stationary_beta_scale

            labels = set(self.probabilities) | set(current)
            smoothed = {}
            for label in labels:
                previous = self.probabilities.get(label, 0.0)
                smoothed[label] = beta * current.get(label, 0.0) + (1 - beta) * previous
            self.probabilities = smoothed

        smoothed_top1 = self._top1(self.probabilities)
        if smoothed_top1 is None:
            self.reset()
            return self._payload(raw_top1=raw_top1, smoothed_top1=None)

        if self.current_label is None:
            self.current_label = smoothed_top1["label"]
            self.switch_candidate = None
            self.switch_confirm_count = 0
        elif smoothed_top1["label"] == self.current_label:
            self.switch_candidate = None
            self.switch_confirm_count = 0
        else:
            current_confidence = self.probabilities.get(self.current_label, 0.0)
            margin = smoothed_top1["confidence"] - current_confidence
            if margin >= self.switch_margin:
                if self.switch_candidate == smoothed_top1["label"]:
                    self.switch_confirm_count += 1
                else:
                    self.switch_candidate = smoothed_top1["label"]
                    self.switch_confirm_count = 1
                if self.switch_confirm_count >= self.switch_confirm_frames:
                    self.current_label = smoothed_top1["label"]
                    self.switch_candidate = None
                    self.switch_confirm_count = 0
            else:
                self.switch_candidate = None
                self.switch_confirm_count = 0

        selected = {
            "label": self.current_label,
            "confidence": self.probabilities.get(self.current_label, 0.0) if self.current_label else 0.0,
        }
        return self._payload(raw_top1=raw_top1, smoothed_top1=smoothed_top1, selected=selected)

    def _payload(self, raw_top1=None, smoothed_top1=None, selected=None):
        return {
            "raw_top1": raw_top1,
            "smoothed_top1": smoothed_top1,
            "smoothed_confidence": (smoothed_top1 or {}).get("confidence"),
            "current_stable_class": (selected or {}).get("label") or self.current_label,
            "current_stable_confidence": (selected or {}).get("confidence") if selected else None,
            "switch_candidate": self.switch_candidate,
            "switch_margin": self.switch_margin,
            "switch_confirm_count": self.switch_confirm_count,
            "switch_confirm_frames": self.switch_confirm_frames,
            "probabilities": dict(self.probabilities),
        }

    def _prediction_vector(self, predictions):
        vector = {}
        for item in predictions or []:
            label = item.get("label")
            if not label:
                continue
            confidence = float(item.get("calibrated_confidence", item.get("confidence", 0)) or 0)
            vector[label] = max(vector.get(label, 0.0), confidence)
        return vector

    def _top1(self, probabilities):
        if not probabilities:
            return None
        label, confidence = max(probabilities.items(), key=lambda item: item[1])
        return {"label": label, "confidence": confidence}
