from dataclasses import dataclass

from ..config import CONFIDENCE_THRESHOLD, MODEL_PATH
from .labels import display_text, load_labels
from .model import SignClassifier, torch


@dataclass
class PredictionResult:
    status: str
    prediction: str | None
    display_text: str
    confidence: float | None


class SignInferenceService:
    def __init__(self):
        self.labels = load_labels()
        self.model = None
        self.model_available = False
        self._load_model()

    def _load_model(self):
        if torch is None or not MODEL_PATH.exists():
            self.model_available = False
            return
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        input_size = int(checkpoint.get("input_size"))
        hidden_size = int(checkpoint.get("hidden_size", 128))
        labels = checkpoint.get("labels") or self.labels
        self.labels = labels
        self.model = SignClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=len(labels))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model_available = True

    def predict(self, sequence):
        if not self.model_available:
            return PredictionResult(
                status="model_unavailable",
                prediction=None,
                display_text="Model penerjemah sedang disiapkan.",
                confidence=None,
            )
        with torch.no_grad():
            tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            probabilities = torch.softmax(self.model(tensor), dim=-1)[0]
            confidence, index = torch.max(probabilities, dim=0)
            confidence_value = float(confidence.item())
            label = self.labels[int(index.item())]
        if confidence_value < CONFIDENCE_THRESHOLD:
            return PredictionResult(
                status="low_confidence",
                prediction=None,
                display_text="Gerakan belum dikenali",
                confidence=confidence_value,
            )
        return PredictionResult(
            status="ok",
            prediction=label,
            display_text=display_text(label),
            confidence=confidence_value,
        )
