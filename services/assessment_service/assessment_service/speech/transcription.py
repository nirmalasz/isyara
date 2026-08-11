from ..config import WHISPER_MODEL


class SpeechTranscriptionService:
    def __init__(self):
        self.model = None
        self.available = False
        try:
            import whisper
        except ImportError:
            return
        try:
            self.model = whisper.load_model(WHISPER_MODEL)
            self.available = True
        except Exception:
            self.available = False

    def transcribe(self, audio_path):
        if not self.available:
            return {
                "status": "transcription_unavailable",
                "language": "id",
                "text": "",
                "message": "Model transkripsi sedang disiapkan.",
            }
        result = self.model.transcribe(str(audio_path), language="id")
        return {"status": "ok", "language": "id", "text": result.get("text", "").strip()}
