import os
import torch
from dotenv import load_dotenv

load_dotenv()

STT_MODE = os.getenv("STT_MODE", "local").lower()

local_whisper_model = None
openai_client = None

if STT_MODE == "local":
    try:
        from faster_whisper import WhisperModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[STT] Initializing Local Faster-Whisper on {device.upper()}...")
        local_whisper_model = WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")
    except Exception as e:
        print(f"[STT] Failed to load local model, falling back to CPU or mock: {e}")

elif STT_MODE == "openai":
    try:
        from openai import OpenAI
        print("[STT] Initializing OpenAI Cloud API...")
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        print(f"[STT] Failed to initialize OpenAI client: {e}")


def transcribe_audio_file(file_path: str) -> str:
    """
    Transcribes audio dynamically depending on the configured STT_MODE.
    """
    if not os.path.exists(file_path):
        return ""

    try:
        if STT_MODE == "local" and local_whisper_model:
            segments, _ = local_whisper_model.transcribe(file_path, language="id", beam_size=1)
            return " ".join([segment.text for segment in segments]).strip()

        elif STT_MODE == "openai" and openai_client:
            with open(file_path, "rb") as audio_file:
                response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="id"
                )
            return response.text.strip()
            
        else:
            print("[STT] Error: No valid STT engine initialized.")
            return ""

    except Exception as e:
        print(f"[STT] Transcription Error ({STT_MODE} mode): {e}")
        return ""