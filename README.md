# ISYARA

Real-Time BISINDO AI Translator.

ISYARA is being refactored from a learning-first MVP into a two-way communication bridge:

- BISINDO signs → Indonesian text → Indonesian voice
- Indonesian speech → Indonesian text

The current MVP is intentionally limited. It does not claim unrestricted BISINDO translation, sentence-level grammar translation, or perfect facial-expression interpretation.

## Architecture

- `isyara/`: Django project configuration.
- `apps/core/`: shared styling, security middleware, and browser-side translator scripts.
- `apps/learning/`: existing Django app, now hosting auth/profile/translator/history routes. Legacy learning models/routes are preserved for migration compatibility but are no longer primary UX.
- `services/assessment_service/`: FastAPI AI service for CV inference, sign classification, and speech transcription.
- `datasets/bisindo/`: local-only dataset layout placeholder.
- `scripts/`: offline dataset preparation, training, and evaluation scaffolds.

The Django app uses PostgreSQL when `DATABASE_URL` is set, with SQLite fallback for local demos.

## Translator Flow

BISINDO → Spoken Indonesian:

Camera → MediaPipe visual overlay → sampled camera frame → FastAPI `/predict-sign` → YOLO11 BISINDO detection → Indonesian text → Browser Text-to-Speech

Spoken Indonesian → Text:

Microphone → FastAPI `/transcribe` → Whisper interface → Indonesian Transcript → Screen

## Current AI Status

Active YOLO label map:

`Anda`, `Apa`, `Berhenti`, `Bodoh`, `Cantik`, `Halo`, `Hati-hati`, `Lelah`, `Maaf`, `Makan`, `Mau`, `Membaca`, `Nama`, `Sama-sama`, `Saya`, `Siapa`, `Sombong`, `Takut`, `Terima kasih`

The active word-level model is expected at `models/bisindo_words.pt` and is loaded with:

```python
from ultralytics import YOLO
model = YOLO("models/bisindo_words.pt")
```

The older alphabet model remains at `models/bisindo_yolo11.pt` for a future fingerspelling mode.

If the model does not exist or cannot load, `/predict-sign` returns:

```json
{
  "status": "model_unavailable",
  "detected": false,
  "class_id": null,
  "raw_label": null,
  "display_label": null,
  "label": null,
  "prediction": null,
  "display_text": "Model BISINDO belum tersedia.",
  "confidence": null
}
```

This is intentional. ISYARA does not fake sign predictions.

FastAPI also exposes `GET /model-info` with class names, class count, device, confidence threshold, and model availability.

Whisper is also optional. If unavailable, transcription returns an explicit unavailable state instead of fake text.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_bisindo
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/translate/
```

FastAPI AI service:

```bash
uvicorn services.assessment_service.assessment_service.main:app --reload --port 8001
```

Set `ASSESSMENT_SERVICE_URL=http://localhost:8001` so Django calls the AI service.

## Offline ML Commands

```bash
python scripts/prepare_dataset.py
python scripts/train_sign_classifier.py
python scripts/evaluate_sign_classifier.py
```

These are scaffolds. Training does not run during web startup.

## Auth and Security

Email/password login is available at `/accounts/login/`, signup at `/accounts/signup/`, and Google OAuth can be enabled with `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`. Translator, history, and profile routes require authentication.

For production, set `DJANGO_DEBUG=0`, provide `DJANGO_SECRET_KEY`, configure `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`, `CORS_ALLOWED_ORIGINS`, enable secure cookies/HSTS behind HTTPS, and keep `FASTAPI_ALLOWED_ORIGINS` limited to the Django origins that should call the AI service.
