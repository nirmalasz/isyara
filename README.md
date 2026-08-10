# ISYARA MVP

AI-powered BISINDO learning web application skeleton.

## Architecture

- `isyara/`: Django project configuration.
- `apps/core/`: shared template tags and presentation helpers.
- `apps/learning/`: user-facing learning flow, models, routes, seed data, and mock assessment adapter.
- `services/assessment_service/`: FastAPI service stub for future MediaPipe/OpenCV/NumPy assessment.

The Django app uses PostgreSQL when `DATABASE_URL` is set, with SQLite fallback for local demos.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_bisindo
python manage.py runserver
```

FastAPI assessment stub:

```bash
uvicorn services.assessment_service.assessment_service.main:app --reload --port 8001
```

Set `ASSESSMENT_SERVICE_URL=http://localhost:8001` to have Django call the service. Without it, Django uses the built-in mock client.

## Auth and Security

Email/password login is available at `/accounts/login/`, signup at `/accounts/signup/`, and Google OAuth can be enabled with `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`. Profile, progress, practice, and result routes require login and only expose records owned by the current user.

For production, set `DJANGO_DEBUG=0`, provide `DJANGO_SECRET_KEY`, configure `ALLOWED_HOSTS`, `CSRF_TRUSTED_ORIGINS`, `CORS_ALLOWED_ORIGINS`, enable secure cookies/HSTS behind HTTPS, and keep `FASTAPI_ALLOWED_ORIGINS` limited to the Django origins that should call the assessment service.
