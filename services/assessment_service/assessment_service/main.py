from importlib.util import find_spec
import os

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="ISYARA Assessment Service",
    description="Computer vision and motion assessment boundary for BISINDO practice.",
    version="0.1.0",
)

allowed_origins = [
    origin
    for origin in os.getenv("FASTAPI_ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",")
    if origin
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type", "authorization"],
)


class AssessmentRequest(BaseModel):
    session_id: int
    lesson_slug: str
    sign_slug: str


class AssessmentResponse(BaseModel):
    score: int
    summary: str
    strengths: list[str]
    improvements: list[str]
    next_action: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "opencv_available": find_spec("cv2") is not None,
        "mediapipe_available": find_spec("mediapipe") is not None,
        "numpy_available": find_spec("numpy") is not None,
    }


@app.post("/assess", response_model=AssessmentResponse)
def assess(payload: AssessmentRequest):
    score = min(96, 78 + ((payload.session_id * 11) % 18))
    return AssessmentResponse(
        score=score,
        summary="Penilaian mock FastAPI selesai. Ekstraksi landmark sengaja ditunda untuk MVP.",
        strengths=[
            "Tangan tetap berada di area kamera yang diharapkan",
            "Tempo gerakan cukup konsisten",
        ],
        improvements=[
            "Skor CV berikutnya akan membandingkan landmark MediaPipe dengan gerakan referensi",
            "Tambahkan sampel frame atau stream landmark saat frontend siap",
        ],
        next_action="Gunakan kontrak respons ini saat mock diganti dengan analisis gerakan nyata.",
    )
