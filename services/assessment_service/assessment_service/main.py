from importlib.util import find_spec
import json
import os
from tempfile import NamedTemporaryFile
from typing import Annotated

from pydantic import BaseModel, Field
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import BISINDO_CLASSIFICATION_CONFIDENCE_THRESHOLD, BISINDO_INFERENCE_MODE, BISINDO_YOLO_CONFIDENCE_THRESHOLD, BISINDO_YOLO_MODEL_PATH
from .config import BISINDO_RELEASE_WINDOW, BISINDO_STABILIZATION_MATCHES, BISINDO_STABILIZATION_WINDOW
from .inference.bisindo_classifier import BisindoYoloClassifier
from .inference.bisindo_detector import BisindoYoloDetector
from .inference.stabilization import PredictionStabilizer
from .speech.transcription import SpeechTranscriptionService


DEBUG_DIAGNOSTICS = os.getenv("FASTAPI_DEBUG", os.getenv("DJANGO_DEBUG", "1")) == "1"

app = FastAPI(
    title="ISYARA AI Service",
    description="Computer vision, sign classification, and Indonesian speech transcription service for ISYARA.",
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


class PredictSignResponse(BaseModel):
    status: str
    detected: bool
    class_id: int | None = None
    raw_label: str | None = None
    display_label: str | None = None
    label: str | None = None
    prediction: str | None = None
    display_text: str
    confidence: float | None = None
    latency_ms: int | None = None
    raw_predictions: list[dict] = Field(default_factory=list)
    detections: int = 0
    threshold_detections: int = 0
    valid_detections: int = 0
    bbox: dict | None = None
    rejection_reason: str | None = None
    stable: bool = False
    suppressed: bool = False
    accepted: bool = False
    accepted_prediction: str | None = None
    accepted_confidence: float | None = None
    stable_prediction: str | None = None
    stable_confidence: float | None = None
    locked_label: str | None = None
    release_misses: int = 0
    reason: str | None = None
    roi: dict | None = None
    roi_type: str | None = None
    inference_model: str | None = None
    selected_candidate: dict | None = None
    candidate_predictions: list[dict] = Field(default_factory=list)
    hands_detected: int | None = None
    handedness: list[str] = Field(default_factory=list)
    frame_id: str | None = None
    mirrored: bool | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_mode: str | None = None


class ExtractLandmarksResponse(BaseModel):
    status: str
    message: str


bisindo_detector = BisindoYoloDetector()
bisindo_classifier = BisindoYoloClassifier()
prediction_stabilizer = PredictionStabilizer(
    window=BISINDO_STABILIZATION_WINDOW,
    stable_count=BISINDO_STABILIZATION_MATCHES,
    release_window=BISINDO_RELEASE_WINDOW,
    min_average_confidence=BISINDO_YOLO_CONFIDENCE_THRESHOLD,
)
speech_transcription = SpeechTranscriptionService()


@app.on_event("startup")
def load_bisindo_model():
    bisindo_detector.load()
    bisindo_classifier.load()


@app.get("/health")
def health():
    model_info = bisindo_detector.model_info()
    classifier_info = bisindo_classifier.model_info()
    return {
        "status": "ok",
        "opencv_available": find_spec("cv2") is not None,
        "mediapipe_available": find_spec("mediapipe") is not None,
        "numpy_available": find_spec("numpy") is not None,
        "torch_available": find_spec("torch") is not None,
        "ultralytics_available": find_spec("ultralytics") is not None,
        "whisper_available": find_spec("whisper") is not None,
        "bisindo_yolo_model_path": str(BISINDO_YOLO_MODEL_PATH),
        "bisindo_yolo_model_available": model_info["model_available"],
        "bisindo_yolo_class_count": model_info["class_count"],
        "bisindo_yolo_device": model_info["device"],
        "bisindo_classifier_model_path": classifier_info["model_path"],
        "bisindo_classifier_model_available": classifier_info["model_available"],
        "bisindo_classifier_class_count": classifier_info["class_count"],
        "bisindo_classifier_device": classifier_info["device"],
        "bisindo_inference_mode": _active_inference_mode(),
        "confidence_threshold": BISINDO_YOLO_CONFIDENCE_THRESHOLD,
        "classification_confidence_threshold": BISINDO_CLASSIFICATION_CONFIDENCE_THRESHOLD,
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


@app.post("/extract-landmarks", response_model=ExtractLandmarksResponse)
def extract_landmarks():
    return ExtractLandmarksResponse(
        status="client_side_active",
        message="Browser MediaPipe extraction is active for MVP. Server-side extraction module is scaffolded for offline training.",
    )


@app.get("/model-info")
def model_info():
    return {
        "status": "ready" if _active_predictor().model_available else "model_unavailable",
        "inference_mode": _active_inference_mode(),
        "active": _active_predictor().model_info(),
        "detector": bisindo_detector.model_info(),
        "classifier": bisindo_classifier.model_info(),
    }


@app.post("/predict-sign", response_model=PredictSignResponse)
async def predict_sign(
    image: UploadFile | None = File(None),
    candidates: list[UploadFile] | None = File(None),
    frame_id: Annotated[str | None, Form()] = None,
    mirrored: Annotated[bool | None, Form()] = None,
    roi_x1: Annotated[float | None, Form()] = None,
    roi_y1: Annotated[float | None, Form()] = None,
    roi_x2: Annotated[float | None, Form()] = None,
    roi_y2: Annotated[float | None, Form()] = None,
    source_width: Annotated[int | None, Form()] = None,
    source_height: Annotated[int | None, Form()] = None,
    hands_detected: Annotated[int | None, Form()] = None,
    handedness: Annotated[str | None, Form()] = None,
    candidates_json: Annotated[str | None, Form()] = None,
):
    candidates = candidates if isinstance(candidates, list) else None
    uploads = candidates or ([image] if image is not None else [])
    if not uploads:
        return PredictSignResponse(
            status="invalid_image",
            detected=False,
            class_id=None,
            raw_label=None,
            display_label=None,
            label=None,
            prediction=None,
            display_text="Frame kamera wajib dikirim.",
            confidence=None,
            latency_ms=None,
            frame_id=frame_id,
            mirrored=mirrored,
            reason="missing_image",
        )
    invalid_upload = next((upload for upload in uploads if not upload.content_type or upload.content_type not in {"image/jpeg", "image/webp", "image/png"}), None)
    if invalid_upload:
        return PredictSignResponse(
            status="invalid_image",
            detected=False,
            class_id=None,
            raw_label=None,
            display_label=None,
            label=None,
            prediction=None,
            display_text="Format gambar harus JPEG, WebP, atau PNG.",
            confidence=None,
            latency_ms=None,
            frame_id=frame_id,
            mirrored=mirrored,
            reason="invalid_image",
        )
    candidate_meta = _candidate_metadata(candidates_json)
    if candidates:
        result, payload, roi = await _predict_candidate_batch(uploads, candidate_meta)
    else:
        result = _active_predictor().predict(await image.read())
        payload = result.as_dict()
        roi = _roi_payload(roi_x1, roi_y1, roi_x2, roi_y2, source_width, source_height)
        if roi and payload.get("bbox"):
            payload["bbox"] = _map_roi_bbox_to_source(payload["bbox"], roi)
        payload["roi_type"] = "single"
        payload["candidate_predictions"] = []
        payload["selected_candidate"] = _selected_candidate_payload(payload, roi, "single")
    stability = prediction_stabilizer.evaluate(result.display_label, result.confidence or 0)
    payload.update(
        {
            "stable": stability["stable"],
            "suppressed": stability["suppressed"],
            "accepted": stability["accepted"],
            "accepted_prediction": result.display_label if stability["accepted"] else None,
            "accepted_confidence": stability["average_confidence"] if stability["accepted"] else None,
            "stable_prediction": stability["stable_label"],
            "stable_confidence": stability["average_confidence"],
            "locked_label": stability["locked_label"],
            "release_misses": stability["release_misses"],
            "reason": payload.get("rejection_reason"),
            "roi": roi,
            "roi_type": payload.get("roi_type"),
            "inference_model": _active_inference_mode(),
            "selected_candidate": payload.get("selected_candidate"),
            "candidate_predictions": payload.get("candidate_predictions", []),
            "hands_detected": hands_detected,
            "handedness": [item for item in (handedness or "").split(",") if item],
            "frame_id": frame_id,
            "mirrored": mirrored,
            "image_width": source_width or payload.get("image_width"),
            "image_height": source_height or payload.get("image_height"),
        }
    )
    if not DEBUG_DIAGNOSTICS:
        payload["raw_predictions"] = []
        payload["candidate_predictions"] = [
            {key: value for key, value in item.items() if key != "raw_predictions"}
            for item in payload.get("candidate_predictions", [])
        ]
    top3 = ", ".join(f"{item['label']}={item['confidence']:.2f}" for item in payload["raw_predictions"]) or "-"
    selected = payload.get("selected_candidate") or {}
    print(
        "[ISYARA AI] predict "
        f"frame={frame_id or '-'} mirrored={mirrored} "
        f"top={result.display_label or '-'} conf={(result.confidence or 0):.2f} "
        f"raw=[{top3}] detections={payload['detections']} threshold={payload['threshold_detections']} valid={payload['valid_detections']} "
        f"hands={hands_detected} handedness={handedness or '-'} roi_type={payload.get('roi_type') or '-'} selected={selected.get('roi_type') or '-'} "
        f"roi={roi or '-'} bbox={payload.get('bbox') or '-'} reject={payload.get('rejection_reason') or '-'} "
        f"stable={payload['stable']} accepted={payload['accepted']} suppressed={payload['suppressed']} "
        f"lock={payload['locked_label'] or '-'} release={payload['release_misses']} "
        f"image={payload.get('image_width')}x{payload.get('image_height')} mode={payload.get('image_mode')} "
        f"latency_ms={payload.get('latency_ms')}"
    )
    return PredictSignResponse(**payload)


def _roi_payload(x1, y1, x2, y2, source_width, source_height):
    values = [x1, y1, x2, y2, source_width, source_height]
    if any(value is None for value in values):
        return None
    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "source_width": int(source_width),
        "source_height": int(source_height),
    }


def _candidate_metadata(candidates_json):
    if not candidates_json:
        return []
    try:
        data = json.loads(candidates_json)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


async def _predict_candidate_batch(uploads, candidate_meta):
    candidate_payloads = []
    selected = None
    fallback = None
    for index, upload in enumerate(uploads):
        meta = candidate_meta[index] if index < len(candidate_meta) and isinstance(candidate_meta[index], dict) else {}
        roi = _roi_from_meta(meta)
        roi_type = str(meta.get("type") or f"candidate_{index + 1}")
        result = _active_predictor().predict(await upload.read())
        payload = result.as_dict()
        if roi and payload.get("bbox"):
            payload["bbox"] = _map_roi_bbox_to_source(payload["bbox"], roi)
        candidate = _selected_candidate_payload(payload, roi, roi_type)
        candidate["raw_predictions"] = payload.get("raw_predictions", [])
        candidate_payloads.append(candidate)
        if fallback is None or (candidate.get("confidence") or 0) > (fallback[1].get("confidence") or 0):
            fallback = (result, payload, roi, roi_type)
        if result.detected and (selected is None or (result.confidence or 0) > (selected[0].confidence or 0)):
            selected = (result, payload, roi, roi_type)

    result, payload, roi, roi_type = selected or fallback
    payload["roi_type"] = roi_type
    payload["candidate_predictions"] = candidate_payloads
    payload["selected_candidate"] = _selected_candidate_payload(payload, roi, roi_type)
    return result, payload, roi


def _active_inference_mode():
    if BISINDO_INFERENCE_MODE == "classifier" and bisindo_classifier.model_available:
        return "classifier"
    return "detector"


def _active_predictor():
    if _active_inference_mode() == "classifier":
        return bisindo_classifier
    return bisindo_detector


def _roi_from_meta(meta):
    return _roi_payload(
        meta.get("x1"),
        meta.get("y1"),
        meta.get("x2"),
        meta.get("y2"),
        meta.get("source_width"),
        meta.get("source_height"),
    )


def _selected_candidate_payload(payload, roi, roi_type):
    return {
        "roi_type": roi_type,
        "status": payload.get("status"),
        "detected": payload.get("detected"),
        "class_id": payload.get("class_id"),
        "raw_label": payload.get("raw_label"),
        "label": payload.get("display_label") or payload.get("label"),
        "confidence": payload.get("confidence"),
        "bbox": payload.get("bbox"),
        "roi": roi,
        "rejection_reason": payload.get("rejection_reason"),
    }


def _map_roi_bbox_to_source(bbox, roi):
    return {
        "x1": bbox["x1"] + roi["x1"],
        "y1": bbox["y1"] + roi["y1"],
        "x2": bbox["x2"] + roi["x1"],
        "y2": bbox["y2"] + roi["y1"],
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        return {"status": "invalid_audio", "language": "id", "text": "", "message": "File audio tidak valid."}
    with NamedTemporaryFile(suffix=".webm") as temp_audio:
        temp_audio.write(await audio.read())
        temp_audio.flush()
        return speech_transcription.transcribe(temp_audio.name)
