from dataclasses import dataclass

import requests
from django.conf import settings


@dataclass(frozen=True)
class AssessmentResult:
    score: int
    summary: str
    strengths: list[str]
    improvements: list[str]
    next_action: str
    component_scores: dict | None = None

    def as_feedback(self):
        return {
            "summary": self.summary,
            "strengths": self.strengths,
            "improvements": self.improvements,
            "next_action": self.next_action,
            "component_scores": self.component_scores or {},
        }


class AssessmentClient:
    def analyze(self, practice_session):
        if settings.ASSESSMENT_SERVICE_URL:
            return self._analyze_remote(practice_session)
        return self._analyze_mock(practice_session)

    def _analyze_remote(self, practice_session):
        response = requests.post(
            f"{settings.ASSESSMENT_SERVICE_URL.rstrip('/')}/assess",
            json={
                "session_id": practice_session.id,
                "lesson_slug": practice_session.lesson.slug,
                "sign_slug": practice_session.lesson.sign.slug,
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        return AssessmentResult(
            score=int(data["score"]),
            summary=data["summary"],
            strengths=data.get("strengths", []),
            improvements=data.get("improvements", []),
            next_action=data.get("next_action", "Latihan lagi saat kamu siap."),
            component_scores=data.get("component_scores", {}),
        )

    def _analyze_mock(self, practice_session):
        score_seed = (practice_session.id * 17 + practice_session.lesson.order * 7) % 21
        score = 76 + score_seed
        ai_reference = getattr(practice_session.lesson, "ai_reference", None)
        component_scores = self._mock_component_scores(score, ai_reference)
        return AssessmentResult(
            score=min(score, 96),
            summary="Ritme isyaratmu sudah cukup jelas untuk percobaan awal. Skor komponen mengikuti metadata referensi AI untuk tanda ini.",
            strengths=[
                "Bentuk tangan utama sudah mudah dikenali",
                "Tempo gerakan cukup stabil",
                "Posisi awal dan akhir terlihat percaya diri",
            ],
            improvements=[
                "Dekatkan tangan dominan ke zona referensi",
                "Tahan pose akhir sebentar sebelum rileks",
            ],
            next_action="Ulangi untuk skor yang lebih rapi, atau lanjut ke pelajaran berikutnya.",
            component_scores=component_scores,
        )

    def _mock_component_scores(self, score, ai_reference):
        labels = {
            "handshape": "Bentuk Tangan",
            "finger_configuration": "Konfigurasi Jari",
            "palm_orientation": "Arah Telapak",
            "location": "Lokasi Isyarat",
            "movement": "Gerakan",
            "timing": "Tempo",
            "facial_expression": "Ekspresi Wajah",
        }
        if not ai_reference:
            weights = {"handshape": 0.25, "finger_configuration": 0.20, "location": 0.20, "palm_orientation": 0.15, "movement": 0.20}
            uses_face = False
            required_hands = ["right"]
        else:
            weights = ai_reference.component_weights or {}
            uses_face = ai_reference.uses_face
            required_hands = ai_reference.required_hands

        components = {}
        for index, (key, label) in enumerate(labels.items()):
            weight = float(weights.get(key, 0) or 0)
            if weight <= 0:
                reason = "Tidak dinilai"
                if key == "facial_expression" and not uses_face:
                    reason = "Tidak dinilai"
                components[key] = {"label": label, "score": None, "weight": 0, "status": reason}
                continue
            components[key] = {
                "label": label,
                "score": max(60, min(98, score - 7 + (index * 3 % 11))),
                "weight": weight,
                "status": "Dinilai",
            }
        components["left_hand"] = {
            "label": "Tangan Kiri",
            "score": None,
            "weight": 0,
            "status": "Wajib" if "left" in required_hands else "Tidak wajib",
        }
        return components
