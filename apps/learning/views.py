from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.mixins import LoginRequiredMixin
from django.conf import settings
from django.db.models import Avg, Count, Max
import json

import requests
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.generic import DetailView, FormView, ListView, TemplateView, UpdateView
from django_ratelimit.decorators import ratelimit

from .chatbot_service import generate_chatbot_reply
from .forms import LoginForm, OnboardingForm, ProfileForm, SignupForm
from .models import Lesson, LearningModule, PracticeSession, Progress, Sign, TranslationHistory, UserProfile
from .progression import build_learning_path, learning_summary, lesson_state
from .services import AssessmentClient, TranslatorAIClient


def current_user_or_none(request):
    return request.user if request.user.is_authenticated else None


class LandingPageView(TemplateView):
    template_name = "learning/landing.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["featured_lessons"] = Lesson.objects.select_related("sign")[:4]
        context["lesson_count"] = Lesson.objects.count()
        return context


@method_decorator(ratelimit(key="ip", rate="10/m", method="POST", block=True), name="post")
class SignupView(FormView):
    template_name = "account/signup.html"
    form_class = SignupForm
    success_url = reverse_lazy("translator")

    def form_valid(self, form):
        user = form.save()
        login(self.request, user, backend="django.contrib.auth.backends.ModelBackend")
        messages.success(self.request, "Akun berhasil dibuat. Selamat belajar di ISYARA.")
        return super().form_valid(form)


@method_decorator(ratelimit(key="ip", rate="5/m", method="POST", block=True), name="post")
class LoginView(FormView):
    template_name = "account/login.html"
    form_class = LoginForm

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["request"] = self.request
        return kwargs

    def get_success_url(self):
        next_url = self.request.GET.get("next") or self.request.POST.get("next")
        if next_url and next_url.startswith("/"):
            return next_url
        return reverse_lazy("translator")

    def form_valid(self, form):
        login(self.request, form.user)
        messages.success(self.request, "Berhasil masuk.")
        return super().form_valid(form)


class LogoutView(LoginRequiredMixin, View):
    def post(self, request):
        logout(request)
        messages.success(request, "Kamu sudah keluar.")
        return redirect("landing")


class PasswordResetView(TemplateView):
    template_name = "account/password_reset.html"


class TranslatorView(LoginRequiredMixin, TemplateView):
    template_name = "translator/translate.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["debug_inference"] = settings.DEBUG
        context["eval_mode"] = settings.DEBUG and self.request.GET.get("eval") == "1"
        return context


class HistoryView(LoginRequiredMixin, TemplateView):
    template_name = "translator/history.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["history_items"] = TranslationHistory.objects.filter(user=self.request.user)[:50]
        return context


class DeleteHistoryItemView(LoginRequiredMixin, View):
    def post(self, request, history_id):
        item = get_object_or_404(TranslationHistory, id=history_id, user=request.user)
        item.delete()
        messages.success(request, "Riwayat terjemahan dihapus.")
        return redirect("translation_history")


class ClearHistoryView(LoginRequiredMixin, View):
    def post(self, request):
        TranslationHistory.objects.filter(user=request.user).delete()
        messages.success(request, "Semua riwayat terjemahan dihapus.")
        return redirect("translation_history")


@method_decorator(ratelimit(key="user_or_ip", rate="8/m", method="POST", block=True), name="post")
class OnboardingView(LoginRequiredMixin, UpdateView):
    template_name = "learning/onboarding.html"
    form_class = OnboardingForm
    success_url = reverse_lazy("learning_path")

    def dispatch(self, request, *args, **kwargs):
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        if profile.onboarding_completed and request.method == "GET":
            return redirect("learning_path")
        return super().dispatch(request, *args, **kwargs)

    def get_object(self, queryset=None):
        profile, _ = UserProfile.objects.get_or_create(user=self.request.user)
        return profile


class LearningPathView(LoginRequiredMixin, TemplateView):
    template_name = "learning/learning_path.html"

    def dispatch(self, request, *args, **kwargs):
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        if not profile.onboarding_completed:
            return redirect("onboarding")
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        path = build_learning_path(self.request.user)
        summary = learning_summary(self.request.user)
        progress = Progress.objects.filter(user=self.request.user)
        sessions = PracticeSession.objects.filter(user=self.request.user, status=PracticeSession.STATUS_ANALYZED)
        average_score = sessions.aggregate(average=Avg("score"))["average"] or 0
        context.update(
            {
                "path_modules": path["modules"],
                "next_lesson": path["next_lesson"],
                "completed_lessons": summary["completed_lessons"],
                "total_practices": summary["total_practices"],
                "average_score": round(average_score),
                "streak_days": 1 if progress.filter(last_practiced_at__date=timezone.localdate()).exists() else 0,
            }
        )
        return context


class ProfileView(LoginRequiredMixin, TemplateView):
    template_name = "learning/profile.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile, _ = UserProfile.objects.get_or_create(user=self.request.user)
        progress = Progress.objects.filter(user=self.request.user)
        sessions = PracticeSession.objects.select_related("lesson").filter(user=self.request.user, status=PracticeSession.STATUS_ANALYZED)
        completed_count = progress.filter(completed=True).count()
        average_score = sessions.aggregate(average=Avg("score"))["average"] or 0
        summary = learning_summary(self.request.user)
        translation_history = TranslationHistory.objects.filter(user=self.request.user)
        context.update(
            {
                "profile": profile,
                "completed_count": completed_count,
                "practice_count": sessions.count(),
                "average_score": round(average_score),
                "mastered_count": progress.filter(best_score__gte=85).count(),
                "current_module": summary["current_module"],
                "translation_count": translation_history.count(),
                "sign_translation_count": translation_history.filter(direction=TranslationHistory.DIRECTION_SIGN_TO_SPEECH).count(),
                "speech_translation_count": translation_history.filter(direction=TranslationHistory.DIRECTION_SPEECH_TO_TEXT).count(),
                "recent_translations": translation_history[:5],
                "recent_sessions": sessions[:5],
            }
        )
        return context


@method_decorator(ratelimit(key="user_or_ip", rate="8/m", method="POST", block=True), name="post")
class ProfileEditView(LoginRequiredMixin, UpdateView):
    template_name = "learning/profile_edit.html"
    form_class = ProfileForm
    success_url = reverse_lazy("profile")

    def get_object(self, queryset=None):
        profile, _ = UserProfile.objects.get_or_create(user=self.request.user)
        return profile

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def form_valid(self, form):
        messages.success(self.request, "Profil berhasil diperbarui.")
        return super().form_valid(form)


class LearningLibraryView(ListView):
    template_name = "learning/library.html"
    context_object_name = "lessons"

    def get_queryset(self):
        queryset = Lesson.objects.select_related("sign").prefetch_related("parts")
        category = self.request.GET.get("category")
        if category in dict(Sign.CATEGORY_CHOICES):
            queryset = queryset.filter(sign__category=category)
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["categories"] = Sign.CATEGORY_CHOICES
        context["active_category"] = self.request.GET.get("category", "")
        return context


class LessonDetailView(DetailView):
    model = Lesson
    template_name = "learning/lesson_detail.html"
    context_object_name = "lesson"

    def get_queryset(self):
        return Lesson.objects.select_related("sign", "module").prefetch_related("parts")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        module_item, lesson_item = lesson_state(self.request.user, self.object) if self.request.user.is_authenticated else (None, None)
        context["module_item"] = module_item
        context["lesson_state"] = lesson_item
        context["next_lesson"] = Lesson.objects.filter(module=self.object.module, order__gt=self.object.order).order_by("order").first()
        context["lesson_parts"] = self.object.parts.all()
        return context

    def get(self, request, *args, **kwargs):
        self.object = self.get_object()
        if request.user.is_authenticated:
            _, state = lesson_state(request.user, self.object)
            if state and not state.get("unlocked", True):
                messages.info(request, "Pelajaran ini masih terkunci. Selesaikan materi sebelumnya dulu.")
                return redirect("learning_path")
        return self.render_to_response(self.get_context_data(object=self.object))


class StartPracticeView(LoginRequiredMixin, View):
    def post(self, request, slug):
        lesson = get_object_or_404(Lesson.objects.select_related("ai_reference", "module"), slug=slug)
        _, state = lesson_state(request.user, lesson)
        if state and not state.get("unlocked", True):
            messages.info(request, "Pelajaran ini masih terkunci. Selesaikan materi sebelumnya dulu.")
            return redirect("learning_path")
        if not lesson.ai_practice_available:
            messages.info(request, "Latihan AI untuk pelajaran ini segera hadir.")
            return redirect(lesson.get_absolute_url())
        if not hasattr(lesson, "ai_reference"):
            messages.info(request, "Referensi AI untuk pelajaran ini belum tersedia.")
            return redirect(lesson.get_absolute_url())
        session = PracticeSession.objects.create(
            lesson=lesson,
            user=request.user,
        )
        return redirect("practice_camera", session_id=session.id)


class AIPracticeView(LoginRequiredMixin, DetailView):
    model = PracticeSession
    pk_url_kwarg = "session_id"
    template_name = "learning/practice_camera.html"
    context_object_name = "session"

    def get_queryset(self):
        return PracticeSession.objects.select_related("lesson", "lesson__sign", "lesson__ai_reference").filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        ai_reference = getattr(self.object.lesson, "ai_reference", None)
        context["ai_reference_config"] = ai_reference.client_config() if ai_reference else {
            "sign": self.object.lesson.slug,
            "required_hands": ["right"],
            "active_hand": "right",
            "uses_face": False,
            "uses_upper_body": True,
            "is_dynamic": True,
            "region": self.object.lesson.region,
            "component_weights": {},
            "reference_features": {},
            "manually_confirmed": False,
        }
        return context


@method_decorator(ratelimit(key="user_or_ip", rate="12/m", method="POST", block=True), name="post")
class AnalyzePracticeView(LoginRequiredMixin, View):
    def post(self, request, session_id):
        session = get_object_or_404(
            PracticeSession.objects.select_related("lesson", "lesson__sign"),
            id=session_id,
            user=request.user,
        )
        result = AssessmentClient().analyze(session)
        session.score = result.score
        session.feedback = result.as_feedback()
        session.status = PracticeSession.STATUS_ANALYZED
        session.analyzed_at = timezone.now()
        session.save(update_fields=["score", "feedback", "status", "analyzed_at"])

        progress, _ = Progress.objects.get_or_create(
            user=request.user,
            lesson=session.lesson,
        )
        progress.attempts += 1
        progress.best_score = max(progress.best_score, result.score)
        progress.latest_score = result.score
        if not progress.completed:
            progress.completed = True
            progress.completed_at = timezone.now()
        progress.last_practiced_at = timezone.now()
        progress.save(update_fields=["attempts", "best_score", "latest_score", "completed", "completed_at", "last_practiced_at"])
        return redirect("practice_result", session_id=session.id)


class PracticeResultView(LoginRequiredMixin, DetailView):
    model = PracticeSession
    pk_url_kwarg = "session_id"
    template_name = "learning/practice_result.html"
    context_object_name = "session"

    def get_queryset(self):
        return PracticeSession.objects.select_related("lesson", "lesson__sign").filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        path = build_learning_path(self.request.user)
        context["next_lesson"] = path["next_lesson"]
        return context


class ProgressDashboardView(LoginRequiredMixin, TemplateView):
    template_name = "learning/dashboard.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        progress = Progress.objects.select_related("lesson", "lesson__sign").filter(user=self.request.user)
        sessions = PracticeSession.objects.select_related("lesson").filter(user=self.request.user)
        totals = progress.aggregate(
            record_count=Count("id"),
            max_best_score=Max("best_score"),
            average_best_score=Avg("best_score"),
        )
        lesson_count = Lesson.objects.count()
        completed_count = progress.filter(completed=True).count()
        context.update(
            {
                "progress_records": progress,
                "path_modules": build_learning_path(self.request.user)["modules"],
                "lesson_count": lesson_count,
                "attempt_count": sum(record.attempts for record in progress),
                "completed_count": completed_count,
                "completion_percent": round((completed_count / lesson_count) * 100) if lesson_count else 0,
                "best_score": totals["max_best_score"] or 0,
                "average_score": round(totals["average_best_score"] or 0),
                "recent_session": sessions.first(),
                "recommended_lesson": Lesson.objects.exclude(progress_records__user=self.request.user, progress_records__completed=True).first(),
            }
        )
        return context


class MeAPIView(LoginRequiredMixin, View):
    def get(self, request):
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        return JsonResponse(
            {
                "id": request.user.id,
                "email": request.user.email,
                "display_name": request.user.display_name,
                "profile": {
                    "learning_level": profile.learning_level,
                    "learning_goal": profile.learning_goal,
                    "bisindo_familiarity": profile.bisindo_familiarity,
                    "onboarding_completed": profile.onboarding_completed,
                },
            }
        )


class LearningPathAPIView(LoginRequiredMixin, View):
    def get(self, request):
        path = build_learning_path(request.user)
        return JsonResponse(
            {
                "next_lesson": path["next_lesson"].slug if path["next_lesson"] else None,
                "modules": [
                    {
                        "id": item["module"].id,
                        "title": item["module"].title,
                        "slug": item["module"].slug,
                        "description": item["module"].description,
                        "completed_count": item["completed_count"],
                        "total_lessons": item["total_lessons"],
                        "completion_percent": item["completion_percent"],
                        "unlocked": item["unlocked"],
                        "checkpoint_unlocked": item["checkpoint_unlocked"],
                        "lessons": [
                            {
                                "slug": lesson_item["lesson"].slug,
                                "title": lesson_item["lesson"].title,
                                "order": lesson_item["number"],
                                "status": lesson_item["status"],
                                "status_label": lesson_item["status_label"],
                                "unlocked": lesson_item["unlocked"],
                                "ai_practice_available": lesson_item["lesson"].ai_practice_available,
                            }
                            for lesson_item in item["lessons"]
                        ],
                    }
                    for item in path["modules"]
                ],
            }
        )


class ModuleDetailAPIView(LoginRequiredMixin, View):
    def get(self, request, module_id):
        module = get_object_or_404(LearningModule.objects.prefetch_related("lessons__sign"), id=module_id, is_active=True)
        path = build_learning_path(request.user)
        module_item = next((item for item in path["modules"] if item["module"].id == module.id), None)
        if not module_item:
            return JsonResponse({"detail": "Module tidak ditemukan."}, status=404)
        return JsonResponse(
            {
                "id": module.id,
                "title": module.title,
                "slug": module.slug,
                "description": module.description,
                "difficulty": module.difficulty,
                "unlocked": module_item["unlocked"],
                "completion_percent": module_item["completion_percent"],
                "lessons": [
                    {
                        "slug": lesson_item["lesson"].slug,
                        "title": lesson_item["lesson"].title,
                        "summary": lesson_item["lesson"].summary,
                        "status": lesson_item["status"],
                        "status_label": lesson_item["status_label"],
                        "unlocked": lesson_item["unlocked"],
                    }
                    for lesson_item in module_item["lessons"]
                ],
            }
        )


class LessonDetailAPIView(LoginRequiredMixin, View):
    def get(self, request, slug):
        lesson = get_object_or_404(Lesson.objects.select_related("sign", "module"), slug=slug)
        _, state = lesson_state(request.user, lesson)
        if state and not state.get("unlocked", True):
            return JsonResponse({"detail": "Pelajaran terkunci.", "reason": state["locked_reason"]}, status=403)
        return JsonResponse(
            {
                "slug": lesson.slug,
                "title": lesson.title,
                "summary": lesson.summary,
                "instruction": lesson.instruction,
                "module": lesson.module.title if lesson.module else None,
                "youtube_embed_url": lesson.youtube_embed_url,
                "youtube_url": lesson.youtube_url,
                "ai_practice_available": lesson.ai_practice_available,
                "status": state.get("status") if state else "current",
                "status_label": state.get("status_label") if state else "Saat Ini",
            }
        )


class StartLessonAPIView(LoginRequiredMixin, View):
    def post(self, request, slug):
        lesson = get_object_or_404(Lesson.objects.select_related("ai_reference", "module"), slug=slug)
        _, state = lesson_state(request.user, lesson)
        if state and not state.get("unlocked", True):
            return JsonResponse({"detail": "Pelajaran terkunci.", "reason": state["locked_reason"]}, status=403)
        if not lesson.ai_practice_available or not hasattr(lesson, "ai_reference"):
            return JsonResponse({"detail": "Latihan AI untuk pelajaran ini segera hadir."}, status=409)
        session = PracticeSession.objects.create(lesson=lesson, user=request.user)
        return JsonResponse(
            {
                "session_id": session.id,
                "practice_url": reverse("practice_camera", kwargs={"session_id": session.id}),
            },
            status=201,
        )


class ProgressAPIView(LoginRequiredMixin, View):
    def get(self, request):
        progress = Progress.objects.select_related("lesson", "lesson__module").filter(user=request.user)
        return JsonResponse(
            {
                "records": [
                    {
                        "lesson": record.lesson.slug,
                        "module": record.lesson.module.slug if record.lesson.module else None,
                        "attempts": record.attempts,
                        "best_score": record.best_score,
                        "latest_score": record.latest_score,
                        "completed": record.completed,
                        "completed_at": record.completed_at.isoformat() if record.completed_at else None,
                    }
                    for record in progress
                ]
            }
        )


@method_decorator(ratelimit(key="user_or_ip", rate="480/m", method="POST", block=True), name="post")
class PredictSignAPIView(LoginRequiredMixin, View):
    def post(self, request):
        image_file = request.FILES.get("image")
        candidate_files = request.FILES.getlist("candidates")
        is_no_hand_ping = request.POST.get("hands_detected") == "0"
        if not image_file and not candidate_files and not is_no_hand_ping:
            return JsonResponse({"status": "error", "message": "Frame kamera wajib dikirim."}, status=400)
        uploads = candidate_files or [image_file]
        if any(upload and not getattr(upload, "content_type", "").startswith("image/") for upload in uploads):
            return JsonResponse({"status": "error", "message": "Frame kamera harus berupa gambar."}, status=400)
        metadata = {
            key: value
            for key, value in {
                "frame_id": request.POST.get("frame_id"),
                "mirrored": request.POST.get("mirrored"),
                "roi_x1": request.POST.get("roi_x1"),
                "roi_y1": request.POST.get("roi_y1"),
                "roi_x2": request.POST.get("roi_x2"),
                "roi_y2": request.POST.get("roi_y2"),
                "source_width": request.POST.get("source_width"),
                "source_height": request.POST.get("source_height"),
                "hands_detected": request.POST.get("hands_detected"),
                "handedness": request.POST.get("handedness"),
                "candidates_json": request.POST.get("candidates_json"),
                "structure_json": request.POST.get("structure_json"),
                "timestamp_ms": request.POST.get("timestamp_ms"),
            }.items()
            if value not in {None, ""}
        }
        try:
            result = TranslatorAIClient().predict_sign(
                image_file,
                metadata=metadata,
                candidate_files=candidate_files,
            )
        except requests.RequestException:
            result = {
                "status": "service_unavailable",
                "detected": False,
                "class_id": None,
                "label": None,
                "prediction": None,
                "display_text": "Layanan penerjemah belum tersedia.",
                "confidence": None,
            }
        return JsonResponse(result)


@method_decorator(ratelimit(key="user_or_ip", rate="12/m", method="POST", block=True), name="post")
class TranscribeSpeechAPIView(LoginRequiredMixin, View):
    def post(self, request):
        audio_file = request.FILES.get("audio")
        if not audio_file:
            return JsonResponse({"status": "error", "message": "Audio wajib dikirim."}, status=400)
        try:
            result = TranslatorAIClient().transcribe(audio_file)
        except requests.RequestException:
            result = {
                "status": "service_unavailable",
                "language": "id",
                "text": "",
                "message": "Layanan transkripsi belum tersedia.",
            }
        return JsonResponse(result)


@method_decorator(ratelimit(key="user_or_ip", rate="20/m", method="POST", block=True), name="post")
class SaveTranslationHistoryAPIView(LoginRequiredMixin, View):
    def post(self, request):
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JsonResponse({"status": "error", "message": "Payload tidak valid."}, status=400)
        direction = payload.get("direction")
        translated_text = (payload.get("translated_text") or "").strip()
        if direction not in dict(TranslationHistory.DIRECTION_CHOICES):
            return JsonResponse({"status": "error", "message": "Arah terjemahan tidak valid."}, status=400)
        if not translated_text:
            return JsonResponse({"status": "error", "message": "Teks terjemahan wajib diisi."}, status=400)
        item = TranslationHistory.objects.create(
            user=request.user,
            direction=direction,
            source_text=(payload.get("source_text") or "").strip(),
            translated_text=translated_text,
            confidence=payload.get("confidence"),
        )
        return JsonResponse({"status": "saved", "id": item.id}, status=201)


@method_decorator(ratelimit(key="user_or_ip", rate="30/m", method="POST", block=True), name="post")
class ChatbotReplyAPIView(LoginRequiredMixin, View):
    def post(self, request):
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JsonResponse({"error": "Payload tidak valid."}, status=400)

        user_query = (payload.get("message") or "").strip()
        if not user_query:
            return JsonResponse({"error": "Pesan tidak boleh kosong."}, status=400)
        if len(user_query) > 300:
            return JsonResponse({"error": "Pesan terlalu panjang."}, status=400)

        return JsonResponse(generate_chatbot_reply(user_query))
