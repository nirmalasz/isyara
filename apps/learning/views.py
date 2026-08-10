from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.mixins import LoginRequiredMixin
from django.conf import settings
from django.db.models import Avg, Count, Max
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.generic import DetailView, FormView, ListView, TemplateView, UpdateView
from django_ratelimit.decorators import ratelimit

from .forms import LoginForm, ProfileForm, SignupForm
from .models import Lesson, PracticeSession, Progress, Sign, UserProfile
from .services import AssessmentClient


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
    success_url = reverse_lazy("library")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["google_oauth_configured"] = bool(settings.SOCIALACCOUNT_PROVIDERS["google"]["APP"]["client_id"])
        return context

    def form_valid(self, form):
        user = form.save()
        login(self.request, user, backend="django.contrib.auth.backends.ModelBackend")
        messages.success(self.request, "Akun berhasil dibuat. Selamat belajar di ISYARA.")
        return super().form_valid(form)


@method_decorator(ratelimit(key="ip", rate="5/m", method="POST", block=True), name="post")
class LoginView(FormView):
    template_name = "account/login.html"
    form_class = LoginForm

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["google_oauth_configured"] = bool(settings.SOCIALACCOUNT_PROVIDERS["google"]["APP"]["client_id"])
        return context

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["request"] = self.request
        return kwargs

    def get_success_url(self):
        next_url = self.request.GET.get("next") or self.request.POST.get("next")
        if next_url and next_url.startswith("/"):
            return next_url
        return reverse_lazy("library")

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


class ProfileView(LoginRequiredMixin, TemplateView):
    template_name = "learning/profile.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        profile, _ = UserProfile.objects.get_or_create(user=self.request.user)
        progress = Progress.objects.filter(user=self.request.user)
        sessions = PracticeSession.objects.select_related("lesson").filter(user=self.request.user, status=PracticeSession.STATUS_ANALYZED)
        completed_count = progress.filter(completed=True).count()
        average_score = sessions.aggregate(average=Avg("score"))["average"] or 0
        context.update(
            {
                "profile": profile,
                "completed_count": completed_count,
                "practice_count": sessions.count(),
                "average_score": round(average_score),
                "mastered_count": progress.filter(best_score__gte=85).count(),
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
        return Lesson.objects.select_related("sign").prefetch_related("parts")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["next_lesson"] = Lesson.objects.filter(order__gt=self.object.order).order_by("order").first()
        context["lesson_parts"] = self.object.parts.all()
        return context


class StartPracticeView(LoginRequiredMixin, View):
    def post(self, request, slug):
        lesson = get_object_or_404(Lesson.objects.select_related("ai_reference"), slug=slug)
        if not lesson.ai_practice_available:
            messages.info(request, "AI practice for this lesson is coming soon.")
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
        progress.completed = progress.completed or result.score >= 80
        progress.last_practiced_at = timezone.now()
        progress.save(update_fields=["attempts", "best_score", "completed", "last_practiced_at"])
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
        context["next_lesson"] = Lesson.objects.filter(order__gt=self.object.lesson.order).order_by("order").first()
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
