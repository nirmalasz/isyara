from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError
from django.db import models
from django.urls import reverse
from django.utils import timezone
from pathlib import Path
from uuid import uuid4

from .utils import youtube_video_id_from_url


def profile_photo_upload_path(instance, filename):
    extension = Path(filename).suffix.lower()
    return f"profiles/{instance.user_id}/{uuid4().hex}{extension}"


def validate_profile_photo(uploaded_file):
    max_size = 2 * 1024 * 1024
    allowed_content_types = {"image/jpeg", "image/png", "image/webp"}
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    extension = Path(uploaded_file.name).suffix.lower()
    if uploaded_file.size > max_size:
        raise ValidationError("Ukuran foto profil maksimal 2 MB.")
    if extension not in allowed_extensions:
        raise ValidationError("Format foto harus JPEG, PNG, atau WebP.")
    content_type = getattr(uploaded_file, "content_type", "")
    if content_type and content_type not in allowed_content_types:
        raise ValidationError("Tipe file foto tidak didukung.")


class User(AbstractUser):
    display_name = models.CharField(max_length=120, blank=True)
    email = models.EmailField(unique=True)

    def __str__(self):
        return self.display_name or self.username


class UserProfile(models.Model):
    LEVEL_BEGINNER = "pemula"
    LEVEL_INTERMEDIATE = "menengah"
    LEVEL_ADVANCED = "lanjutan"
    LEVEL_CHOICES = [
        (LEVEL_BEGINNER, "Pemula"),
        (LEVEL_INTERMEDIATE, "Menengah"),
        (LEVEL_ADVANCED, "Lanjutan"),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    profile_photo = models.ImageField(upload_to=profile_photo_upload_path, validators=[validate_profile_photo], blank=True)
    bio = models.TextField(blank=True, max_length=500)
    learning_level = models.CharField(max_length=20, choices=LEVEL_CHOICES, default=LEVEL_BEGINNER)
    learning_goal = models.CharField(max_length=120, blank=True)
    bisindo_familiarity = models.CharField(max_length=80, blank=True)
    onboarding_completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profil {self.user}"

    @property
    def initials(self):
        name = self.user.display_name or self.user.get_full_name() or self.user.email or self.user.username
        parts = [part[0] for part in name.split() if part]
        return "".join(parts[:2]).upper() or "I"

    @property
    def joined_month(self):
        return timezone.localtime(self.user.date_joined).strftime("%B %Y")

    @property
    def learning_goal_label(self):
        return {
            "daily": "Komunikasi sehari-hari",
            "deaf_family_friends": "Berkomunikasi dengan keluarga/teman Tuli",
            "school": "Sekolah atau kampus",
            "work_service": "Dunia kerja / pelayanan",
            "self": "Belajar untuk diri sendiri",
        }.get(self.learning_goal, "Belum diisi")

    @property
    def bisindo_familiarity_label(self):
        return {
            "new": "Belum pernah belajar",
            "some": "Pernah belajar sedikit",
            "basic": "Sudah memahami dasar",
        }.get(self.bisindo_familiarity, "Belum diisi")


class Sign(models.Model):
    CATEGORY_ALPHABET = "alphabet"
    CATEGORY_VOCABULARY = "vocabulary"
    CATEGORY_CHOICES = [
        (CATEGORY_ALPHABET, "Alphabet"),
        (CATEGORY_VOCABULARY, "Vocabulary"),
    ]

    title = models.CharField(max_length=120)
    slug = models.SlugField(unique=True)
    category = models.CharField(max_length=24, choices=CATEGORY_CHOICES)
    description = models.TextField()
    difficulty = models.PositiveSmallIntegerField(default=1)
    image_hint = models.CharField(max_length=160, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["category", "title"]

    def __str__(self):
        return self.title


class LearningModule(models.Model):
    title = models.CharField(max_length=160)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    order = models.PositiveIntegerField(default=0)
    difficulty = models.CharField(max_length=80, blank=True, default="Pemula")
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["order", "title"]

    def __str__(self):
        return self.title


class Lesson(models.Model):
    module = models.ForeignKey(LearningModule, on_delete=models.SET_NULL, related_name="lessons", null=True, blank=True)
    sign = models.OneToOneField(Sign, on_delete=models.CASCADE, related_name="lesson")
    title = models.CharField(max_length=160)
    slug = models.SlugField(unique=True)
    summary = models.TextField()
    instruction = models.TextField()
    youtube_url = models.URLField(blank=True)
    youtube_video_id = models.CharField(max_length=32, blank=True)
    video_start_seconds = models.PositiveIntegerField(default=0)
    source_name = models.CharField(max_length=120, blank=True)
    source_channel = models.CharField(max_length=160, blank=True)
    region = models.CharField(max_length=120, blank=True, default="BISINDO")
    ai_practice_available = models.BooleanField(default=True)
    order = models.PositiveIntegerField(default=0)
    estimated_minutes = models.PositiveSmallIntegerField(default=5)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["order", "title"]

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        self.youtube_video_id = youtube_video_id_from_url(self.youtube_url)
        update_fields = kwargs.get("update_fields")
        if update_fields is not None:
            kwargs["update_fields"] = set(update_fields) | {"youtube_video_id"}
        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse("lesson_detail", kwargs={"slug": self.slug})

    @property
    def youtube_embed_url(self):
        if not self.youtube_video_id:
            return ""
        start_query = f"?start={self.video_start_seconds}" if self.video_start_seconds else ""
        return f"https://www.youtube.com/embed/{self.youtube_video_id}{start_query}"

    @property
    def youtube_thumbnail_url(self):
        if not self.youtube_video_id:
            return ""
        return f"https://img.youtube.com/vi/{self.youtube_video_id}/hqdefault.jpg"


class LessonPart(models.Model):
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, related_name="parts")
    title = models.CharField(max_length=120)
    order = models.PositiveIntegerField(default=0)
    video_start_seconds = models.PositiveIntegerField(default=0)
    video_end_seconds = models.PositiveIntegerField(null=True, blank=True)
    scoring_weight = models.PositiveSmallIntegerField(default=100)
    ai_practice_available = models.BooleanField(default=False)
    instruction = models.TextField(blank=True)

    class Meta:
        ordering = ["lesson", "order", "title"]
        constraints = [
            models.UniqueConstraint(fields=["lesson", "order"], name="unique_lesson_part_order"),
        ]

    def __str__(self):
        return f"{self.lesson.title} - {self.title}"

    @property
    def youtube_embed_url(self):
        if not self.lesson.youtube_video_id:
            return ""
        query = f"?start={self.video_start_seconds}"
        if self.video_end_seconds:
            query = f"{query}&end={self.video_end_seconds}"
        return f"https://www.youtube.com/embed/{self.lesson.youtube_video_id}{query}"


class AIReference(models.Model):
    ACTIVE_RIGHT = "right"
    ACTIVE_LEFT = "left"
    ACTIVE_BOTH = "both"
    ACTIVE_HAND_CHOICES = [
        (ACTIVE_RIGHT, "Right"),
        (ACTIVE_LEFT, "Left"),
        (ACTIVE_BOTH, "Both"),
    ]

    lesson = models.OneToOneField(Lesson, on_delete=models.CASCADE, related_name="ai_reference")
    required_hands = models.JSONField(default=list, blank=True)
    active_hand = models.CharField(max_length=12, choices=ACTIVE_HAND_CHOICES, default=ACTIVE_RIGHT)
    uses_face = models.BooleanField(default=False)
    uses_upper_body = models.BooleanField(default=True)
    is_dynamic = models.BooleanField(default=True)
    region = models.CharField(max_length=120, blank=True, default="BISINDO")
    reference_video_path = models.CharField(max_length=255, blank=True)
    reference_landmarks_path = models.CharField(max_length=255, blank=True)
    component_weights = models.JSONField(default=dict, blank=True)
    reference_features = models.JSONField(default=dict, blank=True)
    manually_confirmed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["lesson__order"]

    def __str__(self):
        return f"{self.lesson.title} AI reference"

    def client_config(self):
        return {
            "sign": self.lesson.slug,
            "required_hands": self.required_hands,
            "active_hand": self.active_hand,
            "uses_face": self.uses_face,
            "uses_upper_body": self.uses_upper_body,
            "is_dynamic": self.is_dynamic,
            "region": self.region or self.lesson.region,
            "component_weights": self.component_weights,
            "reference_features": self.reference_features,
            "manually_confirmed": self.manually_confirmed,
        }


class PracticeSession(models.Model):
    STATUS_STARTED = "started"
    STATUS_ANALYZED = "analyzed"
    STATUS_CHOICES = [
        (STATUS_STARTED, "Started"),
        (STATUS_ANALYZED, "Analyzed"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="practice_sessions", null=True, blank=True)
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, related_name="practice_sessions")
    status = models.CharField(max_length=24, choices=STATUS_CHOICES, default=STATUS_STARTED)
    score = models.PositiveSmallIntegerField(null=True, blank=True)
    feedback = models.JSONField(default=dict, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    analyzed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        return f"{self.lesson.title} practice ({self.status})"


class TranslationHistory(models.Model):
    DIRECTION_SIGN_TO_SPEECH = "SIGN_TO_SPEECH"
    DIRECTION_SPEECH_TO_TEXT = "SPEECH_TO_TEXT"
    DIRECTION_CHOICES = [
        (DIRECTION_SIGN_TO_SPEECH, "BISINDO ke Suara"),
        (DIRECTION_SPEECH_TO_TEXT, "Suara ke Teks"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="translation_history")
    direction = models.CharField(max_length=24, choices=DIRECTION_CHOICES)
    source_text = models.TextField(blank=True)
    translated_text = models.TextField()
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.get_direction_display()}: {self.translated_text[:40]}"

    @property
    def confidence_percent(self):
        if self.confidence is None:
            return None
        value = self.confidence * 100 if self.confidence <= 1 else self.confidence
        return round(value)


class Progress(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="progress_records", null=True, blank=True)
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, related_name="progress_records")
    attempts = models.PositiveIntegerField(default=0)
    best_score = models.PositiveSmallIntegerField(default=0)
    latest_score = models.PositiveSmallIntegerField(default=0)
    completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)
    last_practiced_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["user", "lesson"], name="unique_user_lesson_progress"),
        ]
        ordering = ["lesson__order"]

    def __str__(self):
        return f"{self.lesson.title}: {self.best_score}"
