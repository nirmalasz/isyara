from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import AIReference, LearningModule, Lesson, LessonPart, PracticeSession, Progress, Sign, TranslationHistory, User, UserProfile


@admin.register(User)
class IsyaraUserAdmin(UserAdmin):
    fieldsets = UserAdmin.fieldsets + (("ISYARA", {"fields": ("display_name",)}),)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "learning_level", "created_at", "updated_at")
    search_fields = ("user__email", "user__display_name", "bio")
    list_filter = ("learning_level",)


@admin.register(Sign)
class SignAdmin(admin.ModelAdmin):
    list_display = ("title", "category", "difficulty")
    list_filter = ("category", "difficulty")
    prepopulated_fields = {"slug": ("title",)}
    search_fields = ("title", "description")


@admin.register(LearningModule)
class LearningModuleAdmin(admin.ModelAdmin):
    list_display = ("title", "order", "difficulty", "is_active")
    list_filter = ("is_active", "difficulty")
    prepopulated_fields = {"slug": ("title",)}
    search_fields = ("title", "description")


@admin.register(Lesson)
class LessonAdmin(admin.ModelAdmin):
    list_display = ("title", "module", "sign", "order", "estimated_minutes", "youtube_video_id", "ai_practice_available")
    list_filter = ("module", "ai_practice_available", "region")
    prepopulated_fields = {"slug": ("title",)}
    search_fields = ("title", "summary", "instruction", "youtube_url", "source_channel")


@admin.register(LessonPart)
class LessonPartAdmin(admin.ModelAdmin):
    list_display = ("lesson", "title", "order", "video_start_seconds", "video_end_seconds", "scoring_weight", "ai_practice_available")
    list_filter = ("ai_practice_available",)
    search_fields = ("lesson__title", "title", "instruction")


@admin.register(AIReference)
class AIReferenceAdmin(admin.ModelAdmin):
    list_display = ("lesson", "active_hand", "required_hands", "uses_face", "uses_upper_body", "is_dynamic", "manually_confirmed")
    list_filter = ("active_hand", "uses_face", "uses_upper_body", "is_dynamic", "manually_confirmed", "region")
    search_fields = ("lesson__title", "lesson__slug", "reference_video_path", "reference_landmarks_path")


@admin.register(PracticeSession)
class PracticeSessionAdmin(admin.ModelAdmin):
    list_display = ("lesson", "user", "status", "score", "started_at")
    list_filter = ("status",)


@admin.register(Progress)
class ProgressAdmin(admin.ModelAdmin):
    list_display = ("lesson", "user", "attempts", "best_score", "completed", "last_practiced_at")
    list_filter = ("completed",)


@admin.register(TranslationHistory)
class TranslationHistoryAdmin(admin.ModelAdmin):
    list_display = ("user", "direction", "translated_text", "confidence", "created_at")
    list_filter = ("direction",)
    search_fields = ("user__email", "user__display_name", "source_text", "translated_text")
