from django.apps import AppConfig


class LearningConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.learning"

    def ready(self):
        from . import signals  # noqa: F401
