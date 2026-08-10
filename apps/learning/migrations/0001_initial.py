import django.contrib.auth.models
import django.contrib.auth.validators
import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("auth", "0012_alter_user_first_name_max_length"),
    ]

    operations = [
        migrations.CreateModel(
            name="User",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("password", models.CharField(max_length=128, verbose_name="password")),
                ("last_login", models.DateTimeField(blank=True, null=True, verbose_name="last login")),
                ("is_superuser", models.BooleanField(default=False)),
                ("username", models.CharField(error_messages={"unique": "A user with that username already exists."}, help_text="Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.", max_length=150, unique=True, validators=[django.contrib.auth.validators.UnicodeUsernameValidator()], verbose_name="username")),
                ("first_name", models.CharField(blank=True, max_length=150, verbose_name="first name")),
                ("last_name", models.CharField(blank=True, max_length=150, verbose_name="last name")),
                ("email", models.EmailField(blank=True, max_length=254, verbose_name="email address")),
                ("is_staff", models.BooleanField(default=False)),
                ("is_active", models.BooleanField(default=True)),
                ("date_joined", models.DateTimeField(default=django.utils.timezone.now)),
                ("display_name", models.CharField(blank=True, max_length=120)),
                ("groups", models.ManyToManyField(blank=True, help_text="The groups this user belongs to.", related_name="user_set", related_query_name="user", to="auth.group", verbose_name="groups")),
                ("user_permissions", models.ManyToManyField(blank=True, help_text="Specific permissions for this user.", related_name="user_set", related_query_name="user", to="auth.permission", verbose_name="user permissions")),
            ],
            options={
                "verbose_name": "user",
                "verbose_name_plural": "users",
                "abstract": False,
            },
            managers=[
                ("objects", django.contrib.auth.models.UserManager()),
            ],
        ),
        migrations.CreateModel(
            name="Sign",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("title", models.CharField(max_length=120)),
                ("slug", models.SlugField(unique=True)),
                ("category", models.CharField(choices=[("alphabet", "Alphabet"), ("vocabulary", "Vocabulary")], max_length=24)),
                ("description", models.TextField()),
                ("difficulty", models.PositiveSmallIntegerField(default=1)),
                ("image_hint", models.CharField(blank=True, max_length=160)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["category", "title"],
            },
        ),
        migrations.CreateModel(
            name="Lesson",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("title", models.CharField(max_length=160)),
                ("slug", models.SlugField(unique=True)),
                ("summary", models.TextField()),
                ("instruction", models.TextField()),
                ("order", models.PositiveIntegerField(default=0)),
                ("estimated_minutes", models.PositiveSmallIntegerField(default=5)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("sign", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="lesson", to="learning.sign")),
            ],
            options={
                "ordering": ["order", "title"],
            },
        ),
        migrations.CreateModel(
            name="PracticeSession",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("status", models.CharField(choices=[("started", "Started"), ("analyzed", "Analyzed")], default="started", max_length=24)),
                ("score", models.PositiveSmallIntegerField(blank=True, null=True)),
                ("feedback", models.JSONField(blank=True, default=dict)),
                ("started_at", models.DateTimeField(auto_now_add=True)),
                ("analyzed_at", models.DateTimeField(blank=True, null=True)),
                ("lesson", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="practice_sessions", to="learning.lesson")),
                ("user", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="practice_sessions", to=settings.AUTH_USER_MODEL)),
            ],
            options={
                "ordering": ["-started_at"],
            },
        ),
        migrations.CreateModel(
            name="Progress",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("attempts", models.PositiveIntegerField(default=0)),
                ("best_score", models.PositiveSmallIntegerField(default=0)),
                ("completed", models.BooleanField(default=False)),
                ("last_practiced_at", models.DateTimeField(blank=True, null=True)),
                ("lesson", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="progress_records", to="learning.lesson")),
                ("user", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="progress_records", to=settings.AUTH_USER_MODEL)),
            ],
            options={
                "ordering": ["lesson__order"],
            },
        ),
        migrations.AddConstraint(
            model_name="progress",
            constraint=models.UniqueConstraint(fields=("user", "lesson"), name="unique_user_lesson_progress"),
        ),
    ]
