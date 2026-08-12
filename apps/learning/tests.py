import json

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from .models import AIReference, LearningModule, Lesson, PracticeSession, Progress, Sign, TranslationHistory, User, UserProfile
from .utils import youtube_video_id_from_url


class LearningFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="siswa@example.com",
            email="siswa@example.com",
            password="testpass-12345",
            display_name="Siswa ISYARA",
        )
        self.other_user = User.objects.create_user(
            username="lain@example.com",
            email="lain@example.com",
            password="testpass-12345",
            display_name="Siswa Lain",
        )
        sign = Sign.objects.create(
            title="Halo",
            slug="halo",
            category=Sign.CATEGORY_VOCABULARY,
            description="Sapaan ramah.",
            difficulty=1,
        )
        self.module = LearningModule.objects.create(
            title="Dasar BISINDO",
            slug="dasar-bisindo",
            description="Materi awal.",
            order=1,
        )
        self.lesson = Lesson.objects.create(
            module=self.module,
            sign=sign,
            title="Halo",
            slug="halo",
            summary="Sapaan ramah.",
            instruction="Latih sapaan dengan gerakan tangan yang jelas.",
            youtube_url="https://youtu.be/GCFfwXFi6hA?si=hzy9N0q5d7Zddg5h",
            order=1,
        )
        AIReference.objects.create(
            lesson=self.lesson,
            required_hands=["right"],
            active_hand="right",
            uses_face=False,
            uses_upper_body=True,
            is_dynamic=True,
            component_weights={"handshape": 0.2, "movement": 0.25},
            reference_features={"right": {"location": {"anchor": "shoulder_midpoint", "x": 0.1, "y": -0.3, "threshold": 0.25}}},
            manually_confirmed=True,
        )

    def test_main_pages_render(self):
        for url_name in ["landing", "library"]:
            response = self.client.get(reverse(url_name), HTTP_HOST="localhost")
            self.assertEqual(response.status_code, 200)

    def test_protected_pages_redirect_anonymous_users(self):
        protected_urls = [
            reverse("dashboard"),
            reverse("profile"),
            reverse("profile_edit"),
        ]
        for url in protected_urls:
            response = self.client.get(url, HTTP_HOST="localhost")
            self.assertEqual(response.status_code, 302)
            self.assertIn(reverse("login"), response.url)

        practice_response = self.client.post(
            reverse("start_practice", kwargs={"slug": self.lesson.slug}),
            HTTP_HOST="localhost",
        )
        self.assertEqual(practice_response.status_code, 302)
        self.assertIn(reverse("login"), practice_response.url)

    def test_signup_creates_profile_and_uses_email_login(self):
        response = self.client.post(
            reverse("signup"),
            {
                "display_name": "Pengguna Baru",
                "email": "baru@example.com",
                "password1": "testpass-12345",
                "password2": "testpass-12345",
            },
            HTTP_HOST="localhost",
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("translator"))
        user = User.objects.get(email="baru@example.com")
        self.assertEqual(user.username, "baru@example.com")
        self.assertTrue(UserProfile.objects.filter(user=user).exists())

    def test_login_redirects_returning_user_to_translator(self):
        response = self.client.post(
            reverse("login"),
            {"email": self.user.email, "password": "testpass-12345"},
            HTTP_HOST="localhost",
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("translator"))

    def test_translator_page_requires_login_and_renders(self):
        anonymous_response = self.client.get(reverse("translator"), HTTP_HOST="localhost")
        self.assertEqual(anonymous_response.status_code, 302)
        self.assertIn(reverse("login"), anonymous_response.url)

        self.client.force_login(self.user)
        response = self.client.get(reverse("translator"), HTTP_HOST="localhost")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "ISYARA Translator")
        self.assertContains(response, 'id="translatorCamera"')
        self.assertContains(response, "translator.js")

    def test_translation_history_is_user_scoped(self):
        own_item = TranslationHistory.objects.create(
            user=self.user,
            direction=TranslationHistory.DIRECTION_SIGN_TO_SPEECH,
            source_text="terima_kasih",
            translated_text="Terima kasih",
            confidence=0.92,
        )
        TranslationHistory.objects.create(
            user=self.other_user,
            direction=TranslationHistory.DIRECTION_SPEECH_TO_TEXT,
            translated_text="Apa kabar?",
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("translation_history"), HTTP_HOST="localhost")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, own_item.translated_text)
        self.assertNotContains(response, "Apa kabar?")

    def test_users_cannot_delete_other_users_history(self):
        other_item = TranslationHistory.objects.create(
            user=self.other_user,
            direction=TranslationHistory.DIRECTION_SPEECH_TO_TEXT,
            translated_text="Besok bertemu.",
        )
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("delete_translation_history", kwargs={"history_id": other_item.id}),
            HTTP_HOST="localhost",
        )
        self.assertEqual(response.status_code, 404)
        self.assertTrue(TranslationHistory.objects.filter(id=other_item.id).exists())

    def test_predict_sign_without_model_is_honest(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("api_predict_sign"),
            data={
                "image": SimpleUploadedFile(
                    "frame.jpg",
                    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xd9",
                    content_type="image/jpeg",
                )
            },
            HTTP_HOST="localhost",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "service_unavailable")
        self.assertIsNone(payload["prediction"])

    def test_predict_sign_requires_image_upload(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("api_predict_sign"),
            data=json.dumps({"sequence": [[0.0, 0.0, 0.0] for _ in range(30)]}),
            content_type="application/json",
            HTTP_HOST="localhost",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "error")

    def test_onboarding_unlocks_learning_path(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("learning_path"), HTTP_HOST="localhost")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("onboarding"))

        onboarding_response = self.client.post(
            reverse("onboarding"),
            {
                "learning_goal": "daily",
                "bisindo_familiarity": "new",
            },
            HTTP_HOST="localhost",
        )
        self.assertEqual(onboarding_response.status_code, 302)
        self.assertEqual(onboarding_response.url, reverse("learning_path"))
        self.user.profile.refresh_from_db()
        self.assertTrue(self.user.profile.onboarding_completed)

        path_response = self.client.get(reverse("learning_path"), HTTP_HOST="localhost")
        self.assertEqual(path_response.status_code, 200)
        self.assertContains(path_response, "Jalur Belajar")
        self.assertContains(path_response, "Halo")

    def test_practice_flow_creates_result_and_progress(self):
        self.client.force_login(self.user)
        start_response = self.client.post(
            reverse("start_practice", kwargs={"slug": self.lesson.slug}),
            HTTP_HOST="localhost",
        )

        self.assertEqual(start_response.status_code, 302)
        session = PracticeSession.objects.get()
        self.assertEqual(session.user, self.user)
        self.assertEqual(start_response.url, reverse("practice_camera", kwargs={"session_id": session.id}))

        camera_response = self.client.get(start_response.url, HTTP_HOST="localhost")
        self.assertEqual(camera_response.status_code, 200)
        self.assertContains(camera_response, 'id="landmarkOverlay"')
        self.assertContains(camera_response, 'referrerpolicy="strict-origin-when-cross-origin"')
        self.assertContains(camera_response, "Buka video referensi di YouTube")
        self.assertContains(camera_response, "Coba Pelacakan AI Lagi")
        self.assertContains(camera_response, 'id="trackingTelemetry"')
        self.assertContains(camera_response, 'id="aiReferenceConfig"')
        self.assertContains(camera_response, '"required_hands": ["right"]')
        self.assertContains(camera_response, "practice_overlay.js")
        self.assertContains(camera_response, "Video Referensi")
        self.assertEqual(camera_response.headers["Referrer-Policy"], "strict-origin-when-cross-origin")

        analyze_response = self.client.post(
            reverse("analyze_practice", kwargs={"session_id": session.id}),
            HTTP_HOST="localhost",
        )

        session.refresh_from_db()
        self.assertEqual(analyze_response.status_code, 302)
        self.assertEqual(session.status, PracticeSession.STATUS_ANALYZED)
        self.assertIsNotNone(session.score)
        self.assertEqual(Progress.objects.count(), 1)
        self.assertEqual(Progress.objects.get().user, self.user)
        self.assertTrue(Progress.objects.get().completed)
        self.assertEqual(Progress.objects.get().latest_score, session.score)
        self.assertIsNotNone(Progress.objects.get().completed_at)

        result_response = self.client.get(analyze_response.url, HTTP_HOST="localhost")
        self.assertEqual(result_response.status_code, 200)
        self.assertContains(result_response, "Hasil Latihan")

    def test_users_cannot_access_other_users_practice_sessions(self):
        session = PracticeSession.objects.create(lesson=self.lesson, user=self.user)
        self.client.force_login(self.other_user)

        camera_response = self.client.get(
            reverse("practice_camera", kwargs={"session_id": session.id}),
            HTTP_HOST="localhost",
        )
        result_response = self.client.get(
            reverse("practice_result", kwargs={"session_id": session.id}),
            HTTP_HOST="localhost",
        )
        analyze_response = self.client.post(
            reverse("analyze_practice", kwargs={"session_id": session.id}),
            HTTP_HOST="localhost",
        )

        self.assertEqual(camera_response.status_code, 404)
        self.assertEqual(result_response.status_code, 404)
        self.assertEqual(analyze_response.status_code, 404)
        self.assertEqual(Progress.objects.count(), 0)

    def test_youtube_video_id_parser_supports_common_formats(self):
        cases = {
            "https://www.youtube.com/watch?v=6JIg_tqs_vw": "6JIg_tqs_vw",
            "https://youtu.be/S-2Lj8OzPqQ?si=b77fZz6CIusyuVB1": "S-2Lj8OzPqQ",
            "https://www.youtube.com/shorts/GCFfwXFi6hA": "GCFfwXFi6hA",
        }
        for url, expected in cases.items():
            self.assertEqual(youtube_video_id_from_url(url), expected)
