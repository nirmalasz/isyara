from django.core.management.base import BaseCommand
from django.utils.text import slugify

from apps.learning.models import AIReference, Lesson, LessonPart, Sign


VOCABULARY = [
    ("Halo", "Sapaan ramah untuk memulai percakapan.", "https://youtu.be/GCFfwXFi6hA?si=hzy9N0q5d7Zddg5h", 0),
    ("Terima Kasih", "Ungkapan sopan untuk menyampaikan rasa terima kasih.", "https://youtu.be/S-2Lj8OzPqQ?si=b77fZz6CIusyuVB1", 0),
    ("Maaf", "Isyarat untuk meminta maaf atau memohon pengertian."),
    ("Nama", "Digunakan saat memperkenalkan nama."),
    ("Saya", "Kata ganti orang pertama untuk merujuk diri sendiri."),
    ("Kamu", "Kata ganti orang kedua saat menyapa lawan bicara."),
    ("Aku", "Kata ganti orang pertama informal untuk merujuk diri sendiri."),
    ("Dia", "Kata ganti orang ketiga tunggal untuk merujuk seseorang."),
    ("Kami", "Kata ganti orang pertama jamak, tidak termasuk lawan bicara."),
    ("Kalian", "Kata ganti orang kedua jamak saat menyapa lebih dari satu orang."),
    ("Makan", "Isyarat aktivitas harian untuk makan."),
    ("Minum", "Isyarat aktivitas harian untuk minum."),
    ("Belajar", "Isyarat yang berkaitan dengan belajar dan studi."),
    ("Sekolah", "Isyarat tempat untuk sekolah."),
    ("Selamat Pagi", "Sapaan pagi untuk memulai hari dengan sopan.", "https://youtu.be/WP6pSzGk1fM?si=CD29AmXvv3cP6k4F", 0),
    ("Selamat Siang", "Sapaan umum untuk waktu tengah hari hingga sore."),
    ("Selamat Sore", "Sapaan untuk menyapa di waktu sore hari."),
    ("Selamat Malam", "Sapaan untuk menyapa atau berpamitan di malam hari."),
    ("Angka", "Pelajaran ringkasan angka dasar dalam BISINDO."),
]

ALPHABET_OVERVIEW = (
    "Abjad",
    "Pelajaran ringkasan alfabet BISINDO. Video kurasi dimulai dari 0:40 dan dibagi per huruf untuk penilaian per bagian nanti.",
    "https://youtu.be/6JIg_tqs_vw?si=ITuP4gzXzFzYXaMw",
    40,
)

DEFAULT_DYNAMIC_WEIGHTS = {
    "handshape": 0.20,
    "finger_configuration": 0.15,
    "location": 0.20,
    "palm_orientation": 0.10,
    "movement": 0.25,
    "timing": 0.10,
    "facial_expression": 0.0,
}

AI_REFERENCE_CONFIGS = {
    "halo": {
        "required_hands": ["right"],
        "active_hand": "right",
        "uses_face": False,
        "uses_upper_body": True,
        "is_dynamic": True,
        "component_weights": DEFAULT_DYNAMIC_WEIGHTS,
        "reference_features": {
            "right": {
                "location": {"anchor": "shoulder_midpoint", "x": 0.18, "y": -0.48, "threshold": 0.22},
                "finger_angles": {"index_pip": 165, "middle_pip": 160, "ring_pip": 150},
                "palm_orientation": {"wrist_index_angle": -1.25, "threshold": 0.75},
            }
        },
    },
    "terima-kasih": {
        "required_hands": ["left", "right"],
        "active_hand": "both",
        "uses_face": False,
        "uses_upper_body": True,
        "is_dynamic": True,
        "component_weights": {
            "handshape": 0.20,
            "finger_configuration": 0.15,
            "location": 0.20,
            "palm_orientation": 0.10,
            "movement": 0.25,
            "timing": 0.10,
            "facial_expression": 0.0,
        },
        "reference_features": {
            "right": {
                "location": {"anchor": "chin", "x": 0.08, "y": 0.12, "threshold": 0.24},
                "finger_angles": {"index_pip": 168, "middle_pip": 166},
                "palm_orientation": {"wrist_index_angle": -0.85, "threshold": 0.75},
            },
            "left": {
                "location": {"anchor": "shoulder_midpoint", "x": -0.18, "y": -0.18, "threshold": 0.28},
                "finger_angles": {"index_pip": 165, "middle_pip": 164},
                "palm_orientation": {"wrist_index_angle": -2.2, "threshold": 0.85},
            },
        },
    },
    "selamat-pagi": {
        "required_hands": ["right"],
        "active_hand": "right",
        "uses_face": True,
        "uses_upper_body": True,
        "is_dynamic": True,
        "component_weights": {
            "handshape": 0.18,
            "finger_configuration": 0.12,
            "location": 0.18,
            "palm_orientation": 0.10,
            "movement": 0.22,
            "timing": 0.10,
            "facial_expression": 0.10,
        },
        "reference_features": {
            "right": {
                "location": {"anchor": "forehead", "x": 0.16, "y": -0.06, "threshold": 0.24},
                "finger_angles": {"index_pip": 160, "middle_pip": 158},
                "palm_orientation": {"wrist_index_angle": -1.0, "threshold": 0.8},
            },
            "face": {"mouth_openness": {"target": 0.04, "threshold": 0.08}},
        },
    },
    "abjad": {
        "required_hands": ["right"],
        "active_hand": "right",
        "uses_face": False,
        "uses_upper_body": True,
        "is_dynamic": False,
        "component_weights": {
            "handshape": 0.50,
            "finger_configuration": 0.30,
            "palm_orientation": 0.20,
            "location": 0.0,
            "movement": 0.0,
            "timing": 0.0,
            "facial_expression": 0.0,
        },
        "reference_features": {
            "right": {
                "location": {"anchor": "shoulder_midpoint", "x": 0.20, "y": -0.28, "threshold": 0.35},
                "finger_angles": {"index_pip": 170, "middle_pip": 165},
                "palm_orientation": {"wrist_index_angle": -1.1, "threshold": 0.9},
            }
        },
    },
}


class Command(BaseCommand):
    help = "Seed placeholder BISINDO vocabulary and alphabet lessons."

    def handle(self, *args, **options):
        created = 0
        order = 1

        for item in VOCABULARY:
            title, description, youtube_url, start_seconds = self._normalize_seed_item(item)
            created += self._upsert_sign_and_lesson(
                title=title,
                category=Sign.CATEGORY_VOCABULARY,
                description=description,
                difficulty=1 if order <= 5 else 2,
                order=order,
                youtube_url=youtube_url,
                video_start_seconds=start_seconds,
                ai_practice_available=True,
            )
            order += 1

        title, description, youtube_url, start_seconds = ALPHABET_OVERVIEW
        abjad_lesson, was_created = self._upsert_sign_and_lesson(
            title=title,
            category=Sign.CATEGORY_ALPHABET,
            description=description,
            difficulty=1,
            order=order,
            youtube_url=youtube_url,
            video_start_seconds=start_seconds,
            ai_practice_available=True,
            return_lesson=True,
        )
        created += 1 if was_created else 0
        self._upsert_alphabet_parts(abjad_lesson)
        order += 1

        stale_localized_letter_slugs = [f"huruf-{letter.lower()}" for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        Sign.objects.filter(slug__in=stale_localized_letter_slugs).delete()
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            created += self._upsert_sign_and_lesson(
                title=f"Huruf {letter}",
                category=Sign.CATEGORY_ALPHABET,
                description=f"Entri alfabet BISINDO sementara untuk huruf {letter}.",
                difficulty=1,
                order=order,
                youtube_url="",
                video_start_seconds=0,
                ai_practice_available=True,
                slug_override=f"letter-{letter.lower()}",
            )
            order += 1

        self._upsert_ai_references()

        self.stdout.write(self.style.SUCCESS(f"Seed complete. {created} lessons created or updated."))

    def _normalize_seed_item(self, item):
        if len(item) == 2:
            title, description = item
            return title, description, "", 0
        return item

    def _upsert_sign_and_lesson(
        self,
        title,
        category,
        description,
        difficulty,
        order,
        youtube_url="",
        video_start_seconds=0,
        ai_practice_available=True,
        return_lesson=False,
        slug_override=None,
    ):
        sign_slug = slug_override or slugify(title)
        sign, _ = Sign.objects.update_or_create(
            slug=sign_slug,
            defaults={
                "title": title,
                "category": category,
                "description": description,
                "difficulty": difficulty,
                "image_hint": f"Referensi visual sementara untuk {title}",
            },
        )
        lesson, created = Lesson.objects.update_or_create(
            slug=sign_slug,
            defaults={
                "sign": sign,
                "title": title,
                "summary": description,
                "instruction": (
                    "Posisikan tubuh dengan jelas di depan kamera, amati bentuk referensi, "
                    "lalu lakukan isyarat dengan gerakan tangan stabil dan pose akhir yang tegas."
                ),
                "youtube_url": youtube_url,
                "video_start_seconds": video_start_seconds,
                "source_name": "YouTube" if youtube_url else "",
                "source_channel": "",
                "region": "BISINDO",
                "ai_practice_available": ai_practice_available,
                "order": order,
                "estimated_minutes": 4 if category == Sign.CATEGORY_ALPHABET else 6,
            },
        )
        if return_lesson:
            return lesson, created
        return 1 if created else 0

    def _upsert_alphabet_parts(self, lesson):
        for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", start=1):
            start_seconds = 40 + ((index - 1) * 5)
            LessonPart.objects.update_or_create(
                lesson=lesson,
                order=index,
                defaults={
                    "title": f"Huruf {letter}",
                    "video_start_seconds": start_seconds,
                    "video_end_seconds": start_seconds + 5,
                    "scoring_weight": 100,
                    "ai_practice_available": False,
                    "instruction": f"Tonton dan latih huruf {letter}. Penilaian AI per bagian akan diaktifkan setelah referensi tervalidasi ditambahkan.",
                },
            )

    def _upsert_ai_references(self):
        for slug, config in AI_REFERENCE_CONFIGS.items():
            lesson = Lesson.objects.filter(slug=slug).first()
            if not lesson:
                continue
            AIReference.objects.update_or_create(
                lesson=lesson,
                defaults={
                    "required_hands": config["required_hands"],
                    "active_hand": config["active_hand"],
                    "uses_face": config["uses_face"],
                    "uses_upper_body": config["uses_upper_body"],
                    "is_dynamic": config["is_dynamic"],
                    "region": "BISINDO",
                    "reference_video_path": f"references/{slug}/reference.mp4",
                    "reference_landmarks_path": f"references/{slug}/landmarks.json",
                    "component_weights": config["component_weights"],
                    "reference_features": config["reference_features"],
                    "manually_confirmed": True,
                },
            )
