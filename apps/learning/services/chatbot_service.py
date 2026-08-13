import os
import re
from groq import Groq
from rapidfuzz import fuzz, process
from apps.learning.models import Lesson

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """Kamu adalah asisten BISINDO bernama ISYAR-AI.

ATURAN KETAT — WAJIB DIIKUTI:
1. Kamu HANYA boleh menjelaskan isyarat yang informasinya diberikan ke kamu di pesan user. Jangan pernah menyebut kata "data lesson", "konteks", atau istilah teknis lain — jawab natural seolah kamu memang tahu informasi itu.
2. Jangan PERNAH menjelaskan cara memperagakan isyarat lain yang tidak diberikan informasinya, meskipun kamu merasa tahu jawabannya.
3. Jangan pernah mengarang deskripsi gerakan tangan sendiri. Gunakan hanya deskripsi yang diberikan ke kamu.
4. PENTING: Jika kamu menerima "Informasi kosakata" dengan Judul, Deskripsi, dan Video referensi — itu artinya isyarat TERSEDIA dan kamu HARUS menjawab dengan percaya diri bahwa isyarat itu ADA, bukan bilang "belum ada" atau "maaf". Langsung jelaskan isyaratnya dan sebutkan link videonya.
5. Kamu HANYA boleh bilang "belum tersedia di database" jika kamu menerima pesan "Informasi: tidak ditemukan kosakata yang cocok di database" — HANYA dalam kondisi itu saja.
6. Jangan pernah minta maaf ("maaf") kecuali benar-benar dalam kondisi aturan 5 di atas.
7. Jawab singkat (maksimal 3-4 kalimat), ramah, dan percaya diri dalam Bahasa Indonesia santai tapi sopan.
8. Jangan gunakan markdown, bullet points, atau format tebal — jawab sebagai teks percakapan biasa.

Kamu TIDAK BOLEH menjawab pertanyaan di luar topik BISINDO/bahasa isyarat/aplikasi ISYARA."""


LETTER_PATTERN = re.compile(r"\b(?:huruf\s+)?([a-zA-Z])\b")
NUMBER_PATTERN = re.compile(r"\b(?:angka\s+)?(\d+)\b")


def detect_alphabet_or_number_intent(user_query: str):
    """Deteksi apakah query soal huruf tunggal atau angka, arahkan ke lesson khusus."""
    query_lower = user_query.lower().strip()

    number_match = NUMBER_PATTERN.search(query_lower)
    if number_match:
        return "Angka"

    letter_keywords = ["huruf", "abjad", "alfabet", "eja", "mengeja"]
    if any(keyword in query_lower for keyword in letter_keywords):
        return "Abjad"

    stripped = query_lower.replace("huruf", "").strip()
    if len(stripped) == 1 and stripped.isalpha():
        return "Abjad"

    return None


def find_matching_lesson(user_query: str, score_cutoff: int = 55):
    lessons = list(Lesson.objects.all())
    if not lessons:
        return None, 0

    choices = {lesson.title: lesson for lesson in lessons}

    forced_title = detect_alphabet_or_number_intent(user_query)
    if forced_title and forced_title in choices:
        return choices[forced_title], 100

    best_match = process.extractOne(
        user_query,
        choices.keys(),
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff,
    )

    if best_match:
        matched_title, score, _ = best_match
        return choices[matched_title], score
    return None, 0


def build_lesson_context(lesson: Lesson) -> str:
    if lesson is None:
        return "Informasi: tidak ditemukan kosakata yang cocok di database"

    video_status = f"Tersedia: {lesson.youtube_url}" if lesson.youtube_url else "Belum tersedia"

    return f"""Informasi kosakata:
Judul: {lesson.title}
Kategori: {lesson.get_category_display() if hasattr(lesson, 'get_category_display') else lesson.sign.category}
Deskripsi: {lesson.summary}
Video referensi: {video_status}
Estimasi durasi belajar: {lesson.estimated_minutes} menit"""


def generate_chatbot_reply(user_query: str) -> dict:
    lesson, score = find_matching_lesson(user_query)
    context = build_lesson_context(lesson)

    user_message = f"""{context}

Pertanyaan user: {user_query}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=250,
        )
        reply_text = response.choices[0].message.content
    except Exception as e:
        reply_text = "Maaf, chatbot sedang bermasalah. Coba lagi sebentar lagi."
        print(f"Groq API error: {e}")

    return {
        "reply": reply_text,
        "matched_lesson": lesson.title if lesson else None,
        "lesson_slug": lesson.slug if lesson else None,
        "video_url": lesson.youtube_url if lesson else None,
        "match_score": score,
    }