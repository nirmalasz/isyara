import os

from .models import Lesson


SYSTEM_PROMPT = """Kamu adalah asisten BISINDO bernama ISYAR-AI.

ATURAN KETAT:
1. Jawab hanya tentang BISINDO, bahasa isyarat, atau aplikasi ISYARA.
2. Jangan mengarang gerakan. Gunakan informasi kosakata dari database.
3. Jika kosakata ditemukan, jawab dengan percaya diri dan singkat.
4. Jika kosakata tidak ditemukan, bilang belum tersedia di database.
5. Jawab maksimal 3-4 kalimat, ramah, natural, dan dalam Bahasa Indonesia.
6. Jangan gunakan markdown atau bullet points."""


def _normalized(value):
    return (value or "").strip().lower()


def _detect_alphabet_or_number_intent(user_query):
    query = _normalized(user_query)
    if any(keyword in query for keyword in ("huruf", "abjad", "alfabet", "eja", "mengeja")):
        return "Abjad"
    if any(keyword in query for keyword in ("angka", "nomor", "bilangan")):
        return "Angka"
    if len(query) == 1 and query.isalpha():
        return "Abjad"
    return None


def _score_lesson(user_query, lesson):
    query = _normalized(user_query)
    title = _normalized(lesson.title)
    sign_title = _normalized(getattr(lesson.sign, "title", ""))
    slug = _normalized(lesson.slug).replace("-", " ")

    if not query:
        return 0
    if query == title or query == sign_title:
        return 100
    if title and title in query:
        return 95
    if sign_title and sign_title in query:
        return 95
    if slug and slug in query:
        return 90

    try:
        from rapidfuzz import fuzz

        return max(
            fuzz.WRatio(query, title),
            fuzz.WRatio(query, sign_title),
            fuzz.WRatio(query, slug),
        )
    except ImportError:
        query_words = set(query.split())
        lesson_words = set(" ".join([title, sign_title, slug]).split())
        if not query_words or not lesson_words:
            return 0
        return round(100 * len(query_words & lesson_words) / len(query_words | lesson_words))


def find_matching_lesson(user_query, score_cutoff=55):
    lessons = list(Lesson.objects.select_related("sign").all())
    if not lessons:
        return None, 0

    forced_title = _detect_alphabet_or_number_intent(user_query)
    if forced_title:
        forced = next((lesson for lesson in lessons if lesson.title.lower() == forced_title.lower()), None)
        if forced:
            return forced, 100

    scored = [(lesson, _score_lesson(user_query, lesson)) for lesson in lessons]
    lesson, score = max(scored, key=lambda item: item[1])
    if score >= score_cutoff:
        return lesson, score
    return None, score


def build_lesson_context(lesson):
    if lesson is None:
        return "Informasi: tidak ditemukan kosakata yang cocok di database"

    category = lesson.sign.get_category_display() if getattr(lesson, "sign", None) else "-"
    video_status = f"Tersedia: {lesson.youtube_url}" if lesson.youtube_url else "Belum tersedia"
    return f"""Informasi kosakata:
Judul: {lesson.title}
Kategori: {category}
Deskripsi: {lesson.summary}
Instruksi: {lesson.instruction}
Video referensi: {video_status}
Estimasi durasi belajar: {lesson.estimated_minutes} menit"""


def _fallback_reply(user_query, lesson):
    if lesson is None:
        return "Kosakata itu belum tersedia di database ISYARA. Coba tanya kata BISINDO lain yang ada di library."

    video_text = f" Kamu bisa lihat video referensinya di {lesson.youtube_url}." if lesson.youtube_url else ""
    summary = lesson.summary.strip() or lesson.instruction.strip()
    return f"Isyarat {lesson.title} tersedia di ISYARA. {summary}{video_text}"


def _generate_with_groq(user_query, context):
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from groq import Groq
    except ImportError:
        return None

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nPertanyaan user: {user_query}"},
        ],
        temperature=0.4,
        max_tokens=250,
    )
    return response.choices[0].message.content


def generate_chatbot_reply(user_query):
    lesson, score = find_matching_lesson(user_query)
    context = build_lesson_context(lesson)

    try:
        reply_text = _generate_with_groq(user_query, context) or _fallback_reply(user_query, lesson)
    except Exception as exc:
        print(f"ISYAR-AI chatbot error: {exc}")
        reply_text = _fallback_reply(user_query, lesson)

    return {
        "reply": reply_text,
        "matched_lesson": lesson.title if lesson else None,
        "lesson_slug": lesson.slug if lesson else None,
        "video_url": lesson.youtube_url if lesson else None,
        "match_score": score,
    }
