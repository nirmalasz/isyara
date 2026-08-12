from django.db.models import Count

from .models import LearningModule, Lesson, Progress


def guided_modules():
    return LearningModule.objects.filter(is_active=True).prefetch_related("lessons__sign").order_by("order", "title")


def progress_map_for(user):
    if not user.is_authenticated:
        return {}
    return {
        record.lesson_id: record
        for record in Progress.objects.filter(user=user).select_related("lesson", "lesson__module")
    }


def build_learning_path(user):
    progress_map = progress_map_for(user)
    modules = list(guided_modules())
    path = []
    previous_module_complete = True
    first_unfinished = None

    for module_index, module in enumerate(modules):
        lessons = list(module.lessons.all().order_by("order", "title"))
        completed_count = sum(1 for lesson in lessons if progress_map.get(lesson.id) and progress_map[lesson.id].completed)
        module_unlocked = previous_module_complete
        lesson_items = []
        previous_lesson_complete = True

        for index, lesson in enumerate(lessons, start=1):
            record = progress_map.get(lesson.id)
            completed = bool(record and record.completed)
            unlocked = module_unlocked and previous_lesson_complete
            if completed:
                status = "completed"
                status_label = "Selesai"
            elif unlocked:
                status = "current"
                status_label = "Saat Ini"
                if first_unfinished is None:
                    first_unfinished = lesson
            else:
                status = "locked"
                status_label = "Terkunci"

            lesson_items.append(
                {
                    "lesson": lesson,
                    "number": index,
                    "record": record,
                    "completed": completed,
                    "unlocked": unlocked,
                    "status": status,
                    "status_label": status_label,
                    "locked_reason": "Selesaikan pelajaran sebelumnya untuk membuka materi ini.",
                }
            )
            previous_lesson_complete = completed

        checkpoint_unlocked = module_unlocked and bool(lessons) and completed_count == len(lessons)
        checkpoint_completed = checkpoint_unlocked
        if checkpoint_unlocked and first_unfinished is None:
            next_module = modules[module_index + 1] if module_index + 1 < len(modules) else None
            first_unfinished = next_module.lessons.order_by("order", "title").first() if next_module else None

        path.append(
            {
                "module": module,
                "lessons": lesson_items,
                "total_lessons": len(lessons),
                "completed_count": completed_count,
                "completion_percent": round((completed_count / len(lessons)) * 100) if lessons else 0,
                "unlocked": module_unlocked,
                "checkpoint_unlocked": checkpoint_unlocked,
                "checkpoint_completed": checkpoint_completed,
            }
        )
        previous_module_complete = checkpoint_completed

    return {"modules": path, "next_lesson": first_unfinished}


def lesson_state(user, lesson):
    path = build_learning_path(user)
    for module_item in path["modules"]:
        for lesson_item in module_item["lessons"]:
            if lesson_item["lesson"].id == lesson.id:
                return module_item, lesson_item
    return None, {"unlocked": True, "status": "current", "status_label": "Saat Ini", "number": lesson.order}


def learning_summary(user):
    progress = Progress.objects.filter(user=user)
    completed_lessons = progress.filter(completed=True).count()
    current_module = None
    path = build_learning_path(user)
    for module_item in path["modules"]:
        if module_item["unlocked"] and module_item["completed_count"] < module_item["total_lessons"]:
            current_module = module_item["module"]
            break
    return {
        "completed_lessons": completed_lessons,
        "total_practices": sum(record.attempts for record in progress),
        "current_module": current_module,
        "module_count": LearningModule.objects.filter(is_active=True).count(),
        "module_activity": progress.values("lesson__module__title").annotate(total=Count("id")),
    }
