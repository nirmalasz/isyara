import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.learning.models import AIReference, Lesson


POSE_INDEXES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}
FACE_INDEXES = [1, 4, 10, 13, 14, 33, 61, 105, 133, 152, 159, 263, 291, 334, 362, 386]


class Command(BaseCommand):
    help = "Build landmark data from a manually validated BISINDO reference video."

    def add_arguments(self, parser):
        parser.add_argument("lesson_slug")
        parser.add_argument("video_path")
        parser.add_argument("--fps", type=int, default=15)
        parser.add_argument("--required-hands", nargs="*", choices=["left", "right"], default=None)
        parser.add_argument("--active-hand", choices=["left", "right", "both"], default=None)
        parser.add_argument("--uses-face", action="store_true")
        parser.add_argument("--no-upper-body", action="store_true")
        parser.add_argument("--static", action="store_true")
        parser.add_argument("--confirm", action="store_true", help="Mark the generated AI reference metadata as manually confirmed.")

    def handle(self, *args, **options):
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise CommandError("OpenCV and MediaPipe are required to build AI references.") from exc

        lesson = Lesson.objects.filter(slug=options["lesson_slug"]).first()
        if not lesson:
            raise CommandError(f"Lesson not found: {options['lesson_slug']}")

        source_video = Path(options["video_path"]).expanduser()
        if not source_video.exists():
            raise CommandError(f"Reference video not found: {source_video}")

        output_dir = Path(settings.BASE_DIR) / "references" / lesson.slug
        output_dir.mkdir(parents=True, exist_ok=True)
        reference_copy_path = output_dir / "reference.mp4"
        if source_video.resolve() != reference_copy_path.resolve():
            reference_copy_path.write_bytes(source_video.read_bytes())

        frames, stats, duration = self._extract_frames(cv2, mp, reference_copy_path, options["fps"])
        suggested_required = self._suggest_required_hands(stats)
        required_hands = options["required_hands"] if options["required_hands"] is not None else suggested_required
        active_hand = options["active_hand"] or ("both" if len(required_hands) == 2 else (required_hands[0] if required_hands else "right"))
        uses_upper_body = not options["no_upper_body"]
        uses_face = options["uses_face"]
        is_dynamic = not options["static"]

        landmarks_path = output_dir / "landmarks.json"
        metadata_path = output_dir / "metadata.json"
        landmarks_payload = {"frames": frames}
        metadata_payload = {
            "sign": lesson.slug,
            "required_hands": required_hands,
            "active_hand": active_hand,
            "uses_face": uses_face,
            "uses_upper_body": uses_upper_body,
            "is_dynamic": is_dynamic,
            "fps": options["fps"],
            "duration": round(duration, 3),
            "detected_reference_structure": stats,
            "manual_confirmation_required": not options["confirm"],
        }
        landmarks_path.write_text(json.dumps(landmarks_payload, indent=2), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

        ai_reference, _ = AIReference.objects.update_or_create(
            lesson=lesson,
            defaults={
                "required_hands": required_hands,
                "active_hand": active_hand,
                "uses_face": uses_face,
                "uses_upper_body": uses_upper_body,
                "is_dynamic": is_dynamic,
                "region": lesson.region,
                "reference_video_path": str(reference_copy_path.relative_to(settings.BASE_DIR)),
                "reference_landmarks_path": str(landmarks_path.relative_to(settings.BASE_DIR)),
                "component_weights": self._default_weights(is_dynamic, uses_face),
                "reference_features": self._reference_features_from_frames(frames, active_hand),
                "manually_confirmed": options["confirm"],
            },
        )

        self.stdout.write(self.style.SUCCESS(f"Built AI reference for {lesson.slug}: {ai_reference.reference_landmarks_path}"))
        self.stdout.write(
            "Detected reference structure: "
            f"right={stats['right_hand_active_percent']}%, "
            f"left={stats['left_hand_active_percent']}%, "
            f"face={stats['face_visible_percent']}%, "
            f"upper_body={stats['upper_body_visible_percent']}%"
        )
        if not options["confirm"]:
            self.stdout.write(self.style.WARNING("Review metadata.json and rerun with --confirm or override flags before treating this as source of truth."))

    def _extract_frames(self, cv2, mp, video_path, target_fps):
        hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
        pose = mp.solutions.pose.Pose(static_image_mode=False)
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        capture = cv2.VideoCapture(str(video_path))
        native_fps = capture.get(cv2.CAP_PROP_FPS) or target_fps
        frame_interval = max(1, round(native_fps / target_fps))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / native_fps if native_fps else 0
        frames = []
        stats = {"right": 0, "left": 0, "face": 0, "upper_body": 0, "total": 0}
        frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_interval:
                frame_index += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_result = hands.process(rgb)
            pose_result = pose.process(rgb)
            face_result = face_mesh.process(rgb)
            frame_payload = {
                "timestamp": round(frame_index / native_fps, 4) if native_fps else 0,
                "right_hand": None,
                "left_hand": None,
                "pose": None,
                "face": None,
            }

            if hand_result.multi_hand_landmarks:
                for landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
                    label = handedness.classification[0].label.lower()
                    points = [self._point(landmark) for landmark in landmarks.landmark]
                    if label == "right":
                        frame_payload["right_hand"] = points
                        stats["right"] += 1
                    elif label == "left":
                        frame_payload["left_hand"] = points
                        stats["left"] += 1

            if pose_result.pose_landmarks:
                frame_payload["pose"] = {
                    name: self._point(pose_result.pose_landmarks.landmark[position])
                    for name, position in POSE_INDEXES.items()
                }
                stats["upper_body"] += 1

            if face_result.multi_face_landmarks:
                face_landmarks = face_result.multi_face_landmarks[0].landmark
                frame_payload["face"] = {
                    str(face_index): self._point(face_landmarks[face_index])
                    for face_index in FACE_INDEXES
                    if face_index < len(face_landmarks)
                }
                stats["face"] += 1

            frames.append(frame_payload)
            stats["total"] += 1
            frame_index += 1

        capture.release()
        hands.close()
        pose.close()
        face_mesh.close()
        total = max(stats["total"], 1)
        return frames, {
            "right_hand_active_percent": round((stats["right"] / total) * 100),
            "left_hand_active_percent": round((stats["left"] / total) * 100),
            "face_visible_percent": round((stats["face"] / total) * 100),
            "upper_body_visible_percent": round((stats["upper_body"] / total) * 100),
            "processed_frames": stats["total"],
        }, duration

    def _point(self, landmark):
        return {"x": landmark.x, "y": landmark.y, "z": landmark.z, "visibility": getattr(landmark, "visibility", None)}

    def _suggest_required_hands(self, stats):
        required = []
        if stats["right_hand_active_percent"] >= 45:
            required.append("right")
        if stats["left_hand_active_percent"] >= 45:
            required.append("left")
        return required or ["right"]

    def _default_weights(self, is_dynamic, uses_face):
        if not is_dynamic:
            return {"handshape": 0.50, "finger_configuration": 0.30, "palm_orientation": 0.20, "location": 0, "movement": 0, "timing": 0, "facial_expression": 0}
        weights = {"handshape": 0.20, "finger_configuration": 0.15, "location": 0.20, "palm_orientation": 0.10, "movement": 0.25, "timing": 0.10, "facial_expression": 0}
        if uses_face:
            weights["facial_expression"] = 0.10
            weights["movement"] = 0.20
            weights["timing"] = 0.05
        return weights

    def _reference_features_from_frames(self, frames, active_hand):
        hand_key = "right_hand" if active_hand != "left" else "left_hand"
        frame = next((item for item in frames if item.get(hand_key)), None)
        if not frame:
            return {}
        return {
            active_hand if active_hand in {"left", "right"} else "right": {
                "location": {"anchor": "shoulder_midpoint", "x": 0.0, "y": -0.25, "threshold": 0.30},
                "finger_angles": {},
                "palm_orientation": {"wrist_index_angle": -1.1, "threshold": 0.85},
            }
        }
