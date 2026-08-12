import asyncio
import json
from pathlib import Path
import unittest

from services.assessment_service.assessment_service import main
from services.assessment_service.assessment_service.config import BISINDO_CLASSIFIER_MODEL_PATH, BISINDO_YOLO_MODEL_PATH
from services.assessment_service.assessment_service.inference.bisindo_classifier import BisindoYoloClassifier
from services.assessment_service.assessment_service.inference.calibration import BisindoCalibration
from services.assessment_service.assessment_service.inference.bisindo_detector import BisindoYoloDetector, DetectionResult, clean_bisindo_label
from services.assessment_service.assessment_service.inference.stabilization import PredictionStabilizer
from services.assessment_service.assessment_service.inference.structured_recognition import TerbisaStructureRecognizer
from services.assessment_service.assessment_service.inference.temporal_smoothing import TemporalProbabilitySmoother


TINY_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xd9"


class FakeUpload:
    def __init__(self, content_type="image/jpeg", data=TINY_JPEG):
        self.content_type = content_type
        self._data = data

    async def read(self):
        return self._data


class FakeDetector:
    def __init__(self, result):
        self.result = result
        self.model_available = True

    def model_info(self):
        return {
            "status": "ready",
            "model_available": True,
            "model_path": "fake.pt",
            "class_count": 1,
            "classes": ["Saya -BISINDO-"],
            "display_classes": ["Saya"],
            "model_names": {0: "Saya -BISINDO-"},
            "device": "cpu",
            "confidence_threshold": 0.65,
            "image_size": 640,
            "error": None,
        }

    def predict(self, image_bytes):
        return self.result


class FakeSequenceDetector(FakeDetector):
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0
        self.model_available = True

    def predict(self, image_bytes):
        result = self.results[self.calls]
        self.calls += 1
        return result


class AssessmentServiceTests(unittest.TestCase):
    def setUp(self):
        main.prediction_smoother.reset()

    def test_health_reports_ai_capabilities(self):
        payload = main.health()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("bisindo_yolo_model_available", payload)
        self.assertIn("bisindo_yolo_class_count", payload)

    def test_model_loading_reads_names_when_weights_exist(self):
        if not BISINDO_YOLO_MODEL_PATH.exists():
            self.skipTest("BISINDO YOLO weights are not available locally.")
        detector = BisindoYoloDetector(model_path=BISINDO_YOLO_MODEL_PATH)
        detector.load()
        self.assertTrue(detector.model_available)
        self.assertEqual(len(detector.names), 19)
        self.assertEqual(detector.names[0], "Anda -BISINDO-")
        self.assertEqual(detector.names[18], "Terima kasih -BISINDO-")
        self.assertEqual(detector.display_names[10], "Mau")

    def test_classifier_loading_reads_names_when_weights_exist(self):
        if not BISINDO_CLASSIFIER_MODEL_PATH.exists():
            self.skipTest("BISINDO classifier weights are not available locally.")
        classifier = BisindoYoloClassifier(model_path=BISINDO_CLASSIFIER_MODEL_PATH)
        classifier.load()
        self.assertTrue(classifier.model_available)
        self.assertEqual(len(classifier.names), 19)
        self.assertIn("Halo", classifier.names.values())
        self.assertIn("Terima-kasih", classifier.names.values())

    def test_missing_model_reports_unavailable(self):
        detector = BisindoYoloDetector(model_path=Path("/tmp/isyara-missing-model.pt"))
        info = detector.model_info()
        self.assertEqual(info["status"], "model_unavailable")
        self.assertFalse(info["model_available"])

    def test_model_info_schema(self):
        info = main.model_info()
        for key in ["status", "inference_mode", "active", "detector", "classifier"]:
            self.assertIn(key, info)
        self.assertIn(info["inference_mode"], {"classifier", "detector"})
        self.assertIn("model", info["active"])
        self.assertIn("task", info["active"])

    def test_predict_sign_rejects_non_image_upload(self):
        response = asyncio.run(main.predict_sign(FakeUpload(content_type="text/plain", data=b"hello")))
        self.assertEqual(response.status, "invalid_image")
        self.assertFalse(response.detected)

    def test_predict_sign_no_detection_schema(self):
        original = main.bisindo_detector
        original_classifier = main.bisindo_classifier
        original_stabilizer = main.prediction_stabilizer
        fake = FakeDetector(
            DetectionResult(
                status="no_detection",
                detected=False,
                class_id=None,
                raw_label=None,
                display_label=None,
                confidence=None,
            )
        )
        main.bisindo_detector = fake
        main.bisindo_classifier = fake
        main.prediction_stabilizer = PredictionStabilizer(window=4, stable_count=3, release_window=6)
        try:
            response = asyncio.run(main.predict_sign(FakeUpload(), frame_id="test-1", mirrored=True))
        finally:
            main.bisindo_detector = original
            main.bisindo_classifier = original_classifier
            main.prediction_stabilizer = original_stabilizer
        self.assertEqual(response.status, "no_detection")
        self.assertFalse(response.detected)
        self.assertIsNone(response.raw_label)
        self.assertIsNone(response.display_label)
        self.assertFalse(response.stable)
        self.assertFalse(response.suppressed)
        self.assertEqual(response.frame_id, "test-1")

    def test_roi_bbox_mapping_offsets_crop_coordinates(self):
        bbox = {"x1": 10, "y1": 20, "x2": 80, "y2": 90}
        roi = {"x1": 120, "y1": 45, "x2": 320, "y2": 245, "source_width": 640, "source_height": 480}
        mapped = main._map_roi_bbox_to_source(bbox, roi)
        self.assertEqual(mapped, {"x1": 130, "y1": 65, "x2": 200, "y2": 135})

    def test_predict_sign_maps_roi_bbox_to_source_frame(self):
        original = main.bisindo_detector
        original_classifier = main.bisindo_classifier
        original_stabilizer = main.prediction_stabilizer
        fake = FakeDetector(
            DetectionResult(
                status="ok",
                detected=True,
                class_id=14,
                raw_label="Saya -BISINDO-",
                display_label="Saya",
                confidence=0.99,
                raw_predictions=[{"class_id": 14, "raw_label": "Saya -BISINDO-", "label": "Saya", "confidence": 0.99}],
                detections=1,
                threshold_detections=1,
                valid_detections=1,
                bbox={"x1": 10, "y1": 20, "x2": 80, "y2": 90},
                image_width=200,
                image_height=200,
            )
        )
        main.bisindo_detector = fake
        main.bisindo_classifier = fake
        main.prediction_stabilizer = PredictionStabilizer(window=4, stable_count=3, release_window=6)
        try:
            response = asyncio.run(
                main.predict_sign(
                    FakeUpload(),
                    frame_id="roi-1",
                    mirrored=False,
                    roi_x1=120,
                    roi_y1=45,
                    roi_x2=320,
                    roi_y2=245,
                    source_width=640,
                    source_height=480,
                    hands_detected=2,
                    handedness="Left,Right",
                )
            )
        finally:
            main.bisindo_detector = original
            main.bisindo_classifier = original_classifier
            main.prediction_stabilizer = original_stabilizer
        self.assertEqual(response.bbox, {"x1": 130, "y1": 65, "x2": 200, "y2": 135})
        self.assertEqual(response.roi["source_width"], 640)
        self.assertEqual(response.image_width, 640)
        self.assertEqual(response.image_height, 480)
        self.assertEqual(response.hands_detected, 2)
        self.assertEqual(response.handedness, ["Left", "Right"])

    def test_predict_sign_selects_best_multi_candidate_once(self):
        original = main.bisindo_detector
        original_classifier = main.bisindo_classifier
        original_stabilizer = main.prediction_stabilizer
        fake = FakeSequenceDetector(
            [
                DetectionResult(
                    status="ok",
                    detected=True,
                    class_id=5,
                    raw_label="Halo -BISINDO-",
                    display_label="Halo",
                    confidence=0.97,
                    raw_predictions=[{"class_id": 5, "raw_label": "Halo -BISINDO-", "label": "Halo", "confidence": 0.97}],
                    detections=1,
                    threshold_detections=1,
                    valid_detections=1,
                    bbox={"x1": 5, "y1": 10, "x2": 70, "y2": 90},
                ),
                DetectionResult(
                    status="ok",
                    detected=True,
                    class_id=5,
                    raw_label="Halo -BISINDO-",
                    display_label="Halo",
                    confidence=0.99,
                    raw_predictions=[{"class_id": 5, "raw_label": "Halo -BISINDO-", "label": "Halo", "confidence": 0.99}],
                    detections=1,
                    threshold_detections=1,
                    valid_detections=1,
                    bbox={"x1": 8, "y1": 12, "x2": 80, "y2": 95},
                ),
                DetectionResult(
                    status="low_confidence",
                    detected=False,
                    class_id=None,
                    raw_label=None,
                    display_label=None,
                    confidence=0.38,
                    raw_predictions=[{"class_id": 6, "raw_label": "Hati-hati -BISINDO-", "label": "Hati-hati", "confidence": 0.38}],
                    detections=1,
                    threshold_detections=0,
                    valid_detections=1,
                    rejection_reason="below_confidence_threshold",
                ),
            ]
        )
        main.bisindo_detector = fake
        main.bisindo_classifier = fake
        main.prediction_stabilizer = PredictionStabilizer(window=1, stable_count=1, release_window=3, min_average_confidence=0.65)
        candidates_json = json.dumps(
            [
                {"type": "left", "x1": 100, "y1": 40, "x2": 240, "y2": 200, "source_width": 640, "source_height": 480},
                {"type": "right", "x1": 360, "y1": 40, "x2": 500, "y2": 200, "source_width": 640, "source_height": 480},
                {"type": "combined", "x1": 100, "y1": 40, "x2": 500, "y2": 220, "source_width": 640, "source_height": 480},
            ]
        )
        try:
            response = asyncio.run(
                main.predict_sign(
                    image=None,
                    candidates=[FakeUpload(), FakeUpload(), FakeUpload()],
                    frame_id="multi-1",
                    mirrored=False,
                    hands_detected=2,
                    handedness="Left,Right",
                    candidates_json=candidates_json,
                )
            )
        finally:
            main.bisindo_detector = original
            main.bisindo_classifier = original_classifier
            main.prediction_stabilizer = original_stabilizer
        self.assertTrue(response.accepted)
        self.assertEqual(response.accepted_prediction, "Halo")
        self.assertTrue(response.transcript_append_expected)
        self.assertEqual(response.roi_type, "right")
        self.assertEqual(response.selected_candidate["roi_type"], "right")
        self.assertEqual(len(response.candidate_predictions), 3)
        self.assertEqual(response.bbox, {"x1": 368, "y1": 52, "x2": 440, "y2": 135})

    def test_label_mapping_removes_dataset_suffix(self):
        self.assertEqual(clean_bisindo_label("Anda -BISINDO-"), "Anda")
        self.assertEqual(clean_bisindo_label("Apa -BISINDO-"), "Apa")
        self.assertEqual(clean_bisindo_label("Mau-Ingin -BISINDO-"), "Mau")
        self.assertEqual(clean_bisindo_label("Terima kasih -BISINDO-"), "Terima kasih")

    def test_detector_threshold_is_configurable(self):
        detector = BisindoYoloDetector(model_path=Path("/tmp/isyara-missing-model.pt"), confidence_threshold=0.82)
        self.assertEqual(detector.confidence_threshold, 0.82)

    def test_stabilization_requires_repeated_label(self):
        stabilizer = PredictionStabilizer(window=4, stable_count=3, release_window=6, min_average_confidence=0.65)
        self.assertFalse(stabilizer.accept("Saya", 0.91))
        self.assertFalse(stabilizer.accept("Makan", 0.93))
        self.assertFalse(stabilizer.accept("Saya", 0.92))
        self.assertTrue(stabilizer.accept("Saya", 0.94))

    def test_stabilization_rejects_mixed_window(self):
        stabilizer = PredictionStabilizer(window=4, stable_count=3, release_window=6, min_average_confidence=0.65)
        self.assertFalse(stabilizer.accept("Saya", 0.91))
        self.assertFalse(stabilizer.accept("Makan", 0.90))
        self.assertFalse(stabilizer.accept("Saya", 0.92))
        self.assertFalse(stabilizer.accept("Halo", 0.94))

    def test_stabilization_uses_average_confidence(self):
        stabilizer = PredictionStabilizer(window=4, stable_count=3, release_window=6, min_average_confidence=0.65)
        self.assertFalse(stabilizer.accept("Saya", 0.40))
        self.assertFalse(stabilizer.accept("Saya", 0.45))
        self.assertFalse(stabilizer.accept("Saya", 0.50))

    def test_stabilization_suppresses_duplicates_until_release(self):
        stabilizer = PredictionStabilizer(window=4, stable_count=2, release_window=3)
        self.assertFalse(stabilizer.accept("Saya", 0.91))
        self.assertTrue(stabilizer.accept("Saya", 0.91))
        self.assertFalse(stabilizer.accept("Saya", 0.91))
        self.assertFalse(stabilizer.accept(None, 0))
        self.assertFalse(stabilizer.accept(None, 0))
        self.assertFalse(stabilizer.accept(None, 0))
        self.assertFalse(stabilizer.accept("Saya", 0.91))
        self.assertTrue(stabilizer.accept("Saya", 0.91))

    def test_stabilization_does_not_repeat_word_held_for_three_seconds(self):
        stabilizer = PredictionStabilizer(window=4, stable_count=2, release_window=3)
        self.assertFalse(stabilizer.accept("Saya", 0.91))
        self.assertTrue(stabilizer.accept("Saya", 0.91))
        for _ in range(20):
            self.assertFalse(stabilizer.accept("Saya", 0.91))

    def test_stabilization_accepts_different_stable_class_while_locked(self):
        stabilizer = PredictionStabilizer(window=4, stable_count=2, release_window=3)
        self.assertFalse(stabilizer.accept("Hati-hati", 0.91))
        self.assertTrue(stabilizer.accept("Hati-hati", 0.91))
        self.assertFalse(stabilizer.accept("Makan", 0.91))
        self.assertTrue(stabilizer.accept("Makan", 0.91))

    def test_default_stabilization_requires_three_of_five_and_duration(self):
        stabilizer = PredictionStabilizer()
        result = None
        for timestamp in [100, 250, 400]:
            result = stabilizer.evaluate("Halo", 0.98, timestamp_ms=timestamp, probabilities={"Halo": 0.98})
        self.assertTrue(result["accepted"])
        self.assertEqual(result["agreement_count"], 3)
        self.assertEqual(result["required_count"], 3)
        self.assertEqual(result["required_window"], 5)
        self.assertEqual(result["stable_duration_ms"], 300)

    def test_stable_high_confidence_sequences_accept_priority_classes(self):
        priority_classes = [
            "Halo",
            "Makan",
            "Maaf",
            "Berhenti",
            "Membaca",
            "Mau",
            "Saya",
            "Anda",
            "Apa",
            "Nama",
            "Lelah",
            "Takut",
            "Bodoh",
            "Siapa",
            "Terima kasih",
        ]
        for label in priority_classes:
            with self.subTest(label=label):
                stabilizer = PredictionStabilizer()
                result = None
                for index in range(3):
                    result = stabilizer.evaluate(label, 0.98, timestamp_ms=100 + index * 150, probabilities={label: 0.98})
                self.assertTrue(result["accepted"])
                self.assertEqual(result["stable_label"], label)

    def test_sentence_flow_accepts_next_words_without_full_no_hand_release(self):
        stabilizer = PredictionStabilizer()
        accepted = []
        sequence = [
            ("Saya", 100),
            ("Saya", 250),
            ("Saya", 500),
            ("Mau", 650),
            ("Mau", 800),
            ("Mau", 950),
            ("Makan", 1100),
            ("Makan", 1250),
            ("Makan", 1400),
        ]
        for label, timestamp in sequence:
            result = stabilizer.evaluate(label, 0.9, timestamp_ms=timestamp, probabilities={label: 0.9})
            if result["accepted"]:
                accepted.append(result["stable_label"])
        self.assertEqual(accepted, ["Saya", "Mau", "Makan"])

    def test_same_word_requires_release_before_second_acceptance(self):
        stabilizer = PredictionStabilizer()
        accepted = []
        for timestamp in [100, 250, 500, 650, 800, 950]:
            result = stabilizer.evaluate("Saya", 0.9, timestamp_ms=timestamp, probabilities={"Saya": 0.9})
            if result["accepted"]:
                accepted.append(result["stable_label"])
        self.assertEqual(accepted, ["Saya"])
        for timestamp in [1100, 1250, 1400]:
            stabilizer.evaluate(None, 0, timestamp_ms=timestamp)
        for timestamp in [1550, 1700, 1850]:
            result = stabilizer.evaluate("Saya", 0.9, timestamp_ms=timestamp, probabilities={"Saya": 0.9})
            if result["accepted"]:
                accepted.append(result["stable_label"])
        self.assertEqual(accepted, ["Saya", "Saya"])

    def test_probability_smoothing_keeps_stationary_label_over_noisy_top1(self):
        smoother = TemporalProbabilitySmoother(beta=0.25, switch_margin=0.10, switch_confirm_frames=2)
        sequence = [
            [{"label": "Saya", "calibrated_confidence": 0.76}, {"label": "Membaca", "calibrated_confidence": 0.20}],
            [{"label": "Membaca", "calibrated_confidence": 0.58}, {"label": "Saya", "calibrated_confidence": 0.55}],
            [{"label": "Saya", "calibrated_confidence": 0.79}, {"label": "Membaca", "calibrated_confidence": 0.18}],
            [{"label": "Anda", "calibrated_confidence": 0.57}, {"label": "Saya", "calibrated_confidence": 0.56}],
            [{"label": "Saya", "calibrated_confidence": 0.81}, {"label": "Membaca", "calibrated_confidence": 0.16}],
        ]
        outputs = [smoother.update(predictions, pose_state="stationary") for predictions in sequence]
        self.assertEqual(outputs[-1]["current_stable_class"], "Saya")
        self.assertEqual(outputs[-1]["smoothed_top1"]["label"], "Saya")

    def test_probability_hysteresis_requires_confirmed_switch(self):
        smoother = TemporalProbabilitySmoother(beta=0.5, switch_margin=0.10, switch_confirm_frames=2)
        smoother.update([{"label": "Saya", "calibrated_confidence": 0.85}, {"label": "Membaca", "calibrated_confidence": 0.10}])
        slight = smoother.update([{"label": "Membaca", "calibrated_confidence": 0.86}, {"label": "Saya", "calibrated_confidence": 0.82}], pose_state="stationary")
        self.assertEqual(slight["current_stable_class"], "Saya")
        first_strong = smoother.update([{"label": "Membaca", "calibrated_confidence": 0.98}, {"label": "Saya", "calibrated_confidence": 0.40}])
        self.assertEqual(first_strong["current_stable_class"], "Saya")
        second_strong = smoother.update([{"label": "Membaca", "calibrated_confidence": 0.98}, {"label": "Saya", "calibrated_confidence": 0.30}])
        self.assertEqual(second_strong["current_stable_class"], "Saya")
        third_strong = smoother.update([{"label": "Membaca", "calibrated_confidence": 0.98}, {"label": "Saya", "calibrated_confidence": 0.30}])
        self.assertEqual(third_strong["current_stable_class"], "Membaca")

    def test_probability_smoothing_repeated_and_jitter_for_all_classes(self):
        labels = [
            "Anda",
            "Apa",
            "Berhenti",
            "Bodoh",
            "Cantik",
            "Halo",
            "Hati-hati",
            "Lelah",
            "Maaf",
            "Makan",
            "Mau",
            "Membaca",
            "Nama",
            "Sama-sama",
            "Saya",
            "Siapa",
            "Sombong",
            "Takut",
            "Terima kasih",
        ]
        for index, label in enumerate(labels):
            distractor = labels[(index + 1) % len(labels)]
            with self.subTest(label=label):
                smoother = TemporalProbabilitySmoother(beta=0.25, switch_margin=0.10, switch_confirm_frames=2)
                for _ in range(20):
                    result = smoother.update(
                        [
                            {"label": label, "calibrated_confidence": 0.86},
                            {"label": distractor, "calibrated_confidence": 0.42},
                        ],
                        pose_state="stationary",
                    )
                self.assertEqual(result["current_stable_class"], label)

                for frame in range(20):
                    target_confidence = 0.82 + (0.03 if frame % 2 == 0 else -0.03)
                    distractor_confidence = 0.76 + (0.04 if frame % 5 == 0 else -0.02)
                    result = smoother.update(
                        [
                            {"label": distractor, "calibrated_confidence": distractor_confidence},
                            {"label": label, "calibrated_confidence": target_confidence},
                        ],
                        pose_state="stationary",
                    )
                self.assertEqual(result["current_stable_class"], label)

    def test_calibration_rejects_low_margin(self):
        calibration = BisindoCalibration()
        predictions = calibration.calibrate_predictions(
            [
                {"label": "Berhenti", "confidence": 0.99},
                {"label": "Halo", "confidence": 0.98},
                {"label": "Apa", "confidence": 0.2},
            ]
        )
        _top, reason, margin = calibration.decision(predictions)
        self.assertEqual(reason, "low_margin")
        self.assertLess(margin, calibration.min_margin)

    def test_roi_probability_aggregation_uses_evidence_across_candidates(self):
        calibration = BisindoCalibration()
        aggregated = calibration.aggregate_candidate_predictions(
            [
                {"raw_predictions": [{"label": "Saya", "confidence": 0.78}, {"label": "Membaca", "confidence": 0.65}]},
                {"raw_predictions": [{"label": "Saya", "confidence": 0.74}, {"label": "Membaca", "confidence": 0.42}]},
                {"raw_predictions": [{"label": "Saya", "confidence": 0.62}, {"label": "Membaca", "confidence": 0.79}]},
            ]
        )
        self.assertEqual(aggregated[0]["label"], "Saya")

    def test_structure_masks_two_hand_class_for_one_hand_input(self):
        recognizer = TerbisaStructureRecognizer()
        predictions = [
            {"label": "Membaca", "confidence": 0.99, "calibrated_confidence": 0.99},
            {"label": "Saya", "confidence": 0.96, "calibrated_confidence": 0.96},
        ]
        result = recognizer.apply(predictions, structure={"hands_detected": 1}, hands_detected=1)
        self.assertIn("Membaca", result["masked_classes"])
        self.assertEqual(result["predictions"][0]["label"], "Saya")
        membaca = next(item for item in result["predictions"] if item["label"] == "Membaca")
        self.assertEqual(membaca["calibrated_confidence"], 0.0)
        self.assertEqual(membaca["structural_rejection_reason"], "insufficient_hands")

    def test_structure_allows_two_hand_class_for_two_hand_input(self):
        recognizer = TerbisaStructureRecognizer()
        result = recognizer.apply([{"label": "Membaca", "confidence": 0.99, "calibrated_confidence": 0.99}], structure={"hands_detected": 2}, hands_detected=2)
        self.assertNotIn("Membaca", result["masked_classes"])
        self.assertEqual(result["predictions"][0]["calibrated_confidence"], 0.99)

    def test_structure_reranks_takut_over_lelah_for_crossed_two_hand_geometry(self):
        recognizer = TerbisaStructureRecognizer()
        structure = {
            "hands_detected": 2,
            "hands": [
                {"body_region": "chest", "finger_states": {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}, "geometry": {"openness": 5.0}},
                {"body_region": "chest", "finger_states": {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}, "geometry": {"openness": 5.0}},
            ],
            "two_hand_geometry": {"horizontal_crossing": True, "overlap": True, "hands_touching": True, "palm_distance": 0.14, "span": 0.36},
        }
        result = recognizer.apply(
            [
                {"label": "Lelah", "confidence": 0.72, "calibrated_confidence": 0.72},
                {"label": "Takut", "confidence": 0.70, "calibrated_confidence": 0.70},
            ],
            structure=structure,
            hands_detected=2,
        )
        self.assertEqual(result["predictions"][0]["label"], "Takut")

    def test_structure_reranks_lelah_over_takut_for_separated_two_hand_geometry(self):
        recognizer = TerbisaStructureRecognizer()
        structure = {
            "hands_detected": 2,
            "hands": [
                {"body_region": "chest", "finger_states": {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}, "geometry": {"openness": 5.0}},
                {"body_region": "chest", "finger_states": {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}, "geometry": {"openness": 5.0}},
            ],
            "two_hand_geometry": {"horizontal_crossing": False, "overlap": False, "hands_touching": False, "palm_distance": 0.38, "span": 0.62},
        }
        result = recognizer.apply(
            [
                {"label": "Takut", "confidence": 0.72, "calibrated_confidence": 0.72},
                {"label": "Lelah", "confidence": 0.70, "calibrated_confidence": 0.70},
            ],
            structure=structure,
            hands_detected=2,
        )
        self.assertEqual(result["predictions"][0]["label"], "Lelah")

    def test_one_hand_membaca_top1_cannot_survive_predict_sign(self):
        original = main.bisindo_detector
        original_classifier = main.bisindo_classifier
        original_stabilizer = main.prediction_stabilizer
        fake = FakeDetector(
            DetectionResult(
                status="ok",
                detected=True,
                class_id=11,
                raw_label="Membaca",
                display_label="Membaca",
                confidence=0.99,
                raw_predictions=[
                    {"class_id": 11, "raw_label": "Membaca", "label": "Membaca", "confidence": 0.99},
                    {"class_id": 2, "raw_label": "Berhenti", "label": "Berhenti", "confidence": 0.98},
                    {"class_id": 14, "raw_label": "Saya", "label": "Saya", "confidence": 0.96},
                ],
                detections=1,
                threshold_detections=1,
                valid_detections=1,
            )
        )
        main.bisindo_detector = fake
        main.bisindo_classifier = fake
        main.prediction_stabilizer = PredictionStabilizer(window=1, stable_count=1, release_window=3, min_stable_duration_ms=0)
        try:
            response = asyncio.run(
                main.predict_sign(
                    FakeUpload(),
                    frame_id="one-hand-membaca",
                    mirrored=False,
                    hands_detected=1,
                    structure_json=json.dumps({"hands_detected": 1, "hands": [{"body_region": "chest"}]}),
                )
            )
        finally:
            main.bisindo_detector = original
            main.bisindo_classifier = original_classifier
            main.prediction_stabilizer = original_stabilizer
        self.assertIn("Membaca", response.masked_classes)
        self.assertIn("Berhenti", response.masked_classes)
        self.assertNotEqual(response.prediction, "Membaca")
        self.assertEqual(response.prediction, "Saya")


if __name__ == "__main__":
    unittest.main()
