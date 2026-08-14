const camera = document.getElementById("translatorCamera");
const overlay = document.getElementById("translatorOverlay");
const fallback = document.getElementById("translatorCameraFallback");
const cameraStatus = document.getElementById("translatorCameraStatus");
const aiStatus = document.getElementById("translatorAiStatus");
const motionStatus = document.getElementById("translatorMotionStatus");
const detectedText = document.getElementById("detectedText");
const detectedConfidence = document.getElementById("detectedConfidence");
const signTranscript = document.getElementById("signTranscript");
const speakSignText = document.getElementById("speakSignText");
const clearSignTranscript = document.getElementById("clearSignTranscript");
const restartSignTranslator = document.getElementById("restartSignTranslator");
const modeSign = document.getElementById("modeSign");
const modeSpeech = document.getElementById("modeSpeech");
const signModePanel = document.getElementById("signModePanel");
const speechModePanel = document.getElementById("speechModePanel");
const microphoneStatus = document.getElementById("microphoneStatus");
const speechTranscript = document.getElementById("speechTranscript");
const recordSpeech = document.getElementById("recordSpeech");
const stopSpeech = document.getElementById("stopSpeech");
const copySpeechText = document.getElementById("copySpeechText");
const debugRawPredictions = document.getElementById("debugRawPredictions");
const debugClassifierInputPreview = document.getElementById("debugClassifierInputPreview");
const debugClassifierInputMeta = document.getElementById("debugClassifierInputMeta");
const evalPanel = document.getElementById("translatorEvalPanel");
const evalTargetSign = document.getElementById("evalTargetSign");
const evalCaptureSample = document.getElementById("evalCaptureSample");
const evalReset = document.getElementById("evalReset");
const evalSummary = document.getElementById("evalSummary");
const evalLog = document.getElementById("evalLog");

const ctx = overlay.getContext("2d");
const MEDIAPIPE_VERSION = "0.10.35";
const MEDIAPIPE_IMPORT_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const MODEL_URLS = {
  hand: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
};
const YOLO_CONFIDENCE_THRESHOLD = 0.75;
const YOLO_REQUEST_INTERVAL_MS = 220;
const DISPLAY_FRAME_MIRRORED = true;
const AI_INPUT_MIRRORED = false;
const LANDMARK_SMOOTHING_ALPHA = Number(window.ISYARA_LANDMARK_SMOOTHING_ALPHA || 0.35);
const ROI_SMOOTHING_ALPHA = Number(window.ISYARA_ROI_SMOOTHING_ALPHA || 0.35);
const LANDMARK_STATIONARY_THRESHOLD = Number(window.ISYARA_LANDMARK_STATIONARY_THRESHOLD || 0.012);
const ROI_CENTER_STATIONARY_THRESHOLD = Number(window.ISYARA_ROI_CENTER_STATIONARY_THRESHOLD || 0.012);
const ROI_SIZE_STATIONARY_THRESHOLD = Number(window.ISYARA_ROI_SIZE_STATIONARY_THRESHOLD || 0.018);
const ROI_PADDING = 0.45;
const ROI_PADDING_BY_TYPE = {
  tight: 0.18,
  medium: 0.55,
  largeContext: 0.9,
  squareContext: 0.75,
  contextual: 1.1,
  upper: 1.35,
  combined: 0.65,
  combinedContext: 1.0,
};
const SUPPORTED_SIGNS = ["Anda", "Apa", "Berhenti", "Bodoh", "Cantik", "Halo", "Hati-hati", "Lelah", "Maaf", "Makan", "Mau", "Membaca", "Nama", "Sama-sama", "Saya", "Siapa", "Sombong", "Takut", "Terima kasih"];

let handLandmarker;
let faceDetector;
let stream;
let lastPredictionRequestTime = 0;
let predictionInFlight = false;
let acceptedTokens = [];
let frameCounter = 0;
let lastPredictionResult = null;
let lastTrackingState = null;
let smoothedHandState = null;
let lastLandmarkMotionScore = 0;
let lastPoseState = "no_hand";
let smoothedRoiState = new Map();
let lastRoiMotionScore = 0;
let classifierInputPreviewUrls = new Map();
let evalSamples = [];
let evalCaptureRemaining = 0;
let evalCaptureTarget = null;
let speechRecorder;
let speechChunks = [];
let speechRecognition;
let speechRecognitionText = "";
let speechRecognitionActive = false;
const yoloCanvas = document.createElement("canvas");
const yoloContext = yoloCanvas.getContext("2d", { willReadFrequently: false });
const roiCanvas = document.createElement("canvas");
const roiContext = roiCanvas.getContext("2d", { willReadFrequently: false });

function csrfToken() {
  return document.querySelector("input[name='csrfmiddlewaretoken']")?.value || "";
}

function setPill(element, state, text) {
  element.textContent = text;
  element.classList.toggle("ok", state === "ok");
  element.classList.toggle("warn", state === "warn");
}

function sentenceToken(label, index) {
  if (!label) return "";
  return index === 0 ? label : label.charAt(0).toLocaleLowerCase("id-ID") + label.slice(1);
}

function transcriptSentence(withPunctuation = false) {
  const sentence = acceptedTokens.map(sentenceToken).join(" ");
  if (!withPunctuation || !sentence.trim()) return sentence;
  return sentence.endsWith(".") ? sentence : `${sentence}.`;
}

function renderTranscript() {
  signTranscript.textContent = acceptedTokens.length ? transcriptSentence(false) : "Belum ada tanda stabil yang dikenali.";
}

function updateDebug(result, stableLabel = null, suppressed = false) {
  if (!debugRawPredictions) return;
  const rawPredictions = result.raw_predictions || [];
  const candidatePredictions = result.candidate_predictions || [];
  debugRawPredictions.textContent = candidatePredictions.length
    ? candidatePredictions.map(formatCandidateDebug).join("\n\n")
    : rawPredictions.length
    ? rawPredictions.map((item) => `${item.label || item.raw_label}: ${Math.round(Number(item.confidence || 0) * 100)}%`).join("\n")
    : "-";
  updateClassifierInputPreview(result);
}

function updateClassifierInputPreview(result) {
  if (!debugClassifierInputPreview || !debugClassifierInputMeta) return;
  const selectedType = result.selected_candidate?.roi_type || result.roi_type;
  const url = classifierInputPreviewUrls.get(selectedType) || classifierInputPreviewUrls.values().next().value;
  if (url) debugClassifierInputPreview.src = url;
  const debug = result.classifier_debug || result.selected_candidate?.debug || {};
  const top5Flipped = (debug.top5_flipped || []).slice(0, 5).map((item) => `${item.label}: ${Math.round(Number(item.confidence || 0) * 100)}%`).join("\n");
  debugClassifierInputMeta.textContent = [
    `roi_type=${selectedType || "-"}`,
    `saved=${debug.exact_input_path || "-"}`,
    `source=${debug.source_size ? `${debug.source_size.width}x${debug.source_size.height}` : "-"}`,
    `full=${debug.full_frame_dimensions ? `${debug.full_frame_dimensions.width}x${debug.full_frame_dimensions.height}` : "-"}`,
    `mirror=${debug.mirrored ?? result.mirrored ?? AI_INPUT_MIRRORED}`,
    `bytes=${debug.input_bytes || "-"}`,
    top5Flipped ? `flipped top5:\n${top5Flipped}` : "flipped top5=-",
  ].join("\n");
}

function formatCandidateDebug(candidate) {
  const raw = candidate.raw_predictions || [];
  const top = raw.length ? raw[0] : candidate;
  const label = top.label || top.raw_label || candidate.label || "-";
  const confidence = Math.round(Number(top.confidence || candidate.confidence || 0) * 100);
  return `candidate_${candidate.roi_type || "unknown"}:\n${label} ${confidence}%`;
}

function formatCalibrationDebug(result) {
  const scores = result.raw_predictions || [];
  return scores
    .slice(0, 3)
    .map((item) => {
      const raw = Math.round(Number(item.confidence || 0) * 100);
      const calibrated = Math.round(Number(item.calibrated_confidence ?? item.confidence ?? 0) * 100);
      const threshold = Math.round(Number(item.class_threshold || 0) * 100);
      return `${item.label}: raw ${raw}% cal ${calibrated}% threshold ${threshold}%`;
    })
    .join("\n") || "-";
}

async function initHandLocalizer() {
  try {
    setPill(aiStatus, "warn", "○ Memuat lokalisasi tangan");
    const visionModule = await import(MEDIAPIPE_IMPORT_URL);
    const vision = await visionModule.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
    handLandmarker = await visionModule.HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URLS.hand },
      runningMode: "VIDEO",
      numHands: 2,
    });
    try {
      faceDetector = await visionModule.FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URLS.face },
        runningMode: "VIDEO",
      });
    } catch (faceError) {
      console.info("[ISYARA Translator] face detector unavailable; using hand-only structure", faceError);
    }
    setPill(aiStatus, "ok", "✓ Lokalisasi tangan siap");
  } catch (error) {
    console.error("[ISYARA Translator] hand localizer init failed", error);
    setPill(aiStatus, "warn", "! Lokalisasi tangan belum tersedia");
  }
}

function syncCanvas() {
  if (!camera.videoWidth || !camera.videoHeight) return false;
  if (overlay.width !== camera.videoWidth || overlay.height !== camera.videoHeight) {
    overlay.width = camera.videoWidth;
    overlay.height = camera.videoHeight;
  }
  return true;
}

function clearOverlay() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawYoloBox(result) {
  clearOverlay();
  if (!result.detected || !result.bbox || !result.image_width || !result.image_height) return;
  const bbox = displayBbox(result.bbox, result.image_width);
  const scaleX = overlay.width / result.image_width;
  const scaleY = overlay.height / result.image_height;
  const x = bbox.x1 * scaleX;
  const y = bbox.y1 * scaleY;
  const width = (bbox.x2 - bbox.x1) * scaleX;
  const height = (bbox.y2 - bbox.y1) * scaleY;
  if (width <= 0 || height <= 0) return;
  const label = `${result.display_label || result.label || result.prediction} · ${Math.round(Number(result.confidence || 0) * 100)}%`;
  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
  ctx.lineWidth = 2;
  ctx.shadowColor = "rgba(0, 0, 0, 0.45)";
  ctx.shadowBlur = 8;
  roundedRect(ctx, x, y, width, height, 8);
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.font = "600 13px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  const textWidth = ctx.measureText(label).width;
  const tagX = Math.max(6, Math.min(x, overlay.width - textWidth - 18));
  const tagY = Math.max(6, y - 26);
  ctx.fillStyle = "rgba(27, 28, 27, 0.88)";
  roundedRect(ctx, tagX, tagY, textWidth + 14, 22, 6);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.fillText(label, tagX + 7, tagY + 15);
  ctx.restore();
}

function displayBbox(bbox, imageWidth) {
  if (DISPLAY_FRAME_MIRRORED === AI_INPUT_MIRRORED) return bbox;
  return { x1: imageWidth - bbox.x2, y1: bbox.y1, x2: imageWidth - bbox.x1, y2: bbox.y2 };
}

function roundedRect(context, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

function localizeHands(timestamp) {
  if (!handLandmarker) return null;
  return smoothHandResults(handLandmarker.detectForVideo(camera, timestamp));
}

function localizeFace(timestamp) {
  if (!faceDetector) return null;
  try {
    return faceDetector.detectForVideo(camera, timestamp);
  } catch (error) {
    return null;
  }
}

function structuralFeatures(handResults, faceResults) {
  const hands = handResults?.landmarks || [];
  const handedness = (handResults?.handednesses || []).map((items) => items?.[0]?.categoryName).filter(Boolean);
  const faceBox = primaryFaceBox(faceResults);
  const handFeatures = hands.map((hand, index) => {
    const bounds = landmarkBounds(hand);
    const center = { x: (bounds.x1 + bounds.x2) / 2, y: (bounds.y1 + bounds.y2) / 2 };
    const bodyDistances = bodyDistancesForHand(center, faceBox);
    return {
      handedness: handedness[index] || `Hand ${index + 1}`,
      center,
      bounds,
      landmarks: hand.map((point) => ({
        x: roundFeature(point.x),
        y: roundFeature(point.y),
        z: roundFeature(point.z || 0),
      })),
      finger_states: fingerStates(hand, handedness[index]),
      geometry: handGeometry(hand),
      body_region: bodyRegion(center, faceBox),
      body_distances: bodyDistances,
    };
  });
  return {
    hands_detected: hands.length,
    handedness,
    face: faceBox,
    hands: handFeatures,
    two_hand_distance: handFeatures.length >= 2 ? distance(handFeatures[0].center, handFeatures[1].center) : null,
    hands_close: handFeatures.length >= 2 ? distance(handFeatures[0].center, handFeatures[1].center) < 0.22 : false,
    two_hand_geometry: twoHandGeometry(handFeatures),
    landmark_motion_score: roundFeature(lastLandmarkMotionScore),
    roi_motion_score: roundFeature(lastRoiMotionScore),
    pose_state: lastPoseState,
    smoothing: {
      landmark_alpha: LANDMARK_SMOOTHING_ALPHA,
      roi_alpha: ROI_SMOOTHING_ALPHA,
      landmark_stationary_threshold: LANDMARK_STATIONARY_THRESHOLD,
      roi_center_stationary_threshold: ROI_CENTER_STATIONARY_THRESHOLD,
      roi_size_stationary_threshold: ROI_SIZE_STATIONARY_THRESHOLD,
    },
  };
}

function smoothHandResults(results) {
  const hands = results?.landmarks || [];
  const handednesses = results?.handednesses || [];
  if (!hands.length) {
    smoothedHandState = null;
    lastLandmarkMotionScore = 0;
    lastPoseState = "no_hand";
    return results;
  }
  const previous = smoothedHandState;
  let displacementTotal = 0;
  let displacementCount = 0;
  const smoothedLandmarks = hands.map((hand, handIndex) => {
    const currentHandedness = handednesses[handIndex]?.[0]?.categoryName;
    const previousHand = previousHandFor(currentHandedness, handIndex);
    return hand.map((point, landmarkIndex) => {
      const prev = previousHand?.[landmarkIndex];
      if (prev) {
        displacementTotal += distance3d(point, prev);
        displacementCount += 1;
      }
      return prev
        ? {
            x: LANDMARK_SMOOTHING_ALPHA * point.x + (1 - LANDMARK_SMOOTHING_ALPHA) * prev.x,
            y: LANDMARK_SMOOTHING_ALPHA * point.y + (1 - LANDMARK_SMOOTHING_ALPHA) * prev.y,
            z: LANDMARK_SMOOTHING_ALPHA * (point.z || 0) + (1 - LANDMARK_SMOOTHING_ALPHA) * (prev.z || 0),
          }
        : { x: point.x, y: point.y, z: point.z || 0 };
    });
  });
  lastLandmarkMotionScore = displacementCount ? displacementTotal / displacementCount : 1;
  lastPoseState = lastLandmarkMotionScore < LANDMARK_STATIONARY_THRESHOLD ? "stationary" : "transitioning";
  smoothedHandState = { landmarks: smoothedLandmarks, handednesses };
  return { ...results, landmarks: smoothedLandmarks, handednesses };
}

function previousHandFor(handedness, fallbackIndex) {
  if (!smoothedHandState?.landmarks?.length) return null;
  const label = String(handedness || "").toLowerCase();
  if (label) {
    const index = (smoothedHandState.handednesses || []).findIndex((items) => String(items?.[0]?.categoryName || "").toLowerCase() === label);
    if (index >= 0) return smoothedHandState.landmarks[index];
  }
  return smoothedHandState.landmarks[fallbackIndex] || null;
}

function roundFeature(value) {
  return Math.round(Number(value || 0) * 10000) / 10000;
}

function fingerStates(hand, handedness) {
  const handed = String(handedness || "").toLowerCase();
  const thumbOpen = handed.includes("left") ? hand[4].x > hand[3].x : hand[4].x < hand[3].x;
  return {
    thumb: thumbOpen,
    index: fingerExtended(hand, 8, 6),
    middle: fingerExtended(hand, 12, 10),
    ring: fingerExtended(hand, 16, 14),
    pinky: fingerExtended(hand, 20, 18),
  };
}

function fingerExtended(hand, tipIndex, pipIndex) {
  return hand[tipIndex].y < hand[pipIndex].y;
}

function handGeometry(hand) {
  const wrist = hand[0];
  const indexTip = hand[8];
  const pinkyTip = hand[20];
  const middleMcp = hand[9];
  const palmWidth = distance(hand[5], hand[17]);
  const palmHeight = distance(wrist, middleMcp);
  const spread = distance(indexTip, pinkyTip) / Math.max(0.001, palmWidth);
  const openness = [4, 8, 12, 16, 20].reduce((total, index) => total + distance(wrist, hand[index]), 0) / Math.max(0.001, palmWidth);
  return {
    palm_aspect: roundFeature(palmWidth / Math.max(0.001, palmHeight)),
    openness: roundFeature(openness),
    rotation: roundFeature(Math.atan2(hand[5].y - hand[17].y, hand[5].x - hand[17].x)),
    wrist_to_index: vector(wrist, indexTip),
    wrist_to_middle: vector(wrist, hand[12]),
    index_vector: normalizedVector(wrist, indexTip, palmWidth),
    middle_vector: normalizedVector(wrist, hand[12], palmWidth),
    thumb_vector: normalizedVector(wrist, hand[4], palmWidth),
    fingertip_spread: roundFeature(spread),
    finger_angles: {
      thumb: jointAngle(hand[2], hand[3], hand[4]),
      index: jointAngle(hand[5], hand[6], hand[8]),
      middle: jointAngle(hand[9], hand[10], hand[12]),
      ring: jointAngle(hand[13], hand[14], hand[16]),
      pinky: jointAngle(hand[17], hand[18], hand[20]),
    },
    fingertip_distances: {
      thumb_index: roundFeature(distance(hand[4], hand[8]) / Math.max(0.001, palmWidth)),
      index_middle: roundFeature(distance(hand[8], hand[12]) / Math.max(0.001, palmWidth)),
      middle_ring: roundFeature(distance(hand[12], hand[16]) / Math.max(0.001, palmWidth)),
      ring_pinky: roundFeature(distance(hand[16], hand[20]) / Math.max(0.001, palmWidth)),
    },
  };
}

function vector(a, b) {
  return { x: roundFeature(b.x - a.x), y: roundFeature(b.y - a.y), z: roundFeature((b.z || 0) - (a.z || 0)) };
}

function normalizedVector(a, b, scale) {
  const safeScale = Math.max(0.001, scale);
  return {
    x: roundFeature((b.x - a.x) / safeScale),
    y: roundFeature((b.y - a.y) / safeScale),
    z: roundFeature(((b.z || 0) - (a.z || 0)) / safeScale),
  };
}

function jointAngle(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
  const cb = { x: c.x - b.x, y: c.y - b.y, z: (c.z || 0) - (b.z || 0) };
  const dot = ab.x * cb.x + ab.y * cb.y + ab.z * cb.z;
  const mag = Math.max(0.0001, Math.hypot(ab.x, ab.y, ab.z) * Math.hypot(cb.x, cb.y, cb.z));
  return roundFeature(Math.acos(Math.max(-1, Math.min(1, dot / mag))) / Math.PI);
}

function bodyDistancesForHand(center, faceBox) {
  const regions = bodyAnchors(faceBox);
  return Object.fromEntries(Object.entries(regions).map(([name, point]) => [name, roundFeature(distance(center, point))]));
}

function bodyRegion(center, faceBox) {
  if (!faceBox) {
    if (center.y < 0.38) return "head";
    if (center.y < 0.58) return "chest";
    return "torso";
  }
  const anchors = bodyAnchors(faceBox);
  const nearest = Object.entries(anchors).sort((a, b) => distance(center, a[1]) - distance(center, b[1]))[0]?.[0] || "torso";
  if (nearest === "mouth" || nearest === "chin" || nearest === "forehead" || nearest === "head") return nearest;
  return nearest;
}

function bodyAnchors(faceBox) {
  if (!faceBox) {
    return {
      forehead: { x: 0.5, y: 0.24 },
      mouth: { x: 0.5, y: 0.36 },
      chin: { x: 0.5, y: 0.42 },
      chest: { x: 0.5, y: 0.62 },
      torso: { x: 0.5, y: 0.76 },
    };
  }
  const h = faceBox.height;
  return {
    forehead: { x: faceBox.center.x, y: faceBox.y1 + h * 0.22 },
    head: faceBox.center,
    mouth: { x: faceBox.center.x, y: faceBox.y1 + h * 0.68 },
    chin: { x: faceBox.center.x, y: faceBox.y2 },
    chest: { x: faceBox.center.x, y: Math.min(1, faceBox.y2 + h * 1.35) },
    shoulders: { x: faceBox.center.x, y: Math.min(1, faceBox.y2 + h * 0.9) },
    torso: { x: faceBox.center.x, y: Math.min(1, faceBox.y2 + h * 2.25) },
  };
}

function twoHandGeometry(hands) {
  if (hands.length < 2) return null;
  const [first, second] = hands;
  const leftToRight = [...hands].sort((a, b) => a.center.x - b.center.x);
  const left = leftToRight[0];
  const right = leftToRight[1];
  const span = Math.max(first.bounds.x2, second.bounds.x2) - Math.min(first.bounds.x1, second.bounds.x1);
  const horizontalCrossing = horizontalOverlap(first.bounds, second.bounds) && Math.abs(first.center.y - second.center.y) < 0.18;
  const labeledLeft = hands.find((hand) => String(hand.handedness || "").toLowerCase().includes("left"));
  const labeledRight = hands.find((hand) => String(hand.handedness || "").toLowerCase().includes("right"));
  const handedOrderCrossed = Boolean(labeledLeft && labeledRight && labeledLeft.center.x > labeledRight.center.x);
  return {
    wrist_distance: roundFeature(distance(first.landmarks[0], second.landmarks[0])),
    palm_distance: roundFeature(distance(first.center, second.center)),
    relative_height: roundFeature(first.center.y - second.center.y),
    overlap: boxesOverlap(first.bounds, second.bounds),
    hands_touching: distance(first.center, second.center) < 0.18,
    horizontal_crossing: horizontalCrossing,
    handed_order_crossed: handedOrderCrossed,
    span: roundFeature(span),
    symmetry: roundFeature(1 - Math.min(1, Math.abs(first.center.y - second.center.y) + Math.abs((first.bounds.x2 - first.bounds.x1) - (second.bounds.x2 - second.bounds.x1)))),
  };
}

function horizontalOverlap(a, b) {
  return Boolean(a && b && a.x1 < b.x2 && a.x2 > b.x1);
}

function boxesOverlap(a, b) {
  return Boolean(a && b && a.x1 < b.x2 && a.x2 > b.x1 && a.y1 < b.y2 && a.y2 > b.y1);
}

function primaryFaceBox(faceResults) {
  const detections = faceResults?.detections || [];
  const detection = detections[0];
  const box = detection?.boundingBox;
  if (!box || !camera.videoWidth || !camera.videoHeight) return null;
  const x = Number(box.originX ?? box.xCenter ?? 0);
  const y = Number(box.originY ?? box.yCenter ?? 0);
  const width = Number(box.width ?? 0);
  const height = Number(box.height ?? 0);
  const normalized = {
    x1: clamp01(x / camera.videoWidth),
    y1: clamp01(y / camera.videoHeight),
    x2: clamp01((x + width) / camera.videoWidth),
    y2: clamp01((y + height) / camera.videoHeight),
  };
  normalized.width = Math.max(0.001, normalized.x2 - normalized.x1);
  normalized.height = Math.max(0.001, normalized.y2 - normalized.y1);
  normalized.center = {
    x: (normalized.x1 + normalized.x2) / 2,
    y: (normalized.y1 + normalized.y2) / 2,
  };
  return normalized;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function distance3d(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function handCandidates(results, faceResults) {
  const hands = results?.landmarks || [];
  if (!hands.length) return [];
  const handedness = (results.handednesses || []).map((items) => items?.[0]?.categoryName).filter(Boolean);
  const sourceWidth = camera.videoWidth;
  const sourceHeight = camera.videoHeight;
  const handBounds = hands.map((hand, index) => ({
    ...landmarkBounds(hand),
    handedness: handedness[index] || `Hand ${index + 1}`,
    index,
  }));
  const candidates = [];
  handBounds.forEach((bounds) => {
    const baseType = candidateType(bounds, handBounds);
    addCandidate(candidates, bounds, `${baseType}_square_context`, sourceWidth, sourceHeight, hands.length, handedness, "squareContext");
    addCandidate(candidates, bounds, `${baseType}_large_context`, sourceWidth, sourceHeight, hands.length, handedness, "largeContext");
    const faceBox = primaryFaceBox(faceResults);
    if (faceBox) {
      addCandidate(candidates, unionBounds([bounds, faceBox]), `${baseType}_face_union`, sourceWidth, sourceHeight, hands.length, handedness, "contextual");
    }
    addCandidate(candidates, bounds, `${baseType}_upper_body`, sourceWidth, sourceHeight, hands.length, handedness, "upper");
  });
  if (handBounds.length > 1) {
    const combined = unionBounds(handBounds);
    addCandidate(candidates, combined, "combined_square_context", sourceWidth, sourceHeight, hands.length, handedness, "squareContext");
    addCandidate(candidates, combined, "combined_large_context", sourceWidth, sourceHeight, hands.length, handedness, "largeContext");
    const faceBox = primaryFaceBox(faceResults);
    if (faceBox) {
      addCandidate(candidates, unionBounds([combined, faceBox]), "combined_face_union", sourceWidth, sourceHeight, hands.length, handedness, "combinedContext");
    }
  }
  return smoothRoiCandidates(uniqueCandidates(candidates).slice(0, 6));
}

function smoothRoiCandidates(candidates) {
  if (!candidates.length) {
    smoothedRoiState = new Map();
    lastRoiMotionScore = 0;
    return [];
  }
  const nextState = new Map();
  let movementTotal = 0;
  let movementCount = 0;
  const smoothed = candidates.map((roi) => {
    const previous = smoothedRoiState.get(roi.type);
    if (!previous || previous.sourceWidth !== roi.sourceWidth || previous.sourceHeight !== roi.sourceHeight) {
      nextState.set(roi.type, roi);
      return roi;
    }
    const motion = roiMotion(previous, roi);
    movementTotal += motion.score;
    movementCount += 1;
    if (motion.center < ROI_CENTER_STATIONARY_THRESHOLD && motion.size < ROI_SIZE_STATIONARY_THRESHOLD) {
      nextState.set(roi.type, previous);
      return { ...previous, motion_score: motion.score, pose_state: lastPoseState };
    }
    const eased = {
      ...roi,
      x1: Math.round(ROI_SMOOTHING_ALPHA * roi.x1 + (1 - ROI_SMOOTHING_ALPHA) * previous.x1),
      y1: Math.round(ROI_SMOOTHING_ALPHA * roi.y1 + (1 - ROI_SMOOTHING_ALPHA) * previous.y1),
      x2: Math.round(ROI_SMOOTHING_ALPHA * roi.x2 + (1 - ROI_SMOOTHING_ALPHA) * previous.x2),
      y2: Math.round(ROI_SMOOTHING_ALPHA * roi.y2 + (1 - ROI_SMOOTHING_ALPHA) * previous.y2),
      motion_score: motion.score,
      pose_state: lastPoseState,
    };
    nextState.set(roi.type, eased);
    return eased;
  });
  smoothedRoiState = nextState;
  lastRoiMotionScore = movementCount ? movementTotal / movementCount : 0;
  if (lastPoseState === "stationary" && lastRoiMotionScore > ROI_CENTER_STATIONARY_THRESHOLD + ROI_SIZE_STATIONARY_THRESHOLD) {
    lastPoseState = "transitioning";
  }
  return smoothed;
}

function roiMotion(previous, current) {
  const prevWidth = Math.max(1, previous.sourceWidth || current.sourceWidth || 1);
  const prevHeight = Math.max(1, previous.sourceHeight || current.sourceHeight || 1);
  const prevCenter = { x: (previous.x1 + previous.x2) / 2 / prevWidth, y: (previous.y1 + previous.y2) / 2 / prevHeight };
  const currentCenter = { x: (current.x1 + current.x2) / 2 / prevWidth, y: (current.y1 + current.y2) / 2 / prevHeight };
  const center = distance(prevCenter, currentCenter);
  const prevSize = { width: (previous.x2 - previous.x1) / prevWidth, height: (previous.y2 - previous.y1) / prevHeight };
  const currentSize = { width: (current.x2 - current.x1) / prevWidth, height: (current.y2 - current.y1) / prevHeight };
  const size = Math.hypot(currentSize.width - prevSize.width, currentSize.height - prevSize.height);
  return { center, size, score: center + size };
}

function landmarkBounds(hand) {
  let x1 = 1;
  let y1 = 1;
  let x2 = 0;
  let y2 = 0;
  hand.forEach((point) => {
    x1 = Math.min(x1, point.x);
    y1 = Math.min(y1, point.y);
    x2 = Math.max(x2, point.x);
    y2 = Math.max(y2, point.y);
  });
  return { x1, y1, x2, y2 };
}

function unionBounds(boundsList) {
  return boundsList.reduce(
    (union, bounds) => ({
      x1: Math.min(union.x1, bounds.x1),
      y1: Math.min(union.y1, bounds.y1),
      x2: Math.max(union.x2, bounds.x2),
      y2: Math.max(union.y2, bounds.y2),
    }),
    { x1: 1, y1: 1, x2: 0, y2: 0 }
  );
}

function candidateType(bounds, allBounds) {
  const side = String(bounds.handedness || "").toLowerCase();
  if (side.includes("left")) return "left";
  if (side.includes("right")) return "right";
  return `hand_${allBounds.indexOf(bounds) + 1}`;
}

function addCandidate(candidates, bounds, type, sourceWidth, sourceHeight, handsDetected, handedness, scaleType = "medium") {
  const roi = roiFromBounds(bounds, type, sourceWidth, sourceHeight, handsDetected, handedness, scaleType);
  if (roi && roi.x2 > roi.x1 && roi.y2 > roi.y1) candidates.push(roi);
}

function roiFromBounds(bounds, type, sourceWidth, sourceHeight, handsDetected, handedness, scaleType = "medium") {
  const width = bounds.x2 - bounds.x1;
  const height = bounds.y2 - bounds.y1;
  const padding = ROI_PADDING_BY_TYPE[scaleType] ?? ROI_PADDING;
  const padX = Math.max(width * padding, 0.05);
  const upperBias = scaleType === "upper" || scaleType === "combinedContext" ? 1.65 : 1;
  const lowerBias = scaleType === "upper" || scaleType === "combinedContext" ? 0.75 : 1;
  const padY = Math.max(height * padding, 0.05);
  let x1 = bounds.x1 - padX;
  let y1 = bounds.y1 - padY * upperBias;
  let x2 = bounds.x2 + padX;
  let y2 = bounds.y2 + padY * lowerBias;
  if (scaleType === "squareContext") {
    const centerX = (bounds.x1 + bounds.x2) / 2;
    const centerY = (bounds.y1 + bounds.y2) / 2;
    const side = Math.max(x2 - x1, y2 - y1, 0.28);
    x1 = centerX - side / 2;
    x2 = centerX + side / 2;
    y1 = centerY - side / 2;
    y2 = centerY + side / 2;
  }
  return {
    type,
    x1: Math.max(0, Math.round(x1 * sourceWidth)),
    y1: Math.max(0, Math.round(y1 * sourceHeight)),
    x2: Math.min(sourceWidth, Math.round(x2 * sourceWidth)),
    y2: Math.min(sourceHeight, Math.round(y2 * sourceHeight)),
    sourceWidth,
    sourceHeight,
    handsDetected,
    handedness,
  };
}

function uniqueCandidates(candidates) {
  const seen = new Set();
  return candidates.filter((roi) => {
    const key = [roi.type, Math.round(roi.x1 / 8), Math.round(roi.y1 / 8), Math.round(roi.x2 / 8), Math.round(roi.y2 / 8)].join(":");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function drawTrackingOverlay(result = {}) {
  clearOverlay();
  const state = lastTrackingState;
  if (!state || !overlay.width || !overlay.height) return;
  const hands = state.structure?.hands || [];
  const selectedRoi = result.roi || state.rois?.[0] || null;
  if (selectedRoi) drawPixelBox(selectedRoi, "area konteks", "rgba(122, 92, 60, 0.58)", true);
  hands.forEach((hand) => {
    const label = String(hand.handedness || "Hand").replace("Left", "Tangan kiri").replace("Right", "Tangan kanan");
    drawNormalizedBox(hand.bounds, label, "rgba(255, 255, 255, 0.92)");
  });
  const stable = result.accepted_prediction || result.stable_prediction;
  const confidence = Number(result.accepted_confidence ?? result.stable_confidence ?? result.confidence ?? 0);
  const text = stable ? `${stable} · ${Math.round(confidence * 100)}%` : `${hands.length} tangan`;
  drawOverlayTag(text, 10, 10);
}

function drawNormalizedBox(bounds, label, color) {
  if (!bounds) return;
  const rect = {
    x1: bounds.x1 * overlay.width,
    y1: bounds.y1 * overlay.height,
    x2: bounds.x2 * overlay.width,
    y2: bounds.y2 * overlay.height,
  };
  if (DISPLAY_FRAME_MIRRORED) {
    const x1 = overlay.width - rect.x2;
    rect.x2 = overlay.width - rect.x1;
    rect.x1 = x1;
  }
  drawRect(rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1, label, color, false);
}

function drawPixelBox(roi, label, color, dashed = false) {
  const scaleX = overlay.width / (roi.source_width || roi.sourceWidth || camera.videoWidth || overlay.width);
  const scaleY = overlay.height / (roi.source_height || roi.sourceHeight || camera.videoHeight || overlay.height);
  let x1 = roi.x1 * scaleX;
  let x2 = roi.x2 * scaleX;
  if (DISPLAY_FRAME_MIRRORED !== AI_INPUT_MIRRORED) {
    const width = overlay.width;
    const mirroredX1 = width - x2;
    x2 = width - x1;
    x1 = mirroredX1;
  }
  drawRect(x1, roi.y1 * scaleY, x2 - x1, (roi.y2 - roi.y1) * scaleY, label, color, dashed);
}

function drawRect(x, y, width, height, label, color, dashed) {
  if (width <= 0 || height <= 0) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  if (dashed) ctx.setLineDash([6, 5]);
  roundedRect(ctx, x, y, width, height, 7);
  ctx.stroke();
  ctx.setLineDash([]);
  drawOverlayTag(label, Math.max(6, x), Math.max(6, y - 24));
  ctx.restore();
}

function drawOverlayTag(text, x, y) {
  ctx.save();
  ctx.font = "600 12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  const textWidth = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(27, 28, 27, 0.78)";
  roundedRect(ctx, x, y, textWidth + 14, 22, 6);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x + 7, y + 15);
  ctx.restore();
}

function cameraFrameBlob(roi) {
  if (!camera.videoWidth || !camera.videoHeight || !roiContext || !roi) return Promise.resolve(null);
  const width = Math.max(1, roi.x2 - roi.x1);
  const height = Math.max(1, roi.y2 - roi.y1);
  roiCanvas.width = width;
  roiCanvas.height = height;
  roiContext.drawImage(camera, roi.x1, roi.y1, width, height, 0, 0, width, height);
  if (AI_INPUT_MIRRORED) {
    yoloCanvas.width = width;
    yoloCanvas.height = height;
    yoloContext.save();
    yoloContext.translate(width, 0);
    yoloContext.scale(-1, 1);
    yoloContext.drawImage(roiCanvas, 0, 0);
    yoloContext.restore();
    return new Promise((resolve) => yoloCanvas.toBlob(resolve, "image/jpeg", 0.95));
  }
  return new Promise((resolve) => roiCanvas.toBlob(resolve, "image/jpeg", 0.95));
}

async function predictCameraFrame(timestamp) {
  if (predictionInFlight || timestamp - lastPredictionRequestTime < YOLO_REQUEST_INTERVAL_MS) return;
  lastPredictionRequestTime = timestamp;
  const handResults = localizeHands(timestamp);
  const faceResults = localizeFace(timestamp);
  const rois = handCandidates(handResults, faceResults);
  const structure = structuralFeatures(handResults, faceResults);
  lastTrackingState = { structure, rois };
  drawTrackingOverlay();
  if (!rois.length) {
    frameCounter += 1;
    submitNoHandFrame(timestamp, faceResults);
    return;
  }
  predictionInFlight = true;
  const blobs = await Promise.all(rois.map((roi) => cameraFrameBlob(roi)));
  const candidateFrames = rois
    .map((roi, index) => ({ roi, blob: blobs[index] }))
    .filter((candidate) => candidate.blob);
  if (!candidateFrames.length) {
    predictionInFlight = false;
    return;
  }
  const formData = new FormData();
  resetClassifierInputPreviewUrls();
  frameCounter += 1;
  candidateFrames.forEach((candidate) => {
    formData.append("candidates", candidate.blob, `${candidate.roi.type}-candidate.jpg`);
    classifierInputPreviewUrls.set(candidate.roi.type, URL.createObjectURL(candidate.blob));
  });
  formData.append("frame_id", String(frameCounter));
  formData.append("mirrored", String(AI_INPUT_MIRRORED));
  formData.append("source_width", String(candidateFrames[0].roi.sourceWidth));
  formData.append("source_height", String(candidateFrames[0].roi.sourceHeight));
  formData.append("hands_detected", String(candidateFrames[0].roi.handsDetected));
  formData.append("handedness", candidateFrames[0].roi.handedness.join(","));
  formData.append("timestamp_ms", String(Math.round(timestamp)));
  formData.append("structure_json", JSON.stringify(structure));
  formData.append("candidates_json", JSON.stringify(candidateFrames.map((candidate) => ({
    type: candidate.roi.type,
    x1: candidate.roi.x1,
    y1: candidate.roi.y1,
    x2: candidate.roi.x2,
    y2: candidate.roi.y2,
    source_width: candidate.roi.sourceWidth,
    source_height: candidate.roi.sourceHeight,
    motion_score: candidate.roi.motion_score || 0,
    pose_state: candidate.roi.pose_state || lastPoseState,
  }))));
  try {
    const response = await fetch("/api/translator/predict-sign/", {
      method: "POST",
      headers: { "X-CSRFToken": csrfToken() },
      body: formData,
    });
    const result = await response.json();
    lastPredictionResult = result;
    handlePrediction(result);
    recordEvaluationSample(result);
  } catch (error) {
    console.error("[ISYARA Translator] prediction failed", error);
    detectedText.textContent = "Layanan penerjemah belum tersedia.";
  } finally {
    predictionInFlight = false;
  }
}

function resetClassifierInputPreviewUrls() {
  classifierInputPreviewUrls.forEach((url) => URL.revokeObjectURL(url));
  classifierInputPreviewUrls = new Map();
}

async function submitNoHandFrame(timestamp, faceResults) {
  if (predictionInFlight) return;
  predictionInFlight = true;
  const formData = new FormData();
  formData.append("frame_id", String(frameCounter));
  formData.append("mirrored", String(AI_INPUT_MIRRORED));
  formData.append("source_width", String(camera.videoWidth));
  formData.append("source_height", String(camera.videoHeight));
  formData.append("hands_detected", "0");
  formData.append("timestamp_ms", String(Math.round(timestamp)));
  formData.append("structure_json", JSON.stringify(structuralFeatures(null, faceResults)));
  try {
    const response = await fetch("/api/translator/predict-sign/", {
      method: "POST",
      headers: { "X-CSRFToken": csrfToken() },
      body: formData,
    });
    handlePrediction(await response.json());
  } catch (error) {
    handlePrediction({
      status: "no_hand",
      detected: false,
      reason: "no_hand",
      display_text: "Tangan belum terdeteksi",
      confidence: null,
      raw_predictions: [],
      hands_detected: 0,
      handedness: [],
      image_width: camera.videoWidth,
      image_height: camera.videoHeight,
      frame_id: String(frameCounter),
      mirrored: AI_INPUT_MIRRORED,
    });
  } finally {
    predictionInFlight = false;
  }
}

function handlePrediction(result) {
  if (result.status === "model_unavailable" || result.status === "service_unavailable") {
    setPill(motionStatus, "warn", "! Model BISINDO belum siap");
    detectedText.textContent = result.display_text || "Model penerjemah sedang disiapkan.";
    detectedConfidence.textContent = "Confidence: -";
    drawTrackingOverlay(result);
    updateDebug(result);
    return;
  }
  if (result.status === "invalid_image" || result.status === "no_hand") {
    detectedText.textContent = result.display_text || "Frame kamera tidak valid.";
    detectedConfidence.textContent = "Confidence: -";
    drawTrackingOverlay(result);
    updateDebug(result);
    return;
  }
  setPill(aiStatus, "ok", "✓ Model BISINDO siap");
  setPill(motionStatus, "ok", "✓ Inference aktif");
  drawTrackingOverlay(result);
  const confidence = Number(result.confidence || 0);
  const label = result.stable_prediction || result.display_label || result.label || result.prediction;
  if (result.accepted && result.accepted_prediction) {
    result.transcript_append_client = true;
    acceptPrediction({
      label: result.accepted_prediction,
      displayText: result.accepted_prediction,
      confidence: result.accepted_confidence ?? confidence,
    });
    detectedText.textContent = `${result.accepted_prediction} · ${Math.round(Number(result.accepted_confidence ?? confidence) * 100)}%`;
    detectedConfidence.textContent = `Confidence: ${Math.round(Number(result.accepted_confidence ?? confidence) * 100)}%`;
    updateDebug(result, result.accepted_prediction, result.suppressed);
    return;
  }
  if (!result.detected || !label || confidence < YOLO_CONFIDENCE_THRESHOLD) {
    detectedText.textContent = "Mendeteksi...";
    detectedConfidence.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
    updateDebug(result);
    return;
  }
  detectedText.textContent = result.stable_prediction ? `Terdeteksi: ${result.stable_prediction}` : "Mendeteksi...";
  detectedConfidence.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
  updateDebug(result, result.stable_prediction || null, result.suppressed);
}

function acceptPrediction(result) {
  acceptedTokens.push(result.displayText);
  renderTranscript();
  saveHistory("SIGN_TO_SPEECH", result.label, transcriptSentence(false), result.confidence);
  console.info("[ISYARA Translator] transcript_append", {
    label: result.label,
    confidence: result.confidence,
    transcript: transcriptSentence(false),
  });
  return true;
}

function predictionLoop(timestamp) {
  requestAnimationFrame(predictionLoop);
  if (!camera.videoWidth || camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  if (!syncCanvas()) return;
  predictCameraFrame(timestamp);
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    camera.srcObject = stream;
    camera.style.transform = DISPLAY_FRAME_MIRRORED ? "scaleX(-1)" : "";
    await camera.play();
    setPill(cameraStatus, "ok", "✓ Kamera siap");
    requestAnimationFrame(predictionLoop);
  } catch (error) {
    console.error("[ISYARA Translator] camera failed", error);
    camera.classList.add("hidden");
    fallback.classList.remove("hidden");
    fallback.classList.add("grid");
    setPill(cameraStatus, "warn", "! Kamera tidak tersedia");
  }
}

function speak(text) {
  if (!("speechSynthesis" in window) || !text.trim()) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "id-ID";
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

async function saveHistory(direction, sourceText, translatedText, confidence = null) {
  if (!translatedText?.trim()) return;
  try {
    await fetch("/api/translator/history/", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken() },
      body: JSON.stringify({ direction, source_text: sourceText, translated_text: translatedText, confidence }),
    });
  } catch (error) {
    console.info("[ISYARA Translator] history save skipped", error);
  }
}

function switchMode(mode) {
  const signMode = mode === "sign";
  signModePanel.classList.toggle("hidden", !signMode);
  speechModePanel.classList.toggle("hidden", signMode);
  modeSign.className = signMode ? "rounded-md bg-[var(--color-primary)] px-4 py-3 text-sm font-semibold text-white" : "rounded-md px-4 py-3 text-sm font-semibold text-[var(--color-dark)]";
  modeSpeech.className = !signMode ? "rounded-md bg-[var(--color-primary)] px-4 py-3 text-sm font-semibold text-white" : "rounded-md px-4 py-3 text-sm font-semibold text-[var(--color-dark)]";
  modeSign.setAttribute("aria-pressed", String(signMode));
  modeSpeech.setAttribute("aria-pressed", String(!signMode));
}

async function startSpeechRecording() {
  if (startBrowserSpeechRecognition()) return;
  try {
    const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    speechChunks = [];
    const recorderOptions = MediaRecorder.isTypeSupported("audio/webm") ? { mimeType: "audio/webm" } : {};
    speechRecorder = new MediaRecorder(audioStream, recorderOptions);
    speechRecorder.ondataavailable = (event) => {
      if (event.data.size) speechChunks.push(event.data);
    };
    speechRecorder.onstop = submitSpeechRecording;
    speechRecorder.start();
    microphoneStatus.textContent = "● Merekam...";
    recordSpeech.disabled = true;
    stopSpeech.disabled = false;
  } catch (error) {
    console.error("[ISYARA Translator] microphone failed", error);
    microphoneStatus.textContent = "! Mikrofon tidak tersedia";
  }
}

function startBrowserSpeechRecognition() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return false;
  if (speechRecognitionActive) return true;

  speechRecognitionText = "";
  speechRecognition = new Recognition();
  speechRecognition.lang = "id-ID";
  speechRecognition.continuous = false;
  speechRecognition.interimResults = true;
  speechRecognition.maxAlternatives = 1;

  speechRecognition.onstart = () => {
    speechRecognitionActive = true;
    microphoneStatus.textContent = "● Merekam...";
    speechTranscript.textContent = "Dengarkan ucapan...";
    recordSpeech.disabled = true;
    stopSpeech.disabled = false;
  };

  speechRecognition.onresult = (event) => {
    let finalText = "";
    let interimText = "";
    for (let index = 0; index < event.results.length; index += 1) {
      const transcript = event.results[index][0]?.transcript || "";
      if (event.results[index].isFinal) {
        finalText += transcript;
      } else {
        interimText += transcript;
      }
    }
    speechRecognitionText = (finalText || interimText).trim();
    speechTranscript.textContent = speechRecognitionText || "Dengarkan ucapan...";
  };

  speechRecognition.onerror = (event) => {
    console.error("[ISYARA Translator] browser speech recognition failed", event.error);
    speechRecognitionActive = false;
    recordSpeech.disabled = false;
    stopSpeech.disabled = true;
    microphoneStatus.textContent = event.error === "not-allowed" ? "! Mikrofon tidak diizinkan" : "! Transkripsi gagal";
    if (!speechRecognitionText) {
      speechTranscript.textContent = event.error === "not-allowed" ? "Izinkan akses mikrofon untuk transkripsi." : "Suara belum berhasil ditranskripsi.";
    }
  };

  speechRecognition.onend = () => {
    speechRecognitionActive = false;
    recordSpeech.disabled = false;
    stopSpeech.disabled = true;
    const text = speechRecognitionText.trim();
    if (!text) {
      microphoneStatus.textContent = "○ Belum ada suara dikenali";
      speechTranscript.textContent = "Tidak ada suara yang dikenali.";
      return;
    }
    microphoneStatus.textContent = "✓ Transkripsi selesai";
    speechTranscript.textContent = text;
    saveHistory("SPEECH_TO_TEXT", "audio", text, null);
  };

  try {
    speechRecognition.start();
    return true;
  } catch (error) {
    console.info("[ISYARA Translator] browser speech recognition unavailable at runtime", error);
    speechRecognitionActive = false;
    return false;
  }
}

function stopSpeechInput() {
  if (speechRecognitionActive && speechRecognition) {
    speechRecognition.stop();
    return;
  }
  if (speechRecorder && speechRecorder.state !== "inactive") {
    speechRecorder.stop();
  }
}

async function submitSpeechRecording() {
  microphoneStatus.textContent = "○ Memproses suara...";
  recordSpeech.disabled = false;
  stopSpeech.disabled = true;
  const blob = new Blob(speechChunks, { type: "audio/webm" });
  const formData = new FormData();
  formData.append("audio", blob, "speech.webm");
  try {
    const response = await fetch("/api/translator/transcribe/", {
      method: "POST",
      headers: { "X-CSRFToken": csrfToken() },
      body: formData,
    });
    const result = await response.json();
    if (result.status && result.status !== "ok") {
      speechTranscript.textContent = result.message || "Model transkripsi sedang disiapkan.";
      microphoneStatus.textContent = "○ Transkripsi belum tersedia";
      return;
    }
    speechTranscript.textContent = result.text || "Tidak ada suara yang dikenali.";
    microphoneStatus.textContent = "✓ Transkripsi selesai";
    if (result.text) saveHistory("SPEECH_TO_TEXT", "audio", result.text, null);
  } catch (error) {
    console.error("[ISYARA Translator] transcription failed", error);
    speechTranscript.textContent = "Layanan transkripsi belum tersedia.";
    microphoneStatus.textContent = "! Transkripsi gagal";
  }
}

function initEvaluationPanel() {
  if (!evalPanel || !evalTargetSign || !evalCaptureSample || !evalReset) return;
  evalTargetSign.innerHTML = SUPPORTED_SIGNS.map((sign) => `<option value="${sign}">${sign}</option>`).join("");
  evalCaptureSample.addEventListener("click", captureEvaluationSample);
  evalReset.addEventListener("click", () => {
    evalSamples = [];
    renderEvaluation();
  });
  renderEvaluation();
}

function captureEvaluationSample() {
  evalCaptureTarget = evalTargetSign.value;
  evalCaptureRemaining = 10;
  evalCaptureSample.disabled = true;
  evalCaptureSample.textContent = "Capturing...";
  renderEvaluation();
}

function recordEvaluationSample(result) {
  if (!evalPanel || evalCaptureRemaining <= 0 || !evalCaptureTarget) return;
  const target = evalCaptureTarget;
  const predicted = result.display_label || result.label || result.prediction || "NO_DETECTION";
  evalSamples.push({
    target,
    predicted,
    confidence: Number(result.confidence || 0),
    correct: predicted === target,
    accepted: Boolean(result.accepted),
    rejection: result.rejection_reason || "",
  });
  evalCaptureRemaining -= 1;
  if (evalCaptureRemaining <= 0) {
    evalCaptureTarget = null;
    evalCaptureSample.disabled = false;
    evalCaptureSample.textContent = "Capture 10 samples";
  }
  renderEvaluation();
}

function renderEvaluation() {
  if (!evalSummary || !evalLog) return;
  if (!evalSamples.length) {
    evalSummary.textContent = "No samples yet.";
    evalLog.textContent = "";
    return;
  }
  const target = evalTargetSign?.value;
  const scoped = evalSamples.filter((sample) => sample.target === target);
  const correct = scoped.filter((sample) => sample.correct).length;
  const wrongCounts = scoped.reduce((counts, sample) => {
    if (!sample.correct) counts[sample.predicted] = (counts[sample.predicted] || 0) + 1;
    return counts;
  }, {});
  const wrongText = Object.entries(wrongCounts).map(([label, count]) => `${label}: ${count}`).join(", ") || "none";
  const active = evalCaptureRemaining > 0 ? ` Capturing ${evalCaptureRemaining} more for ${evalCaptureTarget}.` : "";
  evalSummary.textContent = `Target: ${target}. Samples: ${scoped.length}. Correct: ${correct}. Webcam accuracy: ${scoped.length ? Math.round((correct / scoped.length) * 100) : 0}%. Wrong: ${wrongText}.${active}`;
  evalLog.textContent = evalSamples
    .slice(-80)
    .map((sample, index) => `${index + 1}. target=${sample.target} predicted=${sample.predicted} conf=${Math.round(sample.confidence * 100)}% accepted=${sample.accepted} correct=${sample.correct} ${sample.rejection}`)
    .join("\n");
}

modeSign.addEventListener("click", () => switchMode("sign"));
modeSpeech.addEventListener("click", () => switchMode("speech"));
speakSignText.addEventListener("click", () => speak(transcriptSentence(true)));
clearSignTranscript.addEventListener("click", () => {
  acceptedTokens = [];
  renderTranscript();
});
restartSignTranslator.addEventListener("click", () => {
  acceptedTokens = [];
  detectedText.textContent = "Mendeteksi gerakan...";
  detectedConfidence.textContent = "Confidence: -";
  clearOverlay();
  updateDebug({ raw_predictions: [] });
  renderTranscript();
});
recordSpeech.addEventListener("click", startSpeechRecording);
stopSpeech.addEventListener("click", stopSpeechInput);
copySpeechText.addEventListener("click", () => navigator.clipboard?.writeText(speechTranscript.textContent || ""));
camera.addEventListener("loadedmetadata", syncCanvas);
window.addEventListener("resize", syncCanvas);

initEvaluationPanel();
startCamera();
initHandLocalizer();
