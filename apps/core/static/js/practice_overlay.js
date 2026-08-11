const video = document.getElementById("camera");
const canvas = document.getElementById("landmarkOverlay");
const fallback = document.getElementById("cameraFallback");
const mediaPipeStatus = document.getElementById("mediaPipeStatus");
const cameraStatus = document.getElementById("cameraStatus");
const guidance = document.getElementById("liveGuidance");
const startRecordingButton = document.getElementById("startRecording");
const finishRecordingButton = document.getElementById("finishRecording");
const analyzeButton = document.getElementById("analyzeButton");
const recordingStatus = document.getElementById("recordingStatus");
const countdownOverlay = document.getElementById("countdownOverlay");
const trackingUnavailable = document.getElementById("trackingUnavailable");
const retryTrackingButton = document.getElementById("retryTracking");
const trackingTelemetry = document.getElementById("trackingTelemetry");
const mirrorReferenceToggle = document.getElementById("mirrorReferenceToggle");
const referenceVideo = document.getElementById("referenceVideo");
const aiReferenceConfig = JSON.parse(document.getElementById("aiReferenceConfig")?.textContent || "{}");

const ctx = canvas.getContext("2d");
const DEBUG_TRACKING = true;
const MIRROR_WEBCAM = true;
const MEDIAPIPE_VERSION = "0.10.35";
const MEDIAPIPE_PACKAGE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const MEDIAPIPE_IMPORT_URL = `${MEDIAPIPE_PACKAGE_URL}/vision_bundle.mjs`;
const MEDIAPIPE_WASM_URL = `${MEDIAPIPE_PACKAGE_URL}/wasm`;
const MODEL_URLS = {
  hand: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
};
const PROCESS_INTERVAL_MS = 85;
const TRAJECTORY_WINDOW_MS = 900;
const STABILIZE_FRAMES = 5;
const MIN_POSE_VISIBILITY = 0.45;
const SHOULDER_WIDTH_FALLBACK = 0.33;
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];
const FINGER_VECTORS = [
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
];
const ORIENTATION_VECTORS = [[0, 5], [0, 17]];
const POSE_CONNECTIONS = [[11, 13], [13, 15], [12, 14], [14, 16], [11, 12]];
const POSE_VECTOR_CONNECTIONS = [[11, 13], [13, 15], [12, 14], [14, 16]];
const POSE_POINTS = [11, 12, 13, 14, 15, 16];
const SELECTED_FACE = [1, 4, 13, 14, 33, 61, 105, 133, 152, 159, 263, 291, 334, 362, 386];
const OVERLAY_STYLE = {
  rightHand: "rgba(20, 184, 166, 0.96)",
  leftHand: "rgba(96, 165, 250, 0.96)",
  handPoint: "#ffffff",
  fingerVector: "rgba(250, 204, 21, 0.95)",
  orientationVector: "rgba(244, 63, 94, 0.95)",
  pose: "rgba(45, 212, 191, 0.95)",
  posePoint: "#99f6e4",
  facePoint: "rgba(251, 191, 36, 0.95)",
  locationVector: "rgba(168, 85, 247, 0.86)",
  trajectoryRight: "rgba(56, 189, 248, ALPHA)",
  trajectoryLeft: "rgba(34, 197, 94, ALPHA)",
  shadow: "rgba(15, 23, 42, 0.85)",
};

let handLandmarker;
let poseLandmarker;
let faceLandmarker;
let stream;
let mediaRecorder;
let recordedChunks = [];
let recording = false;
let trackingEnabled = false;
let animationFrameId = null;
let lastProcessTime = 0;
let lastTelemetryTime = 0;
let processedFrames = 0;
let fps = 0;
let stable = {};
let mediaPipeInitializing = false;
let mediaPipeReady = false;
let cameraReady = false;
let trajectories = { Left: [], Right: [], Unknown: [] };

if (analyzeButton) {
  analyzeButton.disabled = true;
  analyzeButton.classList.add("opacity-60", "cursor-not-allowed");
}

function logCv(message) {
  console.info(`[ISYARA CV] ${message}`);
}

async function runCvStage(label, callback) {
  logCv(`${label}: START`);
  try {
    const result = await callback();
    logCv(`${label}: SUCCESS`);
    return result;
  } catch (error) {
    console.error(`[ISYARA CV] ${label}: FAILURE`, error);
    throw error;
  }
}

async function probeAsset(url) {
  try {
    const response = await fetch(url, { method: "HEAD", mode: "cors", cache: "no-store" });
    logCv(`Asset probe ${response.ok ? "OK" : "failed"} URL: ${url} HTTP status: ${response.status}`);
    return { url, ok: response.ok, status: response.status };
  } catch (error) {
    logCv(`Asset probe failed URL: ${url} HTTP status: network-error Original error: ${error?.message || error}`);
    return { url, ok: false, status: "network-error", error };
  }
}

function setTrackingReady() {
  trackingUnavailable.classList.add("hidden");
  mediaPipeStatus.textContent = "Pelacakan AI siap";
  mediaPipeStatus.className = "rounded bg-teal-100 px-2 py-1 text-xs font-black text-teal-800";
}

function setTrackingFailed(error, failedAsset = null) {
  console.error("[ISYARA CV] MediaPipe initialization failed", {
    url: failedAsset?.url || "unknown",
    httpStatus: failedAsset?.status || "unknown",
    browser: navigator.userAgent,
    originalError: error,
  });
  trackingEnabled = false;
  trackingUnavailable.classList.remove("hidden");
  mediaPipeStatus.textContent = "Pelacakan AI tidak tersedia";
  mediaPipeStatus.className = "rounded bg-amber-100 px-2 py-1 text-xs font-black text-amber-800";
  guidance.innerHTML = '<p class="guidance-line neutral">Panduan langsung akan muncul saat pelacakan AI siap.</p>';
}

function setStatus(elementId, ok, okText, warnText) {
  const element = document.getElementById(elementId);
  element.textContent = ok ? `OK ${okText}` : `! ${warnText}`;
  element.classList.toggle("ok", ok);
  element.classList.toggle("warn", !ok);
}

function setNeutralStatus(elementId, text) {
  const element = document.getElementById(elementId);
  element.textContent = `○ ${text}`;
  element.classList.remove("ok", "warn");
}

function stableCondition(key, active) {
  const current = stable[key] || { value: false, count: 0 };
  if (current.value === active) {
    current.count += 1;
  } else {
    current.value = active;
    current.count = 1;
  }
  stable[key] = current;
  return current.value && current.count >= STABILIZE_FRAMES;
}

function resetTrackingBuffers() {
  trajectories = { Left: [], Right: [], Unknown: [] };
  stable = {};
  processedFrames = 0;
  fps = 0;
}

function syncCanvasToVideo() {
  if (!video.videoWidth || !video.videoHeight) return false;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  return true;
}

function landmarkPoint(landmark) {
  return {
    x: (MIRROR_WEBCAM ? 1 - landmark.x : landmark.x) * canvas.width,
    y: landmark.y * canvas.height,
    z: landmark.z || 0,
  };
}

function canonicalPoint(landmark) {
  return {
    x: landmark.x,
    y: landmark.y,
    z: landmark.z || 0,
  };
}

function posePointVisible(landmarks, index) {
  const landmark = landmarks?.[index];
  return Boolean(landmark && (landmark.visibility ?? 1) >= MIN_POSE_VISIBILITY);
}

function drawLine(a, b, color, width = 3) {
  ctx.save();
  ctx.shadowColor = OVERLAY_STYLE.shadow;
  ctx.shadowBlur = 5;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.stroke();
  ctx.restore();
}

function drawDot(p, color, radius = 4) {
  ctx.save();
  ctx.shadowColor = OVERLAY_STYLE.shadow;
  ctx.shadowBlur = 5;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

function drawVector(a, b, color, width = 2.5) {
  drawLine(a, b, color, width);
  const angle = Math.atan2(b.y - a.y, b.x - a.x);
  const size = 10;
  ctx.beginPath();
  ctx.moveTo(b.x, b.y);
  ctx.lineTo(b.x - size * Math.cos(angle - Math.PI / 6), b.y - size * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(b.x - size * Math.cos(angle + Math.PI / 6), b.y - size * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function handCenter(points) {
  const centerIndexes = [0, 5, 9, 13, 17];
  return centerIndexes.reduce(
    (acc, index) => ({ x: acc.x + points[index].x / centerIndexes.length, y: acc.y + points[index].y / centerIndexes.length }),
    { x: 0, y: 0 },
  );
}

function drawHandLandmarks(hand, handedness) {
  const label = handedness || "Unknown";
  const color = label === "Right" ? OVERLAY_STYLE.rightHand : OVERLAY_STYLE.leftHand;
  const points = hand.map(landmarkPoint);

  HAND_CONNECTIONS.forEach(([a, b]) => drawLine(points[a], points[b], color, 3));
  FINGER_VECTORS.forEach(([a, b]) => drawVector(points[a], points[b], OVERLAY_STYLE.fingerVector, 2));
  ORIENTATION_VECTORS.forEach(([a, b]) => drawVector(points[a], points[b], OVERLAY_STYLE.orientationVector, 2.5));
  points.forEach((p, index) => drawDot(p, index === 0 ? OVERLAY_STYLE.handPoint : color, index === 0 ? 6 : 4));

  return { label, points, center: handCenter(points), wrist: points[0] };
}

function analyzeHand(hand, handedness, anchors, shoulderWidth) {
  const points = hand.map(canonicalPoint);
  const centerIndexes = [0, 5, 9, 13, 17];
  const center = centerIndexes.reduce(
    (acc, index) => ({ x: acc.x + points[index].x / centerIndexes.length, y: acc.y + points[index].y / centerIndexes.length }),
    { x: 0, y: 0 },
  );
  return {
    label: handedness || "Unknown",
    points,
    center,
    wrist: points[0],
    shoulderWidth,
    anchors,
    fingerAngles: {
      index_pip: jointAngle(points[5], points[6], points[7]),
      middle_pip: jointAngle(points[9], points[10], points[11]),
      ring_pip: jointAngle(points[13], points[14], points[15]),
      pinky_pip: jointAngle(points[17], points[18], points[19]),
    },
    palmOrientation: Math.atan2(points[5].y - points[0].y, points[5].x - points[0].x),
  };
}

function jointAngle(a, b, c) {
  if (!a || !b || !c) return null;
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.hypot(ab.x, ab.y);
  const magCB = Math.hypot(cb.x, cb.y);
  if (!magAB || !magCB) return null;
  const cosine = Math.max(-1, Math.min(1, dot / (magAB * magCB)));
  return Math.acos(cosine) * (180 / Math.PI);
}

function angleDifference(a, b) {
  let diff = Math.abs(a - b);
  while (diff > Math.PI) diff = Math.abs(diff - Math.PI * 2);
  return diff;
}

function handName(key) {
  return key === "right" ? "kanan" : "kiri";
}

function fingerName(name) {
  return {
    index: "telunjuk",
    middle: "tengah",
    ring: "manis",
    pinky: "kelingking",
  }[name.replace("_pip", "")] || "jari";
}

function requiredHands() {
  return Array.isArray(aiReferenceConfig.required_hands) ? aiReferenceConfig.required_hands : ["right"];
}

function preferredActiveHandLabel() {
  if (aiReferenceConfig.active_hand === "left") return "Left";
  return "Right";
}

function isHandRequired(hand) {
  return requiredHands().includes(hand);
}

function buildCanonicalAnchors(poseLandmarks, faceLandmarks) {
  const anchors = {};
  if (poseLandmarks && posePointVisible(poseLandmarks, 11) && posePointVisible(poseLandmarks, 12)) {
    const leftShoulder = canonicalPoint(poseLandmarks[11]);
    const rightShoulder = canonicalPoint(poseLandmarks[12]);
    anchors.left_shoulder = leftShoulder;
    anchors.right_shoulder = rightShoulder;
    anchors.shoulder_midpoint = {
      x: (leftShoulder.x + rightShoulder.x) / 2,
      y: (leftShoulder.y + rightShoulder.y) / 2,
    };
    anchors.shoulderWidth = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
  }
  if (faceLandmarks) {
    if (faceLandmarks[152]) anchors.chin = canonicalPoint(faceLandmarks[152]);
    if (faceLandmarks[1]) anchors.nose = canonicalPoint(faceLandmarks[1]);
    if (faceLandmarks[10]) anchors.forehead = canonicalPoint(faceLandmarks[10]);
    if (faceLandmarks[13] && faceLandmarks[14]) {
      anchors.mouth_openness = Math.hypot(faceLandmarks[13].x - faceLandmarks[14].x, faceLandmarks[13].y - faceLandmarks[14].y);
    }
  }
  return anchors;
}

function normalizedLocationError(hand, target) {
  const anchor = hand.anchors[target.anchor] || hand.anchors.shoulder_midpoint;
  if (!anchor) return null;
  const scale = hand.shoulderWidth || SHOULDER_WIDTH_FALLBACK;
  const learner = {
    x: (hand.center.x - anchor.x) / scale,
    y: (hand.center.y - anchor.y) / scale,
  };
  return {
    error: Math.hypot(learner.x - target.x, learner.y - target.y),
    dx: learner.x - target.x,
    dy: learner.y - target.y,
    threshold: target.threshold ?? 0.25,
  };
}

function fingerConfigurationError(hand, targets = {}) {
  const errors = Object.entries(targets)
    .map(([name, target]) => {
      const learner = hand.fingerAngles[name];
      if (learner == null) return null;
      return { name, error: Math.abs(learner - target), learner, target };
    })
    .filter(Boolean);
  if (!errors.length) return null;
  errors.sort((a, b) => b.error - a.error);
  return errors[0];
}

function palmOrientationError(hand, target) {
  if (!target || hand.palmOrientation == null) return null;
  return {
    error: angleDifference(hand.palmOrientation, target.wrist_index_angle),
    threshold: target.threshold ?? 0.8,
  };
}

function stabilizeMessage(key, active, message) {
  return stableCondition(key, active) ? message : null;
}

function computeLiveGuidance({ hasFace, hasPose, hasLeftHand, hasRightHand, rightHand, leftHand }) {
  const messages = [];
  const deviations = [];
  const referenceFeatures = aiReferenceConfig.reference_features || {};

  if (isHandRequired("right") && !hasRightHand) {
    const warning = stabilizeMessage("rightMissing", true, ["warn", "Pastikan seluruh tangan kanan terlihat di kamera."]);
    if (warning) messages.push(warning);
  } else {
    stableCondition("rightMissing", false);
  }

  if (isHandRequired("left") && !hasLeftHand) {
    const warning = stabilizeMessage("leftMissing", true, ["warn", "Pastikan seluruh tangan kiri terlihat di kamera."]);
    if (warning) messages.push(warning);
  } else {
    stableCondition("leftMissing", false);
  }

  if (aiReferenceConfig.uses_upper_body && !hasPose) {
    const warning = stabilizeMessage("poseMissing", true, ["warn", "Posisikan tubuh bagian atas di tengah kamera."]);
    if (warning) messages.push(warning);
  } else {
    stableCondition("poseMissing", false);
  }

  if (aiReferenceConfig.uses_face && !hasFace) {
    const warning = stabilizeMessage("faceMissing", true, ["warn", "Pastikan wajah terlihat untuk tanda ini."]);
    if (warning) messages.push(warning);
  } else {
    stableCondition("faceMissing", false);
  }

  if (messages.length) return messages;

  [
    ["right", rightHand],
    ["left", leftHand],
  ].forEach(([key, hand]) => {
    if (!isHandRequired(key) || !hand) return;
    const target = referenceFeatures[key] || {};
    const location = target.location ? normalizedLocationError(hand, target.location) : null;
    if (location && location.error > location.threshold) {
      const vertical = location.dy > 0 ? "higher" : "lower";
      const horizontal = location.dx > 0 ? "left" : "right";
      deviations.push({
        key: `${key}Location`,
        error: location.error / location.threshold,
        message: Math.abs(location.dy) > Math.abs(location.dx)
          ? `${vertical === "higher" ? "Naikkan" : "Turunkan"} tangan ${handName(key)} sedikit.`
          : `Geser tangan ${handName(key)} sedikit ke ${horizontal === "left" ? "kiri" : "kanan"}.`,
      });
    }

    const finger = fingerConfigurationError(hand, target.finger_angles);
    if (finger && finger.error > 25) {
      deviations.push({
        key: `${key}Finger${finger.name}`,
        error: finger.error / 25,
        message: finger.learner < finger.target ? `Luruskan jari ${fingerName(finger.name)} ${handName(key)}.` : `Rilekskan jari ${fingerName(finger.name)} ${handName(key)} sedikit.`,
      });
    }

    const palm = palmOrientationError(hand, target.palm_orientation);
    if (palm && palm.error > palm.threshold) {
      deviations.push({
        key: `${key}Palm`,
        error: palm.error / palm.threshold,
        message: `Putar telapak ${handName(key)} sedikit ke luar.`,
      });
    }
  });

  deviations.sort((a, b) => b.error - a.error);
  const primary = deviations.find((deviation) => stableCondition(deviation.key, true));
  deviations.forEach((deviation) => {
    if (!primary || deviation.key !== primary.key) stableCondition(deviation.key, false);
  });

  if (primary) {
    messages.push(["warn", primary.message]);
  } else {
    messages.push(["ok", "Posisi tangan sudah baik."]);
  }

  const fingerKeys = requiredHands().map((hand) => `${hand}FingerOk`).join("-");
  if (stableCondition(fingerKeys, deviations.every((deviation) => !deviation.key.includes("Finger")))) {
    messages.push(["ok", "Posisi jari sudah baik."]);
  }

  return messages.length ? messages : [["ok", "Pelacakan sudah baik."]];
}

function updateRequirementStatuses({ hasFace, hasPose, hasLeftHand, hasRightHand }) {
  if (isHandRequired("right")) {
    setStatus("statusRightHand", hasRightHand, "Tangan kanan terdeteksi", "Tangan kanan tidak terlihat");
  } else {
    setNeutralStatus("statusRightHand", "Tangan kanan tidak diperlukan");
  }

  if (isHandRequired("left")) {
    setStatus("statusLeftHand", hasLeftHand, "Tangan kiri terdeteksi", "Tangan kiri tidak terlihat");
  } else {
    setNeutralStatus("statusLeftHand", "Tangan kiri tidak diperlukan");
  }

  if (aiReferenceConfig.uses_upper_body) {
    setStatus("statusPose", hasPose, "Tubuh bagian atas terdeteksi", "Tubuh bagian atas tidak terlihat");
  } else {
    setNeutralStatus("statusPose", "Tubuh bagian atas tidak diperlukan");
  }

  if (aiReferenceConfig.uses_face) {
    setStatus("statusFace", hasFace, "Wajah terdeteksi", "Wajah tidak terlihat");
  } else {
    setNeutralStatus("statusFace", "Wajah tidak diperlukan");
  }
}

function drawPoseLandmarks(landmarks) {
  POSE_CONNECTIONS.forEach(([a, b]) => {
    if (posePointVisible(landmarks, a) && posePointVisible(landmarks, b)) {
      drawLine(landmarkPoint(landmarks[a]), landmarkPoint(landmarks[b]), OVERLAY_STYLE.pose, 4);
    }
  });
  POSE_VECTOR_CONNECTIONS.forEach(([a, b]) => {
    if (posePointVisible(landmarks, a) && posePointVisible(landmarks, b)) {
      drawVector(landmarkPoint(landmarks[a]), landmarkPoint(landmarks[b]), OVERLAY_STYLE.pose, 2.5);
    }
  });
  POSE_POINTS.forEach((index) => {
    if (posePointVisible(landmarks, index)) drawDot(landmarkPoint(landmarks[index]), OVERLAY_STYLE.posePoint, 6);
  });
}

function drawFaceLandmarks(landmarks) {
  SELECTED_FACE.forEach((index) => {
    if (landmarks[index]) drawDot(landmarkPoint(landmarks[index]), OVERLAY_STYLE.facePoint, 3.5);
  });
}

function drawSigningLocationVector(anchor, handInfo) {
  if (!anchor || !handInfo) return;
  drawVector(anchor, handInfo.center, OVERLAY_STYLE.locationVector, 3);
}

function updateTrajectory(label, point, timestamp) {
  const key = trajectories[label] ? label : "Unknown";
  trajectories[key].push({ x: point.x, y: point.y, timestamp });
  trajectories[key] = trajectories[key].filter((item) => timestamp - item.timestamp <= TRAJECTORY_WINDOW_MS);
}

function drawTrajectory(label) {
  const points = trajectories[label] || [];
  if (points.length < 2) return;
  const colorTemplate = label === "Right" ? OVERLAY_STYLE.trajectoryRight : OVERLAY_STYLE.trajectoryLeft;
  for (let index = 1; index < points.length; index += 1) {
    const alpha = index / points.length;
    drawLine(points[index - 1], points[index], colorTemplate.replace("ALPHA", String(0.18 + alpha * 0.62)), 2 + alpha * 3);
  }
}

function updateGuidance({ hasFace, hasPose, hasLeftHand, hasRightHand, rightHand, leftHand }) {
  guidance.innerHTML = computeLiveGuidance({ hasFace, hasPose, hasLeftHand, hasRightHand, rightHand, leftHand })
    .slice(0, 2)
    .map(([type, text]) => `<p class="guidance-line ${type}">${text}</p>`)
    .join("");
}

function updateTelemetry({ handCount, hasPose, hasFace, inferenceMs }) {
  if (!DEBUG_TRACKING || !trackingTelemetry) return;
  trackingTelemetry.classList.remove("hidden");
  trackingTelemetry.textContent = `FPS: ${fps || "--"} · Tangan: ${handCount} · Tubuh: ${hasPose ? "ya" : "tidak"} · Wajah: ${hasFace ? "ya" : "tidak"} · Inferensi: ${Math.round(inferenceMs)}ms`;
}

function detectFrame(timestamp) {
  const inferenceStart = performance.now();
  const hands = handLandmarker.detectForVideo(video, timestamp);
  const pose = poseLandmarker.detectForVideo(video, timestamp);
  const face = faceLandmarker.detectForVideo(video, timestamp);
  return {
    hands,
    pose,
    face,
    inferenceMs: performance.now() - inferenceStart,
  };
}

function drawOverlay(results, timestamp) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const handLandmarks = results.hands.landmarks || [];
  const handednesses = results.hands.handednesses || [];
  const poseLandmarks = results.pose.landmarks?.[0];
  const faceLandmarks = results.face.faceLandmarks?.[0];
  const hasPose = POSE_POINTS.every((index) => posePointVisible(poseLandmarks, index));
  const hasFace = Boolean(faceLandmarks);
  let hasLeftHand = false;
  let hasRightHand = false;
  let rightHand = null;
  let leftHand = null;
  let activeHand = null;
  const anchors = buildCanonicalAnchors(poseLandmarks, faceLandmarks);
  const shoulderWidth = anchors.shoulderWidth || SHOULDER_WIDTH_FALLBACK;

  if (hasPose) drawPoseLandmarks(poseLandmarks);
  if (hasFace) drawFaceLandmarks(faceLandmarks);

  const bodyAnchor = hasPose
    ? {
        x: (landmarkPoint(poseLandmarks[11]).x + landmarkPoint(poseLandmarks[12]).x) / 2,
        y: (landmarkPoint(poseLandmarks[11]).y + landmarkPoint(poseLandmarks[12]).y) / 2,
      }
    : faceLandmarks
      ? landmarkPoint(faceLandmarks[1])
      : null;

  handLandmarks.forEach((hand, index) => {
    const label = handednesses[index]?.[0]?.categoryName || "Unknown";
    const handInfo = drawHandLandmarks(hand, label);
    const analysisHand = analyzeHand(hand, label, anchors, shoulderWidth);
    hasRightHand ||= label === "Right";
    hasLeftHand ||= label === "Left";
    if (label === "Right") rightHand = analysisHand;
    if (label === "Left") leftHand = analysisHand;
    activeHand = label === preferredActiveHandLabel() || !activeHand ? handInfo : activeHand;
    updateTrajectory(label, handInfo.wrist, timestamp);
  });

  if (activeHand) drawSigningLocationVector(bodyAnchor, activeHand);
  drawTrajectory("Left");
  drawTrajectory("Right");
  drawTrajectory("Unknown");

  updateRequirementStatuses({ hasFace, hasPose, hasLeftHand, hasRightHand });
  updateGuidance({ hasFace, hasPose, hasLeftHand, hasRightHand, rightHand, leftHand, pose: poseLandmarks });
  updateTelemetry({ handCount: handLandmarks.length, hasPose, hasFace, inferenceMs: results.inferenceMs });
}

async function initMediaPipe() {
  if (mediaPipeReady) return;
  if (mediaPipeInitializing) return;
  mediaPipeInitializing = true;
  try {
    logCv(`Import URL: ${MEDIAPIPE_IMPORT_URL}`);
    logCv(`WASM path: ${MEDIAPIPE_WASM_URL}`);
    logCv(`Hand model path: ${MODEL_URLS.hand}`);
    logCv(`Pose model path: ${MODEL_URLS.pose}`);
    logCv(`Face model path: ${MODEL_URLS.face}`);
    mediaPipeStatus.textContent = "Loading MediaPipe";
    trackingUnavailable.classList.add("hidden");

    const visionModule = await runCvStage("Loading tasks-vision module", () => import(MEDIAPIPE_IMPORT_URL));
    const probes = await runCvStage("Resolving WASM", () => Promise.all([
      probeAsset(`${MEDIAPIPE_WASM_URL}/vision_wasm_internal.wasm`),
      probeAsset(MODEL_URLS.hand),
      probeAsset(MODEL_URLS.pose),
      probeAsset(MODEL_URLS.face),
    ]));
    const failedProbe = probes.find((probe) => !probe.ok);
    if (failedProbe) {
      throw new Error(`MediaPipe asset probe failed: ${failedProbe.url} (${failedProbe.status})`);
    }

    const vision = await runCvStage("Resolving WASM fileset", () => visionModule.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL));
    handLandmarker = await runCvStage("Creating HandLandmarker", () => visionModule.HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URLS.hand },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minTrackingConfidence: 0.5,
    }));
    poseLandmarker = await runCvStage("Creating PoseLandmarker", () => visionModule.PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URLS.pose },
      runningMode: "VIDEO",
      minPoseDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    }));
    faceLandmarker = await runCvStage("Creating FaceLandmarker", () => visionModule.FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URLS.face },
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
    }));
    mediaPipeReady = true;
    setTrackingReady();
    startPredictionLoopIfReady();
  } catch (error) {
    setTrackingFailed(error);
  } finally {
    mediaPipeInitializing = false;
  }
}

function predictionLoop(timestamp) {
  if (!trackingEnabled) return;
  animationFrameId = requestAnimationFrame(predictionLoop);
  if (!video.videoWidth || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  if (!handLandmarker || !poseLandmarker || !faceLandmarker) return;
  if (!syncCanvasToVideo()) return;
  if (timestamp - lastProcessTime < PROCESS_INTERVAL_MS) return;

  const elapsed = timestamp - lastProcessTime;
  lastProcessTime = timestamp;
  processedFrames += 1;
  if (timestamp - lastTelemetryTime >= 500) {
    fps = Math.round(1000 / elapsed);
    lastTelemetryTime = timestamp;
  }

  drawOverlay(detectFrame(timestamp), timestamp);
}

function startPredictionLoop() {
  if (trackingEnabled) return;
  logCv("Starting prediction loop: SUCCESS");
  resetTrackingBuffers();
  trackingEnabled = true;
  lastProcessTime = 0;
  lastTelemetryTime = 0;
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  animationFrameId = requestAnimationFrame(predictionLoop);
}

function startPredictionLoopIfReady() {
  if (!cameraReady || !mediaPipeReady) {
    logCv(`Starting prediction loop: WAITING cameraReady=${cameraReady} mediaPipeReady=${mediaPipeReady}`);
    return;
  }
  startPredictionLoop();
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    video.srcObject = stream;
    await video.play();
    cameraReady = true;
    cameraStatus.textContent = "OK Kamera siap";
    cameraStatus.className = "mt-3 rounded bg-teal-100 px-3 py-2 text-sm font-bold text-teal-800";
    syncCanvasToVideo();
    startPredictionLoopIfReady();
  } catch (error) {
    console.error(error);
    video.classList.add("hidden");
    fallback.classList.remove("hidden");
    fallback.classList.add("grid");
    mediaPipeStatus.textContent = "Kamera diblokir";
    cameraStatus.textContent = "! Kamera tidak tersedia";
    cameraStatus.className = "mt-3 rounded bg-amber-100 px-3 py-2 text-sm font-bold text-amber-800";
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function countdown() {
  countdownOverlay.classList.remove("hidden");
  countdownOverlay.classList.add("grid");
  for (const value of ["3", "2", "1"]) {
    countdownOverlay.textContent = value;
    await sleep(700);
  }
  countdownOverlay.textContent = "Go";
  await sleep(450);
  countdownOverlay.classList.add("hidden");
  countdownOverlay.classList.remove("grid");
}

async function startRecording() {
  if (!stream || recording) return;
  recordedChunks = [];
  resetTrackingBuffers();
  await countdown();
  mediaRecorder = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm" });
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) recordedChunks.push(event.data);
  };
  mediaRecorder.onstop = () => {
    recording = false;
    recordingStatus.textContent = `Urutan latihan tersimpan (${recordedChunks.length} chunks). Siap untuk analisis akhir.`;
    startRecordingButton.disabled = false;
    finishRecordingButton.disabled = true;
    if (analyzeButton) {
      analyzeButton.disabled = false;
      analyzeButton.classList.remove("opacity-60", "cursor-not-allowed");
    }
  };
  mediaRecorder.start(250);
  recording = true;
  recordingStatus.textContent = "Sedang merekam... lakukan gerakan, lalu tekan selesai.";
  startRecordingButton.disabled = true;
  finishRecordingButton.disabled = false;
}

function finishRecording() {
  if (mediaRecorder && recording) mediaRecorder.stop();
}

startRecordingButton.addEventListener("click", startRecording);
finishRecordingButton.addEventListener("click", finishRecording);
retryTrackingButton.addEventListener("click", async () => {
  resetTrackingBuffers();
  await initMediaPipe();
  startPredictionLoopIfReady();
});
mirrorReferenceToggle?.addEventListener("change", () => {
  referenceVideo?.classList.toggle("reference-mirrored", mirrorReferenceToggle.checked);
});
referenceVideo?.classList.toggle("reference-mirrored", Boolean(mirrorReferenceToggle?.checked));
video.addEventListener("loadedmetadata", syncCanvasToVideo);
window.addEventListener("resize", syncCanvasToVideo);

if (window.ISYARA_PRACTICE_OVERLAY_LOADED) {
  logCv("Duplicate practice overlay initialization skipped");
} else {
  window.ISYARA_PRACTICE_OVERLAY_LOADED = true;
  initMediaPipe();
  if (navigator.mediaDevices?.getUserMedia) {
    startCamera();
  } else {
    video.classList.add("hidden");
    fallback.classList.remove("hidden");
    fallback.classList.add("grid");
    mediaPipeStatus.textContent = "Kamera tidak didukung";
    cameraStatus.textContent = "! Kamera tidak didukung";
    cameraStatus.className = "mt-3 rounded bg-amber-100 px-3 py-2 text-sm font-bold text-amber-800";
  }
}
