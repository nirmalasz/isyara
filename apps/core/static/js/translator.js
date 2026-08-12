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
const speakSpeechText = document.getElementById("speakSpeechText");
const debugRawPredictions = document.getElementById("debugRawPredictions");
const debugStablePrediction = document.getElementById("debugStablePrediction");
const debugSuppression = document.getElementById("debugSuppression");
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
};
const YOLO_CONFIDENCE_THRESHOLD = 0.75;
const YOLO_REQUEST_INTERVAL_MS = 180;
const DISPLAY_FRAME_MIRRORED = true;
const AI_INPUT_MIRRORED = false;
const ROI_PADDING = 0.45;
const SUPPORTED_SIGNS = ["Anda", "Apa", "Berhenti", "Bodoh", "Cantik", "Halo", "Hati-hati", "Lelah", "Maaf", "Makan", "Mau", "Membaca", "Nama", "Sama-sama", "Saya", "Siapa", "Sombong", "Takut", "Terima kasih"];

let handLandmarker;
let stream;
let lastPredictionRequestTime = 0;
let predictionInFlight = false;
let acceptedTokens = [];
let frameCounter = 0;
let lastPredictionResult = null;
let evalSamples = [];
let evalCaptureRemaining = 0;
let evalCaptureTarget = null;
let speechRecorder;
let speechChunks = [];
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
  if (!debugRawPredictions || !debugStablePrediction || !debugSuppression) return;
  const rawPredictions = result.raw_predictions || [];
  const candidatePredictions = result.candidate_predictions || [];
  debugRawPredictions.textContent = candidatePredictions.length
    ? candidatePredictions.map(formatCandidateDebug).join("\n\n")
    : rawPredictions.length
    ? rawPredictions.map((item) => `${item.label || item.raw_label}: ${Math.round(Number(item.confidence || 0) * 100)}%`).join("\n")
    : "-";
  const selected = result.selected_candidate;
  debugStablePrediction.textContent = stableLabel || result.stable_prediction || "-";
  debugSuppression.textContent = [
    `hands=${result.hands_detected ?? 0}`,
    `handed=${(result.handedness || []).join(",") || "-"}`,
    `model=${result.inference_model || "-"}`,
    `selected=${selected ? `${selected.roi_type}/${selected.label || "-"}/${Math.round(Number(selected.confidence || 0) * 100)}%` : "-"}`,
    `roi_type=${result.roi_type || "-"}`,
    `roi=${result.roi ? `${Math.round(result.roi.x1)},${Math.round(result.roi.y1)},${Math.round(result.roi.x2)},${Math.round(result.roi.y2)}` : "-"}`,
    `stable=${Boolean(stableLabel || result.stable)}`,
    `suppressed=${Boolean(suppressed || result.suppressed)}`,
    `lock=${result.locked_label || "-"}`,
    `release=${result.release_misses || 0}`,
    `frame=${result.frame_id || "-"}`,
    `image=${result.image_width || "-"}x${result.image_height || "-"}`,
    `mirrored=${result.mirrored ?? AI_INPUT_MIRRORED}`,
  ].join(" ");
}

function formatCandidateDebug(candidate) {
  const raw = candidate.raw_predictions || [];
  const top = raw.length ? raw[0] : candidate;
  const label = top.label || top.raw_label || candidate.label || "-";
  const confidence = Math.round(Number(top.confidence || candidate.confidence || 0) * 100);
  return `candidate_${candidate.roi_type || "unknown"}:\n${label} ${confidence}%`;
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
  return handLandmarker.detectForVideo(camera, timestamp);
}

function handCandidates(results) {
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
  const candidates = handBounds.map((bounds) =>
    roiFromBounds(bounds, candidateType(bounds, handBounds), sourceWidth, sourceHeight, hands.length, handedness)
  );
  if (handBounds.length > 1) {
    candidates.push(
      roiFromBounds(unionBounds(handBounds), "combined", sourceWidth, sourceHeight, hands.length, handedness)
    );
  }
  return candidates.filter((roi) => roi && roi.x2 > roi.x1 && roi.y2 > roi.y1);
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

function roiFromBounds(bounds, type, sourceWidth, sourceHeight, handsDetected, handedness) {
  const width = bounds.x2 - bounds.x1;
  const height = bounds.y2 - bounds.y1;
  const padX = Math.max(width * ROI_PADDING, 0.08);
  const padY = Math.max(height * ROI_PADDING, 0.08);
  return {
    type,
    x1: Math.max(0, Math.round((bounds.x1 - padX) * sourceWidth)),
    y1: Math.max(0, Math.round((bounds.y1 - padY) * sourceHeight)),
    x2: Math.min(sourceWidth, Math.round((bounds.x2 + padX) * sourceWidth)),
    y2: Math.min(sourceHeight, Math.round((bounds.y2 + padY) * sourceHeight)),
    sourceWidth,
    sourceHeight,
    handsDetected,
    handedness,
  };
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
    return new Promise((resolve) => yoloCanvas.toBlob(resolve, "image/jpeg", 0.88));
  }
  return new Promise((resolve) => roiCanvas.toBlob(resolve, "image/jpeg", 0.88));
}

async function predictCameraFrame(timestamp) {
  if (predictionInFlight || timestamp - lastPredictionRequestTime < YOLO_REQUEST_INTERVAL_MS) return;
  lastPredictionRequestTime = timestamp;
  const handResults = localizeHands(timestamp);
  const rois = handCandidates(handResults);
  if (!rois.length) {
    frameCounter += 1;
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
  frameCounter += 1;
  candidateFrames.forEach((candidate) => {
    formData.append("candidates", candidate.blob, `${candidate.roi.type}-candidate.jpg`);
  });
  formData.append("frame_id", String(frameCounter));
  formData.append("mirrored", String(AI_INPUT_MIRRORED));
  formData.append("source_width", String(candidateFrames[0].roi.sourceWidth));
  formData.append("source_height", String(candidateFrames[0].roi.sourceHeight));
  formData.append("hands_detected", String(candidateFrames[0].roi.handsDetected));
  formData.append("handedness", candidateFrames[0].roi.handedness.join(","));
  formData.append("candidates_json", JSON.stringify(candidateFrames.map((candidate) => ({
    type: candidate.roi.type,
    x1: candidate.roi.x1,
    y1: candidate.roi.y1,
    x2: candidate.roi.x2,
    y2: candidate.roi.y2,
    source_width: candidate.roi.sourceWidth,
    source_height: candidate.roi.sourceHeight,
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

function handlePrediction(result) {
  if (result.status === "model_unavailable" || result.status === "service_unavailable") {
    setPill(motionStatus, "warn", "! Model BISINDO belum siap");
    detectedText.textContent = result.display_text || "Model penerjemah sedang disiapkan.";
    detectedConfidence.textContent = "Confidence: -";
    clearOverlay();
    updateDebug(result);
    return;
  }
  if (result.status === "invalid_image" || result.status === "no_hand") {
    detectedText.textContent = result.display_text || "Frame kamera tidak valid.";
    detectedConfidence.textContent = "Confidence: -";
    clearOverlay();
    updateDebug(result);
    return;
  }
  setPill(aiStatus, "ok", "✓ Model BISINDO siap");
  setPill(motionStatus, "ok", "✓ Inference aktif");
  drawYoloBox(result);
  const confidence = Number(result.confidence || 0);
  const label = result.display_label || result.label || result.prediction;
  if (!result.detected || !label || confidence < YOLO_CONFIDENCE_THRESHOLD) {
    detectedText.textContent = result.rejection_reason ? "Belum ada tanda valid" : (result.display_text || "Gerakan belum dikenali");
    detectedConfidence.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
    updateDebug(result);
    return;
  }
  detectedText.textContent = `Terdeteksi: ${result.display_text || label}`;
  detectedConfidence.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
  if (result.accepted && result.accepted_prediction) {
    acceptPrediction({
      label: result.accepted_prediction,
      displayText: result.accepted_prediction,
      confidence: result.accepted_confidence ?? confidence,
    });
  }
  updateDebug(result, result.stable_prediction || null, result.suppressed);
}

function acceptPrediction(result) {
  acceptedTokens.push(result.displayText);
  renderTranscript();
  saveHistory("SIGN_TO_SPEECH", result.label, transcriptSentence(false), result.confidence);
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
stopSpeech.addEventListener("click", () => speechRecorder?.stop());
copySpeechText.addEventListener("click", () => navigator.clipboard?.writeText(speechTranscript.textContent || ""));
speakSpeechText?.addEventListener("click", () => speak(speechTranscript.textContent));
camera.addEventListener("loadedmetadata", syncCanvas);
window.addEventListener("resize", syncCanvas);

initEvaluationPanel();
startCamera();
initHandLocalizer();
