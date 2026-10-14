import {
  PoseLandmarker,
  FaceLandmarker,
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { computeAngles, scorePose, mirrorLandmarks, LM } from "./angles.js";
import { computeFaceMetrics, scoreFace, mirrorFaceMetrics, faceFrame } from "./face.js";
import { computeHandMetrics, scoreHands, mirrorHandMetrics } from "./hands.js";
import { meterPosition, levelFor } from "./scoring.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URLS = {
  // "full" pose trades some speed for meaningfully better accuracy than "lite".
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  hands: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
};

const MIN_DRAW_VISIBILITY = 0.5;
const BODY_POINTS = [
  LM.NOSE,
  LM.LEFT_SHOULDER,
  LM.RIGHT_SHOULDER,
  LM.LEFT_ELBOW,
  LM.RIGHT_ELBOW,
  LM.LEFT_WRIST,
  LM.RIGHT_WRIST,
  LM.LEFT_HIP,
  LM.RIGHT_HIP,
  LM.LEFT_KNEE,
  LM.RIGHT_KNEE,
  LM.LEFT_ANKLE,
  LM.RIGHT_ANKLE,
];
const BODY_CONNECTIONS = [
  [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER],
  [LM.LEFT_SHOULDER, LM.LEFT_ELBOW],
  [LM.LEFT_ELBOW, LM.LEFT_WRIST],
  [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW],
  [LM.RIGHT_ELBOW, LM.RIGHT_WRIST],
  [LM.LEFT_SHOULDER, LM.LEFT_HIP],
  [LM.RIGHT_SHOULDER, LM.RIGHT_HIP],
  [LM.LEFT_HIP, LM.RIGHT_HIP],
  [LM.LEFT_HIP, LM.LEFT_KNEE],
  [LM.LEFT_KNEE, LM.LEFT_ANKLE],
  [LM.RIGHT_HIP, LM.RIGHT_KNEE],
  [LM.RIGHT_KNEE, LM.RIGHT_ANKLE],
].map(([start, end]) => ({ start, end }));

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const refInput = $("ref-input");
const refCanvas = $("ref-canvas");
const liveVideo = $("live-video");
const liveCanvas = $("live-canvas");
const refCtx = refCanvas.getContext("2d");
const liveCtx = liveCanvas.getContext("2d");
// The webcam frame center-cropped to the reference photo's aspect ratio. Detection runs
// on this (not the raw video), so landmarks are measured in the same framing as the
// reference, and what you see, score and export all share it.
const frameCanvas = document.createElement("canvas");
const frameCtx = frameCanvas.getContext("2d");

const mirrorToggle = $("mirror-toggle");
const landmarksToggle = $("show-landmarks");
const debugToggle = $("show-debug");
const debugEl = $("debug");
const meters = {
  body: { marker: $("meter-marker"), caption: $("meter-caption") },
  face: { marker: $("face-meter-marker"), caption: $("face-meter-caption") },
  hands: { marker: $("hands-meter-marker"), caption: $("hands-meter-caption") },
};
const durationButtons = document.querySelectorAll(".duration-btn");
const startTimerBtn = $("start-timer");
const countdownOverlay = $("countdown-overlay");
const preCaptureControls = $("pre-capture-controls");
const postCaptureControls = $("post-capture-controls");
const exportBtn = $("export-photo");
const retryBtn = $("retry");

const LEVEL_LABELS = { green: "Great match", yellow: "Close — adjust", red: "Way off" };

const models = { pose: null, face: null, hands: null };
const modes = { pose: "IMAGE", face: "IMAGE", hands: "IMAGE" };

// The reference photo, processed once. `trueSet` is as-photographed; `mirroredSet` is the
// same pose with left/right swapped, used when the display is mirrored so "looks right on
// my mirrored view" and "scores well" agree. Detection itself is never mirrored — only
// which reference we compare against.
let reference = { trueSet: null, mirroredSet: null };
let referenceImage = null;
let referenceDet = null;

// Latest live detections. Face and hands alternate frames (see liveLoop), so each holds
// its most recent result until its next turn.
const live = { pose: null, face: null, hands: [] };
let frameCount = 0;

let cropAspect = null; // reference photo's width/height; null = uncropped webcam
let selectedSeconds = 3;
let mirrorEnabled = false;
let showLandmarks = true;
let showDebug = false;
// True once a countdown capture has frozen the live panel on a snapshot — the render
// loop stops touching the canvas until Retry restarts it.
let frozen = false;
// True while a still-image detection (reference upload or capture) has the models switched
// to IMAGE mode — the live loop must not call detectForVideo() then, since a model can't
// serve two running modes at once.
let capturing = false;
let lastSnapshot = null;
let lastCaptureDet = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function currentSet() {
  return mirrorEnabled ? reference.mirroredSet : reference.trueSet;
}

async function initModels() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const base = (key) => ({ modelAssetPath: MODEL_URLS[key], delegate: "GPU" });

  [models.pose, models.face, models.hands] = await Promise.all([
    PoseLandmarker.createFromOptions(vision, { baseOptions: base("pose"), runningMode: "IMAGE", numPoses: 1 }),
    FaceLandmarker.createFromOptions(vision, {
      baseOptions: base("face"),
      runningMode: "IMAGE",
      numFaces: 1,
      outputFaceBlendshapes: true,
    }),
    HandLandmarker.createFromOptions(vision, { baseOptions: base("hands"), runningMode: "IMAGE", numHands: 2 }),
  ]);

  setStatus("Ready — choose a reference photo and allow webcam access.");
}

async function ensureMode(key, mode) {
  if (modes[key] === mode) return;
  await models[key].setOptions({ runningMode: mode });
  modes[key] = mode;
}

function unpackFace(result) {
  const lm = result.faceLandmarks[0];
  const cats = result.faceBlendshapes[0]?.categories;
  return lm && cats ? { lm, cats } : null;
}

// The hand model is only worth running when the reference has hands to compare against.
// Otherwise the Hands meter isn't scored anyway, and in testing the model hallucinated a
// hand on a creased sweater sleeve — which would then be drawn into the exported photo.
function referenceHasHands() {
  return Boolean(reference.trueSet?.hands.slots.some(Boolean));
}

// One still image through the models. The reference upload always includes hands (that's
// how we find out whether it has any); later captures only when the reference does.
async function detectStill(source, { hands = true } = {}) {
  const keys = hands ? ["pose", "face", "hands"] : ["pose", "face"];
  await Promise.all(keys.map((key) => ensureMode(key, "IMAGE")));
  return {
    pose: models.pose.detect(source).landmarks[0] ?? null,
    face: unpackFace(models.face.detect(source)),
    hands: hands ? models.hands.detect(source).landmarks : [],
  };
}

const HALO = "rgba(0,0,0,0.5)";

// Line widths are in canvas pixels, so a high-resolution photo shrunk to fit the panel
// would otherwise get sub-pixel lines. `k` scales everything so the overlay looks the same
// thickness on screen at any resolution, and each stroke gets a dark halo so it stays
// visible on white studio backdrops as well as dark rooms.
function drawAll(ctx, det) {
  const k = Math.max(1, ctx.canvas.width / 480);
  const du = new DrawingUtils(ctx);
  const lines = (landmarks, connections, width, color = "#ffffff") => {
    du.drawConnectors(landmarks, connections, { color: HALO, lineWidth: width + 2 * k });
    du.drawConnectors(landmarks, connections, { color, lineWidth: width });
  };
  const dots = (points, radius) =>
    du.drawLandmarks(points, { color: HALO, fillColor: "#ffffff", radius, lineWidth: k });

  ctx.save();
  if (det.face) lines(det.face.lm, FaceLandmarker.FACE_LANDMARKS_CONTOURS, k, "rgba(255,255,255,0.85)");

  for (const hand of det.hands) {
    lines(hand, HandLandmarker.HAND_CONNECTIONS, 1.5 * k);
    dots(hand, 2.5 * k);
  }

  if (det.pose) {
    const visible = (i) => (det.pose[i].visibility ?? 1) >= MIN_DRAW_VISIBILITY;
    lines(
      det.pose,
      BODY_CONNECTIONS.filter(({ start, end }) => visible(start) && visible(end)),
      1.5 * k
    );
    dots(BODY_POINTS.filter(visible).map((i) => det.pose[i]), 3 * k);
  }
  ctx.restore();
}

function redrawReference() {
  if (!referenceImage) return;
  refCtx.drawImage(referenceImage, 0, 0);
  if (showLandmarks && referenceDet) drawAll(refCtx, referenceDet);
}

function buildReference(det, aspect) {
  const frame = det.face ? faceFrame(det.face.lm, aspect) : null;
  const face = det.face ? computeFaceMetrics(det.face.lm, det.face.cats, aspect) : null;
  const hands = computeHandMetrics(det.hands, frame, aspect);

  return {
    trueSet: { angles: det.pose ? computeAngles(det.pose, aspect) : null, face, hands },
    mirroredSet: {
      angles: det.pose ? computeAngles(mirrorLandmarks(det.pose), aspect) : null,
      face: face && mirrorFaceMetrics(face),
      hands: mirrorHandMetrics(hands),
    },
  };
}

async function handleReferenceUpload(event) {
  const file = event.target.files[0];
  if (!file || !models.pose) return;

  const image = new Image();
  image.src = URL.createObjectURL(file);
  await image.decode();

  refCanvas.width = image.naturalWidth;
  refCanvas.height = image.naturalHeight;

  capturing = true;
  const det = await detectStill(image);
  capturing = false;

  referenceImage = image;
  referenceDet = det;
  redrawReference();

  const found = det.pose || det.face || det.hands.length;
  const aspect = image.naturalWidth / image.naturalHeight;
  reference = found ? buildReference(det, aspect) : { trueSet: null, mirroredSet: null };

  // Crop and display the live view to the reference's shape — but only for a usable
  // reference, so a failed upload doesn't leave the camera cropped to nothing in particular.
  resetSmoothing();
  cropAspect = found ? aspect : null;
  if (found) document.documentElement.style.setProperty("--stage-aspect", aspect.toFixed(4));
  else document.documentElement.style.removeProperty("--stage-aspect");

  setStatus(
    found
      ? `Reference captured — body: ${det.pose ? "yes" : "no"}, face: ${det.face ? "yes" : "no"}, hands: ${det.hands.length}`
      : "Nothing found in that photo — try a clearer shot."
  );
}

// --- Scoring ------------------------------------------------------------------------

const pct = (score) => `${Math.round(score * 100)}%`;

function scored(r, suffix = "") {
  return { level: r.level, score: r.score, diffs: r.diffs, suffix };
}

function evalBody(set, det, aspect) {
  if (!set.angles) return { text: "No body found in the reference" };
  if (!det.pose) return { text: "No body detected" };

  const r = scorePose(set.angles, computeAngles(det.pose, aspect));
  if (r.level === "unknown") {
    return { text: `Only ${r.joints.length}/${r.totalJoints} joints visible — step back for a reliable score` };
  }
  return scored(r, ` (${r.joints.length}/${r.totalJoints} joints)`);
}

function evalFace(set, det, aspect) {
  if (!set.face) return { text: "No face found in the reference" };
  if (!det.face) return { text: "No face detected" };
  return scored(scoreFace(set.face, computeFaceMetrics(det.face.lm, det.face.cats, aspect)));
}

function evalHands(set, det, aspect) {
  const frame = det.face ? faceFrame(det.face.lm, aspect) : null;
  const r = scoreHands(set.hands, computeHandMetrics(det.hands, frame, aspect));
  if (r.empty) return { text: "No hands detected in the reference — not scored" };
  return scored(r);
}

function evaluate(set, det, aspect) {
  if (!set) {
    const idle = { text: "Upload a reference first" };
    return { body: idle, face: idle, hands: idle };
  }
  return { body: evalBody(set, det, aspect), face: evalFace(set, det, aspect), hands: evalHands(set, det, aspect) };
}

// Live scores are noisy frame to frame (landmark jitter, blinks, a wobbling lean), so the
// live meters show a short time-weighted average instead of the raw score. Time-based
// rather than per-frame, so it behaves the same on a fast or slow machine.
const SMOOTH_TAU_MS = 400;
const smoothing = { body: null, face: null, hands: null };

function smoothScore(key, score, now) {
  const prev = smoothing[key];
  if (!prev) {
    smoothing[key] = { value: score, at: now };
    return score;
  }
  const alpha = 1 - Math.exp(-(now - prev.at) / SMOOTH_TAU_MS);
  const value = prev.value + alpha * (score - prev.value);
  smoothing[key] = { value, at: now };
  return value;
}

function resetSmoothing() {
  for (const key of Object.keys(smoothing)) smoothing[key] = null;
}

function renderMeter(key, result, smooth) {
  const meter = meters[key];
  if (!result.level) {
    smoothing[key] = null; // not scored this frame — don't blend across the gap
    meter.marker.classList.add("hidden");
    meter.caption.className = "meter-caption";
    meter.caption.textContent = result.text;
    return;
  }

  const score = smooth ? smoothScore(key, result.score, performance.now()) : result.score;
  const level = levelFor(score);
  meter.marker.classList.remove("hidden");
  meter.marker.style.left = `${(meterPosition(score) * 100).toFixed(1)}%`;
  meter.caption.className = `meter-caption level-${level}`;
  meter.caption.textContent = `${LEVEL_LABELS[level]} — ${pct(score)} off${result.suffix ?? ""}`;
}

// `smooth` is for the live view only; a captured frame shows its exact, unsmoothed score.
function renderResults(results, { smooth = false } = {}) {
  for (const key of Object.keys(meters)) renderMeter(key, results[key], smooth);
  if (!showDebug) return;

  const lines = [];
  for (const [key, result] of Object.entries(results)) {
    const worst = Object.entries(result.diffs ?? {})
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([name, diff]) => `${name} ${pct(diff)}`);
    lines.push(`${key.padEnd(6)} ${worst.length ? worst.join("  ") : "-"}`);
  }
  debugEl.textContent = lines.join("\n");
}

// --- Live loop ----------------------------------------------------------------------

async function startWebcam() {
  // Ask for HD so the crop to a portrait reference still has decent resolution.
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  liveVideo.srcObject = stream;
  await new Promise((resolve) => (liveVideo.onloadedmetadata = resolve));
  liveVideo.play();

  requestAnimationFrame(liveLoop);
}

// Largest centered rectangle of the video frame that has the given aspect ratio.
function cropRect(vw, vh, aspect) {
  if (!aspect) return { sx: 0, sy: 0, sw: vw, sh: vh };
  if (vw / vh > aspect) {
    const sw = vh * aspect;
    return { sx: (vw - sw) / 2, sy: 0, sw, sh: vh };
  }
  const sh = vw / aspect;
  return { sx: 0, sy: (vh - sh) / 2, sw: vw, sh };
}

// Resizing a canvas clears it, so only touch the size when the crop actually changed.
function fitFrame({ sw, sh }) {
  const w = Math.round(sw);
  const h = Math.round(sh);
  if (frameCanvas.width === w && frameCanvas.height === h) return;
  frameCanvas.width = liveCanvas.width = w;
  frameCanvas.height = liveCanvas.height = h;
}

async function liveLoop() {
  // Once frozen, the canvas holds the captured snapshot until Retry restarts this loop.
  if (frozen) return;

  if (liveVideo.readyState >= 2 && liveVideo.videoWidth) {
    const rect = cropRect(liveVideo.videoWidth, liveVideo.videoHeight, cropAspect);
    fitFrame(rect);
    frameCtx.drawImage(liveVideo, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, frameCanvas.width, frameCanvas.height);

    if (models.pose && !capturing) {
      const ts = performance.now();

      await ensureMode("pose", "VIDEO");
      live.pose = models.pose.detectForVideo(frameCanvas, ts).landmarks[0] ?? null;

      // Three models per frame is a lot of GPU work, so when hands are in play, face and
      // hands take turns; with no hands to score, face runs every frame.
      const useHands = referenceHasHands();
      if (!useHands || frameCount % 2 === 0) {
        await ensureMode("face", "VIDEO");
        live.face = unpackFace(models.face.detectForVideo(frameCanvas, ts));
      }
      if (useHands && frameCount % 2 === 1) {
        await ensureMode("hands", "VIDEO");
        live.hands = models.hands.detectForVideo(frameCanvas, ts).landmarks;
      } else if (!useHands) {
        live.hands = [];
      }
      frameCount++;

      renderResults(evaluate(currentSet(), live, frameCanvas.width / frameCanvas.height), { smooth: true });
    }

    liveCtx.drawImage(frameCanvas, 0, 0);
    if (showLandmarks) drawAll(liveCtx, live);
  }

  requestAnimationFrame(liveLoop);
}

// --- Timer + capture ----------------------------------------------------------------

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runCountdown(seconds) {
  if (!reference.trueSet) {
    setStatus("Upload a reference photo before starting the timer.");
    return;
  }

  startTimerBtn.disabled = true;
  countdownOverlay.classList.remove("hidden");

  for (let s = seconds; s > 0; s--) {
    countdownOverlay.textContent = s;
    await wait(1000);
  }
  countdownOverlay.textContent = "📸";

  await capturePose();

  countdownOverlay.classList.add("hidden");
  startTimerBtn.disabled = false;
}

// Redraws the frozen frame from the cached snapshot — used after capture and whenever
// the landmarks toggle changes while frozen, so toggling doesn't re-run detection.
function redrawFrozenFrame() {
  if (!lastSnapshot) return;
  liveCtx.drawImage(lastSnapshot, 0, 0);
  if (showLandmarks && lastCaptureDet) drawAll(liveCtx, lastCaptureDet);
}

async function capturePose() {
  capturing = true;

  const snapshot = document.createElement("canvas");
  snapshot.width = frameCanvas.width;
  snapshot.height = frameCanvas.height;
  snapshot.getContext("2d").drawImage(frameCanvas, 0, 0);

  // A dedicated still-image pass on the frozen frame, same path as the reference photo —
  // not whatever the live loop last happened to see.
  const det = await detectStill(snapshot, { hands: referenceHasHands() });
  capturing = false;

  // Freeze on this exact frame. Always the true (unmirrored) view, scored against the
  // true reference — this is what gets exported, so it represents the real photo
  // regardless of which mirror mode you were practicing in.
  frozen = true;
  liveCanvas.classList.remove("mirrored");
  liveCanvas.width = snapshot.width;
  liveCanvas.height = snapshot.height;
  lastSnapshot = snapshot;
  lastCaptureDet = det;
  redrawFrozenFrame();

  renderResults(evaluate(reference.trueSet, det, snapshot.width / snapshot.height));

  preCaptureControls.classList.add("hidden");
  postCaptureControls.classList.remove("hidden");
}

// --- Controls -----------------------------------------------------------------------

refInput.addEventListener("change", handleReferenceUpload);

mirrorToggle.addEventListener("click", () => {
  mirrorEnabled = !mirrorEnabled;
  mirrorToggle.textContent = mirrorEnabled ? "Mirror: On" : "Mirror: Off";
  resetSmoothing();
  if (!frozen) liveCanvas.classList.toggle("mirrored", mirrorEnabled);
});

landmarksToggle.addEventListener("change", () => {
  showLandmarks = landmarksToggle.checked;
  redrawReference();
  if (frozen) redrawFrozenFrame();
});

debugToggle.addEventListener("change", () => {
  showDebug = debugToggle.checked;
  debugEl.classList.toggle("hidden", !showDebug);
});

durationButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    durationButtons.forEach((b) => b.classList.remove("selected"));
    btn.classList.add("selected");
    selectedSeconds = Number(btn.dataset.seconds);
  });
});

startTimerBtn.addEventListener("click", () => runCountdown(selectedSeconds));

exportBtn.addEventListener("click", () => {
  const link = document.createElement("a");
  link.download = `flashframe-${Date.now()}.png`;
  link.href = liveCanvas.toDataURL("image/png");
  link.click();
});

retryBtn.addEventListener("click", () => {
  frozen = false;
  resetSmoothing();
  lastSnapshot = null;
  lastCaptureDet = null;
  liveCanvas.classList.toggle("mirrored", mirrorEnabled);
  preCaptureControls.classList.remove("hidden");
  postCaptureControls.classList.add("hidden");
  requestAnimationFrame(liveLoop);
});

initModels()
  .then(startWebcam)
  .catch((err) => {
    console.error(err);
    setStatus(`Error: ${err.message}`);
  });
