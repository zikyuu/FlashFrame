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
import { cueFromResults, createStabilizer, createSpeaker } from "./coach.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URLS = {
  // "full" pose trades some speed for meaningfully better accuracy than "lite".
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  hands: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
};

// Auto-snap runs until it gets a genuinely good take, not for a fixed time:
//  - a frame within `stopOff` of the reference (10% off = 90% match) ends the session
//    (after a beat, so a few frames around the peak are caught too);
//  - frames within `keepOff` (25% off = 75% match) are kept, best `count` of them, spaced
//    apart so they're really different shots;
//  - if you never get that good, the best few frames seen are kept anyway as a fallback,
//    and `maxSeconds` stops the session so it can't run forever.
// Scores are "how far off", so match % = 100 - off %.
const AUTO = {
  stopOff: 0.1,
  keepOff: 0.25,
  count: 6,
  minShots: 3,
  spacingMs: 800,
  afterHitMs: 1500,
  maxSeconds: 300,
  // Skip frames mid-blink: they'd waste one of the few slots. (Eyes-half-closed on a frame
  // that is otherwise a good match is the classic ruined shot.)
  maxBlink: 0.45,
};
const MAX_LOOP_ERRORS = 30; // consecutive failed frames before we stop retrying

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
// The camera frame center-cropped to the reference photo's aspect ratio. Detection runs on
// this (not the raw video), so landmarks are measured in the same framing as the reference,
// and what you see, score and export all share it.
const frameCanvas = document.createElement("canvas");
const frameCtx = frameCanvas.getContext("2d");

const mirrorToggle = $("mirror-toggle");
const landmarksToggle = $("show-landmarks");
const debugToggle = $("show-debug");
const voiceToggle = $("voice-toggle");
const ignoreEyesToggle = $("ignore-eyes");
const flipCameraBtn = $("flip-camera");
const refPip = $("ref-pip");
const debugEl = $("debug");
const coachEl = $("coach");
const meters = {
  body: { marker: $("meter-marker"), caption: $("meter-caption") },
  face: { marker: $("face-meter-marker"), caption: $("face-meter-caption") },
  hands: { marker: $("hands-meter-marker"), caption: $("hands-meter-caption") },
};
const durationButtons = document.querySelectorAll(".duration-btn");
const startTimerBtn = $("start-timer");
const autoSnapBtn = $("auto-snap");
const countdownOverlay = $("countdown-overlay");
const sessionBadge = $("session-badge");
const preCaptureControls = $("pre-capture-controls");
const postCaptureControls = $("post-capture-controls");
const exportBtn = $("export-photo");
const retryBtn = $("retry");
const gallery = $("gallery");
const thumbsEl = $("thumbs");

const LEVEL_LABELS = { green: "Great match", yellow: "Close — adjust", red: "Way off" };

// All three models run in still-image mode, for the reference photo, captures AND the live
// feed. MediaPipe's video mode tracks landmarks across frames and settles on noticeably
// different estimates than a still pass (8-26° on hips/shoulders/neck for the same image in
// testing), so comparing a live video-mode pose to a still-mode reference built in a ~25%
// score offset before you'd moved at all. Same estimator on both sides removes it.
const models = { pose: null, face: null, hands: null };

// The reference photo, processed once. `trueSet` is as-photographed; `mirroredSet` is the
// same pose with left/right swapped, used when the display is mirrored so "looks right on
// my mirrored view" and "scores well" agree. Detection itself is never mirrored — only
// which reference we compare against.
let reference = { trueSet: null, mirroredSet: null };
let referenceImage = null;
let referenceDet = null;

// Latest live detections. Face and hands alternate frames (see liveStep), so each holds
// its most recent result until its next turn.
const live = { pose: null, face: null, hands: [] };
let frameCount = 0;

let cropAspect = null; // reference photo's width/height; null = uncropped camera
let selectedSeconds = 3;
let mirrorEnabled = false;
let showLandmarks = true;
let showDebug = false;
let ignoreEyes = false; // for sunglasses / glare: don't score eye expressions
let facing = "user"; // "user" = front camera, "environment" = back camera
let stream = null;
let loopStarted = false;
let loopErrors = 0;
let frameMs = 0; // smoothed time between live frames, shown in Debug so you can see what three models cost on your device
let lastFrameAt = 0;
// True once a capture has frozen the live panel on a snapshot — the render loop stops
// touching the canvas until Retry restarts it.
let frozen = false;
let lastSnapshot = null;
let lastCaptureDet = null;

const speaker = createSpeaker();
const stabilizeCue = createStabilizer(700);
let shownCue = "";

let auto = null; // running auto-snap session: { startedAt, good, backup, hitAt }
let leadIn = null; // lead-in countdown before a session: { cancelled }
let selectedShot = -1;
let shots = [];

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

  setStatus("Ready — choose a reference photo and allow camera access.");
}

function unpackFace(result) {
  const lm = result.faceLandmarks[0];
  const cats = result.faceBlendshapes[0]?.categories;
  return lm && cats ? { lm, cats } : null;
}

// Attaches the model's Left/Right label to each hand's landmark array; hands.js uses it to
// decide which slot a lone hand belongs in.
function unpackHands(result) {
  return result.landmarks.map((lm, i) => Object.assign(lm, { label: result.handedness?.[i]?.[0]?.categoryName }));
}

// The hand model is only worth running when the reference has hands to compare against.
// Otherwise the Hands meter isn't scored anyway, and in testing the model hallucinated a
// hand on a creased sweater sleeve — which would then be drawn into the exported photo.
function referenceHasHands() {
  return Boolean(reference.trueSet?.hands.slots.some(Boolean));
}

// One image through the models. The reference upload always includes hands (that's how we
// find out whether it has any); the live feed and captures only when the reference does.
function detect(source, { hands = true } = {}) {
  return {
    pose: models.pose.detect(source).landmarks[0] ?? null,
    face: unpackFace(models.face.detect(source)),
    hands: hands ? unpackHands(models.hands.detect(source)) : [],
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
  syncRefPip();
}

// On phones the reference is shown as a small thumbnail over the camera view (CSS hides this
// on wider screens, where it sits beside the camera instead).
function syncRefPip() {
  refPip.width = 360;
  refPip.height = Math.round((360 * refCanvas.height) / refCanvas.width);
  refPip.getContext("2d").drawImage(refCanvas, 0, 0, refPip.width, refPip.height);
  refPip.classList.remove("hidden");
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
  if (!file) return;
  if (!models.pose) {
    setStatus("Still loading the models — pick the photo again in a moment.");
    return;
  }

  setStatus("Analyzing your reference photo…");
  try {
    // Wait for `load`, not `decode()`: decode() can stall indefinitely when the tab isn't
    // visible, while load fires regardless (drawImage decodes on demand).
    const image = new Image();
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("that file isn't an image the browser can read"));
      image.src = URL.createObjectURL(file);
    });

    refCanvas.width = image.naturalWidth;
    refCanvas.height = image.naturalHeight;

    const det = detect(image);

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
  } catch (err) {
    console.error(err);
    setStatus(`Couldn't read that photo: ${err.message}`);
  }
}

// --- Scoring ------------------------------------------------------------------------

const pct = (score) => `${Math.round(score * 100)}%`;

function scored(r, suffix = "") {
  return { level: r.level, score: r.score, diffs: r.diffs, deltas: r.deltas, suffix };
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
  return scored(scoreFace(set.face, computeFaceMetrics(det.face.lm, det.face.cats, aspect), { ignoreEyes }));
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
  if (frameMs) lines.push(`speed  ${Math.round(1000 / frameMs)} fps (${Math.round(frameMs)} ms per frame)`);
  debugEl.textContent = lines.join("\n");
}

// --- Coach (text + voice) -----------------------------------------------------------

function setCoachText(text) {
  if (text === shownCue) return;
  shownCue = text;
  coachEl.textContent = text;
}

function updateCoach(results, now) {
  const text = stabilizeCue(cueFromResults(results).text, now);
  setCoachText(text);
  speaker.say(text, now);
}

// --- Camera + live loop -------------------------------------------------------------

async function startCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  // Ask for HD so the crop to a portrait reference still has decent resolution.
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  liveVideo.srcObject = stream;
  await new Promise((resolve) => (liveVideo.onloadedmetadata = resolve));
  liveVideo.play().catch(() => {}); // the autoplay attribute does the work; don't block on it

  // Start the loop before anything optional below, so a hiccup there can't leave the app
  // without a live view.
  if (!loopStarted) {
    loopStarted = true;
    requestAnimationFrame(liveLoop);
  }
  keepScreenOn();

  // Show the flip button where there's something to flip to: more than one camera, or a
  // touch device (phones can under-report cameras until permission is granted, and the button
  // reports honestly if the switch doesn't work). Cosmetic, so it must never block or fail the
  // camera start.
  const touchDevice = matchMedia("(pointer: coarse)").matches;
  try {
    const devices = await Promise.race([
      navigator.mediaDevices.enumerateDevices(),
      new Promise((_, reject) => setTimeout(() => reject(new Error("timed out")), 2000)),
    ]);
    const cameras = devices.filter((d) => d.kind === "videoinput").length;
    flipCameraBtn.classList.toggle("hidden", cameras < 2 && !touchDevice);
  } catch {
    flipCameraBtn.classList.toggle("hidden", !touchDevice);
  }
}

// The live loop runs on animation frames, which browsers stop when the screen sleeps or the
// tab is hidden — and so would the voice coaching. Asking for a wake lock keeps the screen on
// while the camera is running (important when the phone is propped up facing away). Browsers
// drop the lock when the tab is hidden, so take it again when we come back.
let wakeLock = null;
async function keepScreenOn() {
  try {
    wakeLock = (await navigator.wakeLock?.request("screen")) ?? null;
  } catch {
    // unsupported, or refused (e.g. low battery) — the app still works, the screen may just dim
  }
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && stream) keepScreenOn();
});

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

function copyCanvas(source) {
  const copy = document.createElement("canvas");
  copy.width = source.width;
  copy.height = source.height;
  copy.getContext("2d").drawImage(source, 0, 0);
  return copy;
}

async function liveLoop() {
  // Once frozen, the canvas holds the captured snapshot until Retry restarts this loop.
  if (frozen) return;

  try {
    await liveStep();
    const now = performance.now();
    if (lastFrameAt) frameMs = frameMs ? frameMs * 0.9 + (now - lastFrameAt) * 0.1 : now - lastFrameAt;
    lastFrameAt = now;
    if (loopErrors > 0) {
      loopErrors = 0;
      setStatus("Live view recovered.");
    }
  } catch (err) {
    loopErrors++;
    console.error("Live frame failed:", err);
    if (loopErrors >= MAX_LOOP_ERRORS) {
      setStatus(`The live view stopped after repeated errors (${err.message}) — reload the page.`);
      return;
    }
    if (loopErrors === 1) setStatus(`Live view hiccup (${err.message}) — retrying…`);
  }

  requestAnimationFrame(liveLoop);
}

async function liveStep() {
  if (!(liveVideo.readyState >= 2 && liveVideo.videoWidth)) return;

  const rect = cropRect(liveVideo.videoWidth, liveVideo.videoHeight, cropAspect);
  fitFrame(rect);
  frameCtx.drawImage(liveVideo, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, frameCanvas.width, frameCanvas.height);

  if (models.pose) {
    live.pose = models.pose.detect(frameCanvas).landmarks[0] ?? null;

    // Three models per frame is a lot of GPU work, so when hands are in play, face and
    // hands take turns; with no hands to score, face runs every frame.
    const useHands = referenceHasHands();
    if (!useHands || frameCount % 2 === 0) live.face = unpackFace(models.face.detect(frameCanvas));
    if (useHands && frameCount % 2 === 1) live.hands = unpackHands(models.hands.detect(frameCanvas));
    else if (!useHands) live.hands = [];
    frameCount++;

    const set = currentSet();
    const results = evaluate(set, live, frameCanvas.width / frameCanvas.height);
    renderResults(results, { smooth: true });

    if (!set) {
      setCoachText("Upload a reference photo and I'll coach you into it.");
    } else {
      const now = performance.now();
      updateCoach(results, now);
      if (auto) {
        feedAuto(results, now);
        updateSessionBadge(now);
        const peaked = auto.hitAt !== null && now - auto.hitAt >= AUTO.afterHitMs;
        const timedOut = now - auto.startedAt >= AUTO.maxSeconds * 1000;
        if (peaked || timedOut) {
          finishAuto(peaked ? "hit" : "timeout");
          return; // the frozen snapshot is on screen now — don't draw the live frame over it
        }
      }
    }
  }

  liveCtx.drawImage(frameCanvas, 0, 0);
  if (showLandmarks) drawAll(liveCtx, live);
}

// --- Freezing on a shot -------------------------------------------------------------

// Redraws the frozen frame from the cached snapshot — used on freeze and whenever the
// landmarks toggle changes while frozen, so toggling doesn't re-run detection.
function redrawFrozenFrame() {
  if (!lastSnapshot) return;
  liveCtx.drawImage(lastSnapshot, 0, 0);
  if (showLandmarks && lastCaptureDet) drawAll(liveCtx, lastCaptureDet);
}

// Stops the live view on one specific frame. Shown (and exported) the way you were posing
// for it: if you practiced in mirror mode, it's mirrored, so the picture matches what you saw.
function showFrozen(snapshot, det, results) {
  frozen = true;
  speaker.stop();
  liveCanvas.classList.toggle("mirrored", mirrorEnabled);
  liveCanvas.width = snapshot.width;
  liveCanvas.height = snapshot.height;
  lastSnapshot = snapshot;
  lastCaptureDet = det;
  redrawFrozenFrame();
  renderResults(results);

  preCaptureControls.classList.add("hidden");
  postCaptureControls.classList.remove("hidden");
}

// --- Timer capture ------------------------------------------------------------------

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setBusy(busy) {
  startTimerBtn.disabled = busy;
}

// Counts down on screen (and out loud, if the voice coach is on). Returns false if cancelled.
async function countdown(seconds, token) {
  countdownOverlay.classList.remove("hidden");
  for (let s = seconds; s > 0; s--) {
    if (token?.cancelled) {
      countdownOverlay.classList.add("hidden");
      return false;
    }
    countdownOverlay.textContent = s;
    speaker.announce(String(s));
    await wait(1000);
  }
  return !token?.cancelled;
}

async function runCountdown(seconds) {
  if (!reference.trueSet) {
    setStatus("Upload a reference photo before starting the timer.");
    return;
  }

  setBusy(true);
  autoSnapBtn.disabled = true;
  await countdown(seconds);
  countdownOverlay.textContent = "📸";

  try {
    await capturePose();
  } catch (err) {
    console.error(err);
    setStatus(`Capture failed: ${err.message}`);
  } finally {
    countdownOverlay.classList.add("hidden");
    setBusy(false);
    autoSnapBtn.disabled = false;
  }
}

async function capturePose() {
  const snapshot = copyCanvas(frameCanvas);

  // A fresh pass on the frozen frame, scored against the same reference the live meters
  // were using (mirrored if you were in mirror mode).
  const det = detect(snapshot, { hands: referenceHasHands() });

  showFrozen(snapshot, det, evaluate(currentSet(), det, snapshot.width / snapshot.height));
  setCoachText("Captured. Export it, or Retry.");
}

// --- Auto-snap ----------------------------------------------------------------------

// One number for "how good is this frame overall": the mean of whichever meters were scored.
function sessionScore(results) {
  const scoredResults = Object.values(results).filter((r) => r.level);
  if (scoredResults.length === 0) return null;
  return scoredResults.reduce((sum, r) => sum + r.score, 0) / scoredResults.length;
}

function blinkAmount(face) {
  if (!face) return 0;
  const get = (name) => face.cats.find((c) => c.categoryName === name)?.score ?? 0;
  return (get("eyeBlinkLeft") + get("eyeBlinkRight")) / 2;
}

const matchPct = (score) => `${Math.round((1 - score) * 100)}%`;

// Adds a frame to a list of the best `cap` frames, keeping them spaced apart in time: a frame
// near an existing one only replaces it if it's better; otherwise it competes with the worst.
function admit(list, cap, score, now, make) {
  const near = list.find((c) => Math.abs(now - c.at) < AUTO.spacingMs);
  if (near) {
    if (score < near.score) list[list.indexOf(near)] = make();
  } else if (list.length < cap) {
    list.push(make());
  } else {
    const worst = list.reduce((a, b) => (b.score > a.score ? b : a));
    if (score < worst.score) list[list.indexOf(worst)] = make();
  }
}

function feedAuto(results, now) {
  const score = sessionScore(results);
  if (score === null || blinkAmount(live.face) > AUTO.maxBlink) return;

  // Copy the frame at most once, and only if some list actually wants it.
  let candidate = null;
  const make = () =>
    (candidate ??= {
      canvas: copyCanvas(frameCanvas),
      det: { pose: live.pose, face: live.face, hands: live.hands },
      results,
      score,
      at: now,
    });

  if (score <= AUTO.keepOff) admit(auto.good, AUTO.count, score, now, make);
  admit(auto.backup, AUTO.minShots, score, now, make); // the fallback if nothing gets good enough

  if (score <= AUTO.stopOff && auto.hitAt === null) {
    auto.hitAt = now;
    speaker.announce("That's the one — hold still.");
  }
}

function updateSessionBadge(now) {
  const secs = Math.floor((now - auto.startedAt) / 1000);
  const clock = `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
  const all = [...auto.good, ...auto.backup];
  const best = all.length ? Math.min(...all.map((c) => c.score)) : null;
  sessionBadge.textContent =
    `● ${clock} · ${auto.good.length} saved` +
    (best === null ? "" : ` · best ${matchPct(best)} match`) +
    (auto.hitAt !== null ? " · got it!" : ` · aiming for ${matchPct(AUTO.stopOff)}`);
}

function setSessionUi(active) {
  autoSnapBtn.textContent = active ? "Stop" : "Auto-snap";
  startTimerBtn.disabled = active;
  sessionBadge.classList.toggle("hidden", !active);
}

async function startAutoSnap() {
  if (!reference.trueSet) {
    setStatus("Upload a reference photo before starting auto-snap.");
    return;
  }

  setSessionUi(true);
  leadIn = { cancelled: false };
  speaker.announce("Get ready.");
  const token = leadIn;
  const ok = await countdown(selectedSeconds, token);
  countdownOverlay.classList.add("hidden");
  leadIn = null;
  if (!ok) {
    setSessionUi(false);
    return;
  }

  speaker.announce(`Go. I'll keep going until you hit a ${matchPct(AUTO.stopOff)} match.`);
  auto = { startedAt: performance.now(), good: [], backup: [], hitAt: null };
  updateSessionBadge(performance.now());
}

// reason: "hit" (got a take at the target), "timeout", or "stopped" (you pressed Stop).
function finishAuto(reason = "stopped") {
  const session = auto;
  auto = null;
  setSessionUi(false);

  const byScore = (a, b) => a.score - b.score;
  const picked = [...session.good].sort(byScore);
  // Fewer than a handful that were good enough: top up with the best of what else was seen,
  // so you always get something to choose from.
  for (const extra of [...session.backup].sort(byScore)) {
    if (picked.length >= AUTO.minShots) break;
    if (!picked.includes(extra)) picked.push(Object.assign(extra, { fallback: true }));
  }
  shots = picked.sort(byScore);

  if (shots.length === 0) {
    setCoachText("Didn't catch a usable frame — check the camera can see you, then try again.");
    speaker.announce("I didn't catch a usable shot. Try again.");
    return;
  }

  renderGallery();
  selectShot(0);
  const best = matchPct(shots[0].score);
  const spoken =
    reason === "hit"
      ? `Got it. ${shots.length} shots, the best is a ${best} match.`
      : `${reason === "timeout" ? "Time's up." : "Stopped."} Here are the best ${shots.length}. The top one is a ${best} match.`;
  speaker.announce(spoken);
}

function renderGallery() {
  thumbsEl.replaceChildren(
    ...shots.map((shot, i) => {
      const thumb = document.createElement("button");
      thumb.type = "button";
      thumb.className = "thumb";
      thumb.addEventListener("click", () => selectShot(i));

      const canvas = document.createElement("canvas");
      canvas.width = 160;
      canvas.height = Math.round((160 * shot.canvas.height) / shot.canvas.width);
      canvas.getContext("2d").drawImage(shot.canvas, 0, 0, canvas.width, canvas.height);

      const label = document.createElement("span");
      label.className = `thumb-label level-${levelFor(shot.score)}`;
      label.textContent = `#${i + 1} · ${matchPct(shot.score)} match${shot.fallback ? " (best effort)" : ""}`;

      thumb.append(canvas, label);
      return thumb;
    })
  );
  gallery.classList.remove("hidden");
}

function selectShot(i) {
  selectedShot = i;
  [...thumbsEl.children].forEach((el, j) => el.classList.toggle("selected", j === i));
  const shot = shots[i];
  showFrozen(shot.canvas, shot.det, shot.results);
  setCoachText(`Shot ${i + 1} of ${shots.length} — pick your favourite, then Export.`);
}

function cancelAuto() {
  if (leadIn) leadIn.cancelled = true;
  if (auto) finishAuto();
}

// --- Controls -----------------------------------------------------------------------

refInput.addEventListener("change", handleReferenceUpload);

mirrorToggle.addEventListener("click", () => {
  mirrorEnabled = !mirrorEnabled;
  mirrorToggle.textContent = mirrorEnabled ? "Mirror: On" : "Mirror: Off";
  resetSmoothing();
  liveCanvas.classList.toggle("mirrored", mirrorEnabled);
  gallery.classList.toggle("mirrored", mirrorEnabled);
});

landmarksToggle.addEventListener("change", () => {
  showLandmarks = landmarksToggle.checked;
  redrawReference();
  if (frozen) redrawFrozenFrame();
});

ignoreEyesToggle.addEventListener("change", () => {
  ignoreEyes = ignoreEyesToggle.checked;
  resetSmoothing();
});

debugToggle.addEventListener("change", () => {
  showDebug = debugToggle.checked;
  debugEl.classList.toggle("hidden", !showDebug);
});

if (!speaker.supported) {
  voiceToggle.disabled = true;
  voiceToggle.parentElement.title = "This browser has no text-to-speech";
}
voiceToggle.addEventListener("change", () => {
  if (voiceToggle.checked) {
    if (!speaker.enable()) voiceToggle.checked = false;
  } else {
    speaker.disable();
  }
});

flipCameraBtn.addEventListener("click", async () => {
  const previous = facing;
  facing = facing === "user" ? "environment" : "user";
  try {
    await startCamera();

    // `facingMode` is only a preference — a device with one camera quietly gives the same one
    // back. Check what we actually got.
    const actual = stream.getVideoTracks()[0]?.getSettings?.().facingMode;
    if (actual && actual !== facing) {
      facing = previous;
      setStatus("This device only gave one camera, so there was nothing to switch to.");
      await startCamera();
      return;
    }

    // The back camera isn't a selfie view — a mirrored display would just be confusing.
    if (facing === "environment" && mirrorEnabled) mirrorToggle.click();
    resetSmoothing();
    setStatus(facing === "environment" ? "Using the back camera." : "Using the front camera.");
  } catch (err) {
    console.error(err);
    facing = previous;
    setStatus(`Couldn't switch camera: ${err.message}`);
    await startCamera().catch(() => {});
  }
});

// Tap the reference thumbnail (phones) to look at it properly, tap again to shrink it.
refPip.addEventListener("click", () => refPip.classList.toggle("expanded"));

durationButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    durationButtons.forEach((b) => b.classList.remove("selected"));
    btn.classList.add("selected");
    selectedSeconds = Number(btn.dataset.seconds);
  });
});

startTimerBtn.addEventListener("click", () => runCountdown(selectedSeconds));

autoSnapBtn.addEventListener("click", () => {
  if (auto || leadIn) cancelAuto();
  else startAutoSnap();
});

// Exports exactly what's on screen. A mirrored display is only a CSS flip of the canvas, so
// bake the flip into the saved pixels — otherwise the file comes out the other way round
// from the picture you were posing for.
exportBtn.addEventListener("click", () => {
  const out = document.createElement("canvas");
  out.width = liveCanvas.width;
  out.height = liveCanvas.height;
  const ctx = out.getContext("2d");
  if (mirrorEnabled) {
    ctx.translate(out.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(liveCanvas, 0, 0);

  const link = document.createElement("a");
  link.download = `flashframe-${Date.now()}.png`;
  link.href = out.toDataURL("image/png");
  link.click();
});

retryBtn.addEventListener("click", () => {
  frozen = false;
  resetSmoothing();
  lastSnapshot = null;
  lastCaptureDet = null;
  shots = [];
  selectedShot = -1;
  gallery.classList.add("hidden");
  liveCanvas.classList.toggle("mirrored", mirrorEnabled);
  preCaptureControls.classList.remove("hidden");
  postCaptureControls.classList.add("hidden");
  requestAnimationFrame(liveLoop);
});

initModels()
  .then(startCamera)
  .catch((err) => {
    console.error(err);
    setStatus(`Error: ${err.message}`);
  });
