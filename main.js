import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const statusEl = document.getElementById("status");
const refInput = document.getElementById("ref-input");
const refCanvas = document.getElementById("ref-canvas");
const liveVideo = document.getElementById("live-video");
const liveCanvas = document.getElementById("live-canvas");

const refCtx = refCanvas.getContext("2d");
const liveCtx = liveCanvas.getContext("2d");

// Exposed for the next build step (angle calculation) to read from.
window.referenceLandmarks = null;
window.liveLandmarks = null;

let poseLandmarker;
let runningMode = "IMAGE";

function setStatus(text) {
  statusEl.textContent = text;
}

async function initLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode,
    numPoses: 1,
  });
  setStatus("Ready — choose a reference photo and allow webcam access.");
}

function drawLandmarksOn(ctx, canvas, result) {
  const drawingUtils = new DrawingUtils(ctx);
  ctx.save();
  for (const landmarks of result.landmarks) {
    drawingUtils.drawLandmarks(landmarks, {
      radius: (data) => Math.max(1, 4 - data.from.z),
    });
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS);
  }
  ctx.restore();
}

async function handleReferenceUpload(event) {
  const file = event.target.files[0];
  if (!file || !poseLandmarker) return;

  const image = new Image();
  image.src = URL.createObjectURL(file);
  await image.decode();

  refCanvas.width = image.naturalWidth;
  refCanvas.height = image.naturalHeight;
  refCtx.drawImage(image, 0, 0);

  // detect() requires IMAGE mode; detectForVideo() (the webcam loop) requires VIDEO mode.
  // Both share one model instance, so switch modes rather than loading it twice.
  if (runningMode !== "IMAGE") {
    await poseLandmarker.setOptions({ runningMode: "IMAGE" });
    runningMode = "IMAGE";
  }

  const result = poseLandmarker.detect(image);
  window.referenceLandmarks = result.landmarks[0] ?? null;

  if (result.landmarks.length) {
    drawLandmarksOn(refCtx, refCanvas, result);
    setStatus("Reference pose captured.");
  } else {
    setStatus("No pose found in that photo — try one with the full body visible.");
  }
}

async function startWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
  });
  liveVideo.srcObject = stream;
  await new Promise((resolve) => (liveVideo.onloadedmetadata = resolve));
  liveVideo.play();

  liveCanvas.width = liveVideo.videoWidth;
  liveCanvas.height = liveVideo.videoHeight;

  requestAnimationFrame(liveLoop);
}

async function liveLoop() {
  if (poseLandmarker && liveVideo.readyState >= 2) {
    if (runningMode !== "VIDEO") {
      await poseLandmarker.setOptions({ runningMode: "VIDEO" });
      runningMode = "VIDEO";
    }

    const result = poseLandmarker.detectForVideo(liveVideo, performance.now());
    window.liveLandmarks = result.landmarks[0] ?? null;

    liveCtx.drawImage(liveVideo, 0, 0, liveCanvas.width, liveCanvas.height);
    if (result.landmarks.length) drawLandmarksOn(liveCtx, liveCanvas, result);
  }

  requestAnimationFrame(liveLoop);
}

refInput.addEventListener("change", handleReferenceUpload);

initLandmarker()
  .then(startWebcam)
  .catch((err) => {
    console.error(err);
    setStatus(`Error: ${err.message}`);
  });
