import { norm, summarize } from "./scoring.js";

// MediaPipe Face Landmarker mesh indices: nose tip and the outer corner of each eye.
const NOSE_TIP = 1;
const RIGHT_EYE_OUTER = 33;
const LEFT_EYE_OUTER = 263;

// Blendshape groups (each the average of its left/right or related pair). A subset of
// the model's 52 — the ones that visibly change a photo's expression.
const BLEND_GROUPS = {
  mouth_smile: ["mouthSmileLeft", "mouthSmileRight"],
  mouth_frown: ["mouthFrownLeft", "mouthFrownRight"],
  mouth_pucker: ["mouthPucker", "mouthFunnel"],
  mouth_press: ["mouthPressLeft", "mouthPressRight"],
  jaw_open: ["jawOpen"],
  brow_raise: ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"],
  brow_down: ["browDownLeft", "browDownRight"],
  eye_squint: ["eyeSquintLeft", "eyeSquintRight"],
  eye_wide: ["eyeWideLeft", "eyeWideRight"],
  cheek_puff: ["cheekPuff"],
  // No eye_blink: a blink lasts a fraction of a second and isn't part of a pose, but it
  // showed up at 24-49% off in real use and would flick the meter (or the capture) red.
};

// `aspect` is image width / height — x is scaled by it so distances and angles are
// measured in the same units on both axes.
function eyeGeometry(landmarks, aspect) {
  const a = { x: landmarks[RIGHT_EYE_OUTER].x * aspect, y: landmarks[RIGHT_EYE_OUTER].y };
  const b = { x: landmarks[LEFT_EYE_OUTER].x * aspect, y: landmarks[LEFT_EYE_OUTER].y };
  const nose = { x: landmarks[NOSE_TIP].x * aspect, y: landmarks[NOSE_TIP].y };
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  return { a, b, nose, mid, eyeDist: Math.hypot(b.x - a.x, b.y - a.y) };
}

// Reference point + scale for placing hands relative to the face ("hand on chin",
// "hand by the cheek") independent of how far from the camera someone is.
export function faceFrame(landmarks, aspect) {
  const { nose, eyeDist } = eyeGeometry(landmarks, aspect);
  return { cx: nose.x, cy: nose.y, width: eyeDist, aspect };
}

function blendAverage(categories, names) {
  const scores = names.map((n) => categories.find((c) => c.categoryName === n)?.score ?? 0);
  return scores.reduce((a, b) => a + b, 0) / scores.length;
}

export function computeFaceMetrics(landmarks, categories, aspect = 1) {
  const { a, b, nose, mid, eyeDist } = eyeGeometry(landmarks, aspect);

  const metrics = {
    // Roll: eye line off horizontal. Yaw/pitch: where the nose tip sits relative to the
    // eye midpoint, in eye-distance units — simple proxies, not a full head-pose solve.
    head_tilt: (Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI,
    head_yaw: (nose.x - mid.x) / eyeDist,
    head_pitch: (nose.y - mid.y) / eyeDist,
  };
  for (const [name, blendNames] of Object.entries(BLEND_GROUPS)) {
    metrics[name] = blendAverage(categories, blendNames);
  }
  return metrics;
}

// Mirroring the display flips roll and yaw; blendshape groups are already left/right
// averages, and pitch is unaffected.
export function mirrorFaceMetrics(metrics) {
  return { ...metrics, head_tilt: -metrics.head_tilt, head_yaw: -metrics.head_yaw };
}

// Yaw and pitch are crude nose-vs-eyes proxies that shift with the webcam's height and
// angle (which won't match the reference photographer's), so they get a wide scale.
const GEOMETRY_SCALES = { head_tilt: 30, head_yaw: 0.5, head_pitch: 0.5 };
const BLEND_SCALE = 0.5; // blendshape diff (0-1 scores) that counts as fully off
const BLEND_ACTIVE_MIN = 0.1; // ignore expressions neither frame is actually making

export function scoreFace(reference, live) {
  const diffs = {};
  for (const [name, scale] of Object.entries(GEOMETRY_SCALES)) {
    diffs[name] = norm(reference[name] - live[name], scale);
  }
  for (const name of Object.keys(BLEND_GROUPS)) {
    if (reference[name] < BLEND_ACTIVE_MIN && live[name] < BLEND_ACTIVE_MIN) continue;
    diffs[name] = norm(reference[name] - live[name], BLEND_SCALE);
  }
  return { ...summarize(diffs), diffs };
}
