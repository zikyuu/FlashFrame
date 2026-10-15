import { norm, summarize } from "./scoring.js";

// MediaPipe Face Landmarker mesh indices: nose tip and the outer corner of each eye.
const NOSE_TIP = 1;
const RIGHT_EYE_OUTER = 33;
const LEFT_EYE_OUTER = 263;

// Blendshape groups (each the average of its left/right or related pair): the expressions
// that clearly change a photo. Deliberately NOT included:
//  - eyeBlink — a blink lasts a fraction of a second and isn't part of a pose;
//  - frown, brow-down, cheek puff — subtle, and their resting values differ from face to face.
// Lip pucker / press ARE included (that's where "relax your lips" comes from) but are just as
// baseline-dependent, so they get a wider deadzone below and only fire on clearly tense lips.
const BLEND_GROUPS = {
  mouth_smile: ["mouthSmileLeft", "mouthSmileRight"],
  mouth_pucker: ["mouthPucker", "mouthFunnel"],
  mouth_press: ["mouthPressLeft", "mouthPressRight"],
  jaw_open: ["jawOpen"],
  brow_raise: ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"],
  eye_squint: ["eyeSquintLeft", "eyeSquintRight"],
  eye_wide: ["eyeWideLeft", "eyeWideRight"],
};
// The eye metrics are read from the eyes themselves, so they're unreliable when the eyes
// are hidden (sunglasses) or distorted (glasses glare) — the app can switch them off.
const EYE_METRICS = new Set(["eye_squint", "eye_wide"]);

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
// Blendshape scores wander a little even on an unmoving face, and resting values differ per
// person, so the first bit of difference doesn't count.
const BLEND_DEADZONE = 0.08;
const BLEND_DEADZONE_OVERRIDES = { mouth_pucker: 0.15, mouth_press: 0.15 };

export function scoreFace(reference, live, { ignoreEyes = false } = {}) {
  const diffs = {};
  const deltas = {}; // live minus reference, signed, for the coach
  for (const [name, scale] of Object.entries(GEOMETRY_SCALES)) {
    deltas[name] = live[name] - reference[name];
    diffs[name] = norm(deltas[name], scale);
  }
  for (const name of Object.keys(BLEND_GROUPS)) {
    if (ignoreEyes && EYE_METRICS.has(name)) continue;
    if (reference[name] < BLEND_ACTIVE_MIN && live[name] < BLEND_ACTIVE_MIN) continue;
    deltas[name] = live[name] - reference[name];
    diffs[name] = norm(Math.max(0, Math.abs(deltas[name]) - (BLEND_DEADZONE_OVERRIDES[name] ?? BLEND_DEADZONE)), BLEND_SCALE);
  }
  return { ...summarize(diffs), diffs, deltas };
}
