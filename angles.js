import { norm, summarize } from "./scoring.js";

// MediaPipe Pose landmark indices we use (of the 33 it returns).
export const LM = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
};

// Full left/right swap for all 33 landmarks — needed to correctly mirror a pose (not
// just flip x, but swap which side each point represents). Index i -> source index to
// pull from. Unpaired points (just NOSE, 0) map to themselves.
const MIRROR_INDEX = [
  0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23,
  26, 25, 28, 27, 30, 29, 32, 31,
];

export function mirrorLandmarks(landmarks) {
  return MIRROR_INDEX.map((srcIndex) => {
    const src = landmarks[srcIndex];
    return { ...src, x: 1 - src.x };
  });
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

// Angle at vertex b, between rays b->a and b->c. 0-180 degrees, direction-agnostic.
export function angleAt(a, b, c) {
  const v1 = Math.atan2(a.y - b.y, a.x - b.x);
  const v2 = Math.atan2(c.y - b.y, c.x - b.x);
  let deg = Math.abs(((v1 - v2) * 180) / Math.PI);
  if (deg > 180) deg = 360 - deg;
  return deg;
}

// How far p2 sits below p1, as an angle off horizontal (0 when level, positive when p2 is
// lower). Uses |dx| so it doesn't matter which point is on the left of the image — a plain
// atan2 would read ~±180° for a person facing the camera (their left point is on the image's
// right) and flip sign across level, turning a tiny wobble into a "100% off".
export function tiltFromHorizontal(p1, p2) {
  return (Math.atan2(p2.y - p1.y, Math.abs(p2.x - p1.x)) * 180) / Math.PI;
}

// Signed angle of the line base->top off vertical (0 when top is directly above base).
export function tiltFromVertical(base, top) {
  return (Math.atan2(top.x - base.x, base.y - top.y) * 180) / Math.PI;
}

function minVisibility(...points) {
  return Math.min(...points.map((p) => p.visibility ?? 1));
}

// `aspect` is image width / height. Landmarks are normalized per-axis, so on a non-square
// image a 45° line isn't 45° in normalized space — scaling x by the aspect ratio first
// keeps angles honest, which matters when comparing a portrait photo to a landscape webcam.
export function computeAngles(landmarks, aspect = 1) {
  const p = (i) => ({ ...landmarks[i], x: landmarks[i].x * aspect });
  const shoulderMid = midpoint(p(LM.LEFT_SHOULDER), p(LM.RIGHT_SHOULDER));
  const hipMid = midpoint(p(LM.LEFT_HIP), p(LM.RIGHT_HIP));

  const angles = {
    // Shoulder angles are measured off the shoulder-to-hip-midpoint (spine) axis rather
    // than the same-side hip, so torso/shoulder width doesn't bias the resting angle.
    left_shoulder: angleAt(p(LM.LEFT_ELBOW), p(LM.LEFT_SHOULDER), hipMid),
    right_shoulder: angleAt(p(LM.RIGHT_ELBOW), p(LM.RIGHT_SHOULDER), hipMid),
    left_elbow: angleAt(p(LM.LEFT_SHOULDER), p(LM.LEFT_ELBOW), p(LM.LEFT_WRIST)),
    right_elbow: angleAt(p(LM.RIGHT_SHOULDER), p(LM.RIGHT_ELBOW), p(LM.RIGHT_WRIST)),
    left_hip: angleAt(p(LM.LEFT_SHOULDER), p(LM.LEFT_HIP), p(LM.LEFT_KNEE)),
    right_hip: angleAt(p(LM.RIGHT_SHOULDER), p(LM.RIGHT_HIP), p(LM.RIGHT_KNEE)),
    left_knee: angleAt(p(LM.LEFT_HIP), p(LM.LEFT_KNEE), p(LM.LEFT_ANKLE)),
    right_knee: angleAt(p(LM.RIGHT_HIP), p(LM.RIGHT_KNEE), p(LM.RIGHT_ANKLE)),
    hip_tilt: tiltFromHorizontal(p(LM.LEFT_HIP), p(LM.RIGHT_HIP)),
    shoulder_tilt: tiltFromHorizontal(p(LM.LEFT_SHOULDER), p(LM.RIGHT_SHOULDER)),
    torso_lean: tiltFromVertical(hipMid, shoulderMid),
    neck_tilt: tiltFromVertical(shoulderMid, p(LM.NOSE)),
  };

  const visibility = {
    left_shoulder: minVisibility(p(LM.LEFT_ELBOW), p(LM.LEFT_SHOULDER), p(LM.LEFT_HIP), p(LM.RIGHT_HIP)),
    right_shoulder: minVisibility(p(LM.RIGHT_ELBOW), p(LM.RIGHT_SHOULDER), p(LM.LEFT_HIP), p(LM.RIGHT_HIP)),
    left_elbow: minVisibility(p(LM.LEFT_SHOULDER), p(LM.LEFT_ELBOW), p(LM.LEFT_WRIST)),
    right_elbow: minVisibility(p(LM.RIGHT_SHOULDER), p(LM.RIGHT_ELBOW), p(LM.RIGHT_WRIST)),
    left_hip: minVisibility(p(LM.LEFT_SHOULDER), p(LM.LEFT_HIP), p(LM.LEFT_KNEE)),
    right_hip: minVisibility(p(LM.RIGHT_SHOULDER), p(LM.RIGHT_HIP), p(LM.RIGHT_KNEE)),
    left_knee: minVisibility(p(LM.LEFT_HIP), p(LM.LEFT_KNEE), p(LM.LEFT_ANKLE)),
    right_knee: minVisibility(p(LM.RIGHT_HIP), p(LM.RIGHT_KNEE), p(LM.RIGHT_ANKLE)),
    hip_tilt: minVisibility(p(LM.LEFT_HIP), p(LM.RIGHT_HIP)),
    shoulder_tilt: minVisibility(p(LM.LEFT_SHOULDER), p(LM.RIGHT_SHOULDER)),
    torso_lean: minVisibility(p(LM.LEFT_SHOULDER), p(LM.RIGHT_SHOULDER), p(LM.LEFT_HIP), p(LM.RIGHT_HIP)),
    neck_tilt: minVisibility(p(LM.LEFT_SHOULDER), p(LM.RIGHT_SHOULDER), p(LM.NOSE)),
  };

  return { angles, visibility };
}

const MIN_VISIBILITY = 0.5;
// Below this many contributing joints there's no real averaging left — one noisy joint
// can swing the score between frames — so we refuse to show a confident color at all.
const MIN_JOINTS = 3;
// Angle diff that counts as fully off, for every joint and tilt alike. Tilts used to get a
// stricter 30° scale, but in a real test run a visibly close pose sat at 10-15° error on
// torso lean / hip / neck tilt and still scored red — people can't eyeball a lean closer
// than that.
const JOINT_SCALE_DEG = 60;

export function scorePose(reference, live) {
  const totalJoints = Object.keys(reference.angles).length;
  const joints = Object.keys(reference.angles).filter(
    (joint) => reference.visibility[joint] >= MIN_VISIBILITY && live.visibility[joint] >= MIN_VISIBILITY
  );

  if (joints.length < MIN_JOINTS) {
    return { level: "unknown", score: null, diffs: {}, deltas: {}, joints, totalJoints };
  }

  // `deltas` keep the sign (live minus reference, in degrees) so the coach can say which
  // way to move; `diffs` are the normalized sizes that drive the score.
  const diffs = {};
  const deltas = {};
  for (const joint of joints) {
    deltas[joint] = live.angles[joint] - reference.angles[joint];
    diffs[joint] = norm(deltas[joint], JOINT_SCALE_DEG);
  }

  return { ...summarize(diffs), diffs, deltas, joints, totalJoints };
}
