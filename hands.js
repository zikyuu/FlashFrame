import { angleAt, tiltFromVertical } from "./angles.js";
import { norm, summarize } from "./scoring.js";

// MediaPipe Hand Landmarker: 21 points per hand. Each finger's curl is the angle at its
// middle joint (~180° straight, smaller as it curls). Thumb uses MCP-IP-TIP.
const FINGERS = {
  thumb: [2, 3, 4],
  index: [5, 6, 7],
  middle: [9, 10, 11],
  ring: [13, 14, 15],
  pinky: [17, 18, 19],
};
const WRIST = 0;
const MIDDLE_MCP = 9;

// Describes one hand: finger curls, which way it points, and where its wrist sits
// relative to the face (null if no face was found to measure against).
function describeHand(landmarks, frame, aspect) {
  const p = (i) => ({ x: landmarks[i].x * aspect, y: landmarks[i].y });

  const curls = {};
  for (const [name, [a, b, c]] of Object.entries(FINGERS)) {
    curls[name] = angleAt(p(a), p(b), p(c));
  }

  const wrist = p(WRIST);
  return {
    curls,
    orient: tiltFromVertical(wrist, p(MIDDLE_MCP)),
    rel: frame ? { x: (wrist.x - frame.cx) / frame.width, y: (wrist.y - frame.cy) / frame.width } : null,
  };
}

// Hands go into two slots by where they sit in the image (left half / right half), not
// by the model's Left/Right label — that label assumes a pre-mirrored image, which ours
// isn't. Two hands on the same side fall back to left-to-right order.
export function computeHandMetrics(hands, frame, aspect = 1) {
  const sorted = [...hands].sort((a, b) => a[WRIST].x - b[WRIST].x);
  const slots = [null, null];

  if (sorted.length === 1) {
    slots[sorted[0][WRIST].x < 0.5 ? 0 : 1] = describeHand(sorted[0], frame, aspect);
  } else if (sorted.length >= 2) {
    slots[0] = describeHand(sorted[0], frame, aspect);
    slots[1] = describeHand(sorted[1], frame, aspect);
  }
  return { slots };
}

// Mirroring swaps which side each hand is on and flips its orientation and face-relative x.
export function mirrorHandMetrics(metrics) {
  const flip = (hand) =>
    hand && {
      curls: hand.curls,
      orient: -hand.orient,
      rel: hand.rel && { x: -hand.rel.x, y: hand.rel.y },
    };
  return { slots: [flip(metrics.slots[1]), flip(metrics.slots[0])] };
}

const CURL_SCALE_DEG = 90;
const ORIENT_SCALE_DEG = 90;
const POSITION_SCALE = 3; // in face widths

export function scoreHands(reference, live) {
  // No hands found in the reference (often because they're cropped out of frame) means
  // there's nothing to compare against — penalizing the live hands would be a mismatch
  // the user has no way to fix.
  if (!reference.slots[0] && !reference.slots[1]) return { diffs: {}, empty: true };

  const diffs = {};

  for (let i = 0; i < 2; i++) {
    const r = reference.slots[i];
    const l = live.slots[i];
    if (!r && !l) continue;

    // A hand in one frame but not the other is the biggest mismatch there is.
    if (!r || !l) {
      diffs[`hand${i}_presence`] = 1;
      continue;
    }

    for (const finger of Object.keys(FINGERS)) {
      diffs[`hand${i}_${finger}`] = norm(r.curls[finger] - l.curls[finger], CURL_SCALE_DEG);
    }
    diffs[`hand${i}_orient`] = norm(r.orient - l.orient, ORIENT_SCALE_DEG);
    if (r.rel && l.rel) {
      diffs[`hand${i}_pos_x`] = norm(r.rel.x - l.rel.x, POSITION_SCALE);
      diffs[`hand${i}_pos_y`] = norm(r.rel.y - l.rel.y, POSITION_SCALE);
    }
  }

  return { ...summarize(diffs), diffs };
}
