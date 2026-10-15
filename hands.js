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

// Two slots: 0 = the image's left hand (the person's RIGHT hand in a true camera view),
// 1 = the image's right hand. Two hands are ordered left-to-right by position. A single
// hand uses the model's Left/Right label when there is one (`hand.label`, attached when the
// result is unpacked). Checked against real detections: for our un-mirrored frames the model
// calls the image-LEFT hand "Right" (the person's actual right hand) and the image-right hand
// "Left". That keeps a lone hand resting near the middle of the frame from flipping slots;
// without a label it falls back to which half of the image it's in.
export function computeHandMetrics(hands, frame, aspect = 1) {
  const sorted = [...hands].sort((a, b) => a[WRIST].x - b[WRIST].x);
  const slots = [null, null];

  if (sorted.length === 1) {
    const hand = sorted[0];
    const slot = hand.label ? (hand.label === "Right" ? 0 : 1) : hand[WRIST].x < 0.5 ? 0 : 1;
    slots[slot] = describeHand(hand, frame, aspect);
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
  if (!reference.slots[0] && !reference.slots[1]) return { diffs: {}, deltas: {}, empty: true };

  const diffs = {};
  const deltas = {}; // live minus reference, signed, for the coach
  const add = (name, delta, scale) => {
    deltas[name] = delta;
    diffs[name] = norm(delta, scale);
  };

  for (let i = 0; i < 2; i++) {
    const r = reference.slots[i];
    const l = live.slots[i];
    if (!r && !l) continue;

    // A hand in one frame but not the other is the biggest mismatch there is.
    // Delta is +1 when you have a hand the reference doesn't, -1 when you're missing one.
    if (!r || !l) {
      add(`hand${i}_presence`, l ? 1 : -1, 1);
      continue;
    }

    for (const finger of Object.keys(FINGERS)) {
      add(`hand${i}_${finger}`, l.curls[finger] - r.curls[finger], CURL_SCALE_DEG);
    }
    add(`hand${i}_orient`, l.orient - r.orient, ORIENT_SCALE_DEG);
    if (r.rel && l.rel) {
      add(`hand${i}_pos_x`, l.rel.x - r.rel.x, POSITION_SCALE);
      add(`hand${i}_pos_y`, l.rel.y - r.rel.y, POSITION_SCALE);
    }
  }

  return { ...summarize(diffs), diffs, deltas };
}
