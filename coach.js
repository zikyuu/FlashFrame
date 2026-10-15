// Turns the three meters' signed deltas into director-style instructions ("raise your right
// arm a little, bend your right elbow"), and speaks them with the browser's built-in
// text-to-speech (speechSynthesis — the device's own voice, works in iOS Safari and Android
// Chrome with no key or network).
//
// Direction conventions. A delta is always (live - reference), compared in the same space
// the meters use, so sides are the USER's body sides: the live frame is the true camera view,
// where the person's right hand/shoulder appears on the image's LEFT. When the display is
// mirrored the reference is mirrored too, so "your right arm" stays correct either way.

const CUE_MIN = 0.22; // below this a metric is close enough not to nag about

const FACE_NAMES = new Set([
  "head_tilt", "head_yaw", "head_pitch", "mouth_smile", "mouth_frown", "mouth_pucker",
  "mouth_press", "jaw_open", "brow_raise", "brow_down", "eye_squint", "eye_wide", "cheek_puff",
]);

const tail = (a) => (a ? ` ${a}` : "");
const more = (a) => (a ? `${a} more` : "more");
const less = (a) => (a ? `${a} less` : "less");

// "a little" / "" / "a lot" from how far off the metric is (0-1).
const amount = (mag) => (mag < 0.35 ? "a little" : mag < 0.6 ? "" : "a lot");

function bodyCue(name, d, a) {
  const side = name.startsWith("left_") ? "left" : "right";
  const joint = name.replace(/^(left|right)_/, "");

  switch (joint) {
    case "elbow":
      return d > 0 ? `bend your ${side} elbow${tail(a)}` : `straighten your ${side} arm${tail(a)}`;
    case "shoulder": // angle between arm and spine: bigger = arm raised higher
      return d > 0 ? `lower your ${side} arm${tail(a)}` : `raise your ${side} arm${tail(a)}`;
    case "knee":
      return d > 0 ? `bend your ${side} knee${tail(a)}` : `straighten your ${side} leg${tail(a)}`;
    case "hip":
      return d > 0 ? `bend forward at the hips${tail(a)}` : `stand ${a ? a + " " : ""}taller`;
    default:
  }

  switch (name) {
    // tilts: positive = the person's right side is lower
    case "shoulder_tilt":
      return `lift your ${d > 0 ? "right" : "left"} shoulder${tail(a)}`;
    case "hip_tilt":
      return `lift your ${d > 0 ? "right" : "left"} hip${tail(a)}`;
    // lean: positive = leaning toward the image's right = the person's left
    case "torso_lean":
      return `lean ${more(a)} toward your ${d > 0 ? "right" : "left"}`;
    case "neck_tilt":
      return `move your head${tail(a)} toward your ${d > 0 ? "right" : "left"}`;
    default:
      return null;
  }
}

function faceCue(name, d, a) {
  switch (name) {
    case "head_tilt": // positive = head tilted toward the person's left
      return `tilt your head${tail(a)} toward your ${d > 0 ? "right" : "left"}`;
    case "head_yaw": // positive = nose toward the person's left
      return `turn your head${tail(a)} toward your ${d > 0 ? "right" : "left"}`;
    case "head_pitch": // positive = nose lower in the eye line = chin raised
      return d > 0 ? `lower your chin${tail(a)}` : `lift your chin${tail(a)}`;
    case "mouth_smile":
      return d > 0 ? `smile ${less(a)}` : `smile ${more(a)}`;
    case "mouth_frown":
      return d > 0 ? "relax your mouth" : null;
    case "mouth_pucker":
      return d > 0 ? "relax your lips" : `purse your lips${tail(a)}`;
    case "mouth_press":
      return d > 0 ? "relax your lips" : "press your lips together";
    case "jaw_open":
      return d > 0 ? `close your mouth${tail(a)}` : `open your mouth${tail(a)}`;
    case "brow_raise":
      return d > 0 ? "relax your eyebrows" : `raise your eyebrows${tail(a)}`;
    case "brow_down":
      return d > 0 ? "relax your frown" : `bring your brows down${tail(a)}`;
    case "eye_squint":
      return d > 0 ? "soften your squint" : `squint${tail(a)}`;
    case "eye_wide":
      return d > 0 ? "open your eyes less wide" : "open your eyes wider";
    case "cheek_puff":
      return d > 0 ? "relax your cheeks" : null;
    default:
      return null;
  }
}

// Hand slot 0 is the image's left half = the person's RIGHT hand (true camera view).
function handCue(name, d, a) {
  const m = name.match(/^hand(\d)_(.+)$/);
  if (!m) return null;
  const slot = Number(m[1]);
  const hand = slot === 0 ? "right" : "left";
  const key = m[2];
  // d > 0 means you're further toward the image's right than the reference, so the fix is to
  // go image-left — which is outward for the right hand and inward for the left hand.
  const outward = d > 0 === (slot === 0);

  switch (key) {
    case "presence":
      return d > 0 ? `lower your ${hand} hand out of the frame` : `bring your ${hand} hand into the frame`;
    case "pos_y":
      return d > 0 ? `raise your ${hand} hand${tail(a)}` : `lower your ${hand} hand${tail(a)}`;
    case "pos_x":
      return `move your ${hand} hand${tail(a)} ${outward ? "outward" : "inward"}`;
    case "orient": // same rule: pointing too far image-right means rotate it image-left
      return `turn your ${hand} hand${tail(a)} ${outward ? "outwards" : "inwards"}`;
    case "thumb":
      return d > 0 ? `curl your ${hand} thumb${tail(a)}` : `open your ${hand} thumb${tail(a)}`;
    default: // index / middle / ring / pinky — bigger curl angle = straighter finger
      return d > 0 ? `curl your ${hand} ${key} finger${tail(a)}` : `straighten your ${hand} ${key} finger${tail(a)}`;
  }
}

function cueFor(name, delta, mag) {
  if (delta === undefined) return null;
  const a = amount(mag);
  if (name.startsWith("hand")) return handCue(name, delta, a);
  if (FACE_NAMES.has(name)) return faceCue(name, delta, a);
  return bodyCue(name, delta, a);
}

// Which part an instruction is about — two instructions are only combined when they're
// about the same part (the same hand, the same arm/leg, the face, the trunk).
function groupOf(name) {
  if (name.startsWith("hand")) return name.slice(0, 5); // hand0 / hand1
  if (FACE_NAMES.has(name)) return "face";
  if (name.startsWith("left_")) return "left";
  if (name.startsWith("right_")) return "right";
  return "trunk";
}

// results: { body, face, hands } as built by the app — each has level/diffs/deltas when it
// was actually scored. Returns the instruction to give right now.
export function cueFromResults(results) {
  const scored = Object.values(results).filter((r) => r.level);
  if (scored.length === 0) {
    return { kind: "unseen", text: "I can't see enough of you — step back or face the camera." };
  }

  const items = [];
  for (const r of scored) {
    for (const [name, mag] of Object.entries(r.diffs ?? {})) {
      const text = cueFor(name, r.deltas?.[name], mag);
      if (text) items.push({ name, mag, text });
    }
  }
  items.sort((x, y) => y.mag - x.mag);

  const top = items.filter((i) => i.mag >= CUE_MIN);
  if (top.length === 0) {
    const perfect = scored.every((r) => r.level === "green");
    return { kind: perfect ? "perfect" : "good", text: perfect ? "Perfect — hold still." : "Looking good — hold it there." };
  }

  // Add a second instruction when it's about the same part and nearly as off — "raise your
  // right hand, turn it inwards" — rather than reading out every metric.
  const first = top[0];
  const second = top.find(
    (i) => i !== first && i.text !== first.text && groupOf(i.name) === groupOf(first.name) && i.mag >= first.mag * 0.6
  );
  // "raise your right hand a little, move it a little outward" — not the full noun twice.
  let secondText = second?.text;
  const subject = first.text.match(/your (?:left|right) [a-z]+/)?.[0];
  if (secondText && subject) secondText = secondText.replace(subject, "it");
  const text = secondText ? `${first.text}, ${secondText}` : first.text;
  return { kind: "adjust", text: text.charAt(0).toUpperCase() + text.slice(1) + "." };
}

// Only show/say an instruction once it has been the top one for a moment, so the text and
// voice don't flap between two near-equal metrics every frame.
export function createStabilizer(holdMs = 700) {
  let shown = "";
  let candidate = "";
  let since = 0;
  return (text, now) => {
    if (text === shown) {
      candidate = text;
      return shown;
    }
    if (text !== candidate) {
      candidate = text;
      since = now;
    }
    if (now - since >= holdMs) shown = text;
    return shown;
  };
}

export function createSpeaker() {
  const supported = typeof window !== "undefined" && "speechSynthesis" in window;
  const MIN_GAP_MS = 2200; // breathing room between instructions
  const REPEAT_MS = 8000; // don't repeat the same instruction more often than this
  let enabled = false;
  let lastText = "";
  let lastAt = 0;

  function speak(text, now) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    window.speechSynthesis.speak(utterance);
    lastText = text;
    lastAt = now;
  }

  return {
    supported,
    get enabled() {
      return enabled;
    },
    // Must be called from a user gesture (tap) — mobile browsers refuse to speak otherwise.
    enable() {
      if (!supported) return false;
      enabled = true;
      window.speechSynthesis.cancel();
      speak("Voice coach on.", performance.now());
      return true;
    },
    disable() {
      enabled = false;
      if (supported) window.speechSynthesis.cancel();
    },
    // The regular, rate-limited stream of instructions.
    say(text, now) {
      if (!enabled || !text) return;
      if (window.speechSynthesis.speaking || window.speechSynthesis.pending) return;
      if (now - lastAt < MIN_GAP_MS) return;
      if (text === lastText && now - lastAt < REPEAT_MS) return;
      speak(text, now);
    },
    // One-off announcements ("Get ready", "Got it") that cut in.
    announce(text) {
      if (!enabled) return;
      window.speechSynthesis.cancel();
      speak(text, performance.now());
    },
    stop() {
      if (supported) window.speechSynthesis.cancel();
    },
  };
}
