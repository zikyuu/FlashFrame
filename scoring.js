// Shared scoring for all three meters. Every metric is first normalized to a 0-1 "how far
// off" diff by its caller (see norm), so body angles, face blendshapes and hand curls
// can be judged with the same thresholds.
// Calibrated on recordings of someone deliberately holding a pose that visibly matched the
// reference: it scored a steady ~28% (best 21%), while clearly wrong poses (standing up
// against a leaning reference, mid-motion, a close-up face) scored 38-61%. These cutoffs put
// a good match in green/yellow and every clearly wrong pose in red.
export const THRESHOLDS = { greenMax: 0.25, yellowMax: 0.35 };

// Where the meter marker sits (0 = red end, 1 = green end). Each colour band gets a third
// of the bar, so the marker's position always agrees with the label and colour.
export function meterPosition(score) {
  const { greenMax: g, yellowMax: y } = THRESHOLDS;
  if (score <= g) return 1 - score / g / 3;
  if (score <= y) return 2 / 3 - (score - g) / (y - g) / 3;
  return Math.max(0, 1 / 3 - Math.min((score - y) / y, 1) / 3);
}

export function norm(diff, scale) {
  return Math.min(Math.abs(diff) / scale, 1);
}

function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

// Blend of the plain average and the average of the two worst metrics. A plain mean lets
// a few matching metrics hide a big mismatch (raise both hands, face still "matches");
// blending in the worst offenders means one clear miss can pull the meter out of green.
export function summarize(diffs) {
  const values = Object.values(diffs);
  if (values.length === 0) return null;

  const sorted = [...values].sort((a, b) => b - a);
  const score = (mean(sorted) + mean(sorted.slice(0, 2))) / 2;
  return { score, level: levelFor(score) };
}

export function levelFor(score) {
  return score <= THRESHOLDS.greenMax ? "green" : score <= THRESHOLDS.yellowMax ? "yellow" : "red";
}
