/**
 * Hand landmark utilities, ported from pose_viewer.html.
 *
 * Inputs accept either raw 21-landmark arrays (the in-frame shape) or the
 * `{handedness, landmarks}` hand frames emitted by extract_pose.py and the
 * MediaPipe Solutions runtime.
 */

import type { HandFrame, NormalizedLandmark } from './types.js';
import { cosineSimilarity2D, type Vec2 } from './similarity.js';
import { distance, getVector } from './geometry.js';

export function getHandScale(points: readonly NormalizedLandmark[] | null | undefined): number {
  if (!Array.isArray(points) || points.length < 18) return 0;
  const wrist = points[0];
  const indexBase = points[5];
  const pinkyBase = points[17];
  if (!wrist || !indexBase || !pinkyBase) return 0;
  return Math.max(
    distance(wrist, indexBase),
    distance(wrist, pinkyBase),
    distance(indexBase, pinkyBase),
    0.001,
  );
}

export interface NormalizedHandPoint {
  x: number;
  y: number;
}

export function normalizeHandLandmarks(
  points: readonly NormalizedLandmark[] | null | undefined,
): NormalizedHandPoint[] | null {
  if (!Array.isArray(points) || points.length < 21) return null;
  const wrist = points[0];
  const scale = getHandScale(points);
  if (!wrist || !scale) return null;
  return points.map((point) => ({
    x: (point.x - wrist.x) / scale,
    y: (point.y - wrist.y) / scale,
  }));
}

function extractLandmarks(hand: HandFrame | NormalizedLandmark[] | null | undefined): readonly NormalizedLandmark[] | null {
  if (!hand) return null;
  if (Array.isArray(hand)) return hand;
  return Array.isArray(hand.landmarks) ? hand.landmarks : null;
}

/**
 * Compare a user hand frame against the teacher hand frame.
 * Returns an integer score in [0, 100].
 */
export function compareHands(
  userHand: HandFrame | NormalizedLandmark[] | null | undefined,
  targetHand: HandFrame | NormalizedLandmark[] | null | undefined,
): number {
  const userLandmarks = extractLandmarks(userHand);
  const targetLandmarks = extractLandmarks(targetHand);
  const normalizedUser = normalizeHandLandmarks(userLandmarks);
  const normalizedTarget = normalizeHandLandmarks(targetLandmarks);
  if (!normalizedUser || !normalizedTarget) return 0;

  const fingertipIndices = [4, 8, 12, 16, 20] as const;
  let fingertipScore = 0;
  fingertipIndices.forEach((index) => {
    const userPoint = normalizedUser[index];
    const targetPoint = normalizedTarget[index];
    if (!userPoint || !targetPoint) return;
    const gap = Math.hypot(userPoint.x - targetPoint.x, userPoint.y - targetPoint.y);
    fingertipScore += 1 - Math.min(1, gap / 1.2);
  });
  fingertipScore /= fingertipIndices.length;

  const fingerSegments: Array<[number, number]> = [
    [0, 4],
    [0, 8],
    [0, 12],
    [0, 16],
    [0, 20],
    [5, 8],
    [9, 12],
    [13, 16],
    [17, 20],
  ];
  let segmentScore = 0;
  fingerSegments.forEach(([a, b]) => {
    const userVector = getVector(normalizedUser, a, b) as Vec2 | null;
    const targetVector = getVector(normalizedTarget, a, b) as Vec2 | null;
    segmentScore += (cosineSimilarity2D(userVector, targetVector) + 1) / 2;
  });
  segmentScore /= fingerSegments.length;

  return Math.max(0, Math.round((segmentScore * 0.65 + fingertipScore * 0.35) * 100));
}

/**
 * Pick the best user-side hand frame that matches a teacher hand frame.
 *
 * 1. handedness match wins outright.
 * 2. otherwise fall back to the hand whose wrist landmark is closest to the
 *    teacher wrist in the raw coordinate space.
 */
export function findMatchingHand(
  userHands: readonly HandFrame[] | null | undefined,
  targetHand: HandFrame | null | undefined,
): HandFrame | null {
  if (!Array.isArray(userHands) || !userHands.length || !targetHand) return null;

  const targetLabel = targetHand.handedness;
  if (targetLabel) {
    const exact = userHands.find((hand) => hand.handedness === targetLabel);
    if (exact) return exact;
  }

  const targetRoot = targetHand.landmarks?.[0];
  if (!targetRoot) return userHands[0] ?? null;

  return userHands.reduce<HandFrame | null>((best, hand) => {
    if (!hand?.landmarks?.[0]) return best;
    if (!best?.landmarks?.[0]) return hand;
    const bestDistance = distance(best.landmarks[0], targetRoot);
    const nextDistance = distance(hand.landmarks[0], targetRoot);
    return nextDistance < bestDistance ? hand : best;
  }, null);
}
