/**
 * Torso-based reference frame.
 *
 * Reproduces getPoseReference + transformPointsWithReferences + normalizeLandmarks
 * from pose_viewer.html. Do not change the shape of `PoseReference` without
 * updating the legacy HTML in lockstep, or the golden-sample regression will
 * fail once PracticeView is migrated.
 */

import type { NormalizedLandmark, PoseReference } from './types.js';
import { distance, getPerpendicular, getVisibility, normalizeVector } from './geometry.js';

export function getPoseReference(
  points: readonly (NormalizedLandmark | null | undefined)[] | null | undefined,
): PoseReference | null {
  if (!Array.isArray(points) || points.length < 29) return null;
  const leftShoulder = points[11];
  const rightShoulder = points[12];
  const leftHip = points[23];
  const rightHip = points[24];
  if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) return null;

  const shoulderCenter = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2,
  };
  const hipCenter = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2,
  };
  const center = {
    x: (shoulderCenter.x + hipCenter.x) / 2,
    y: (shoulderCenter.y + hipCenter.y) / 2,
  };

  const shoulderAxis = normalizeVector({
    x: rightShoulder.x - leftShoulder.x,
    y: rightShoulder.y - leftShoulder.y,
  });
  const hipAxis = normalizeVector({
    x: rightHip.x - leftHip.x,
    y: rightHip.y - leftHip.y,
  });
  const torsoAxisRaw = {
    x: hipCenter.x - shoulderCenter.x,
    y: hipCenter.y - shoulderCenter.y,
  };

  const xAxis = shoulderAxis || hipAxis || { x: 1, y: 0 };
  let yAxis =
    normalizeVector({
      x: torsoAxisRaw.x - (torsoAxisRaw.x * xAxis.x + torsoAxisRaw.y * xAxis.y) * xAxis.x,
      y: torsoAxisRaw.y - (torsoAxisRaw.x * xAxis.x + torsoAxisRaw.y * xAxis.y) * xAxis.y,
    }) || normalizeVector(getPerpendicular(xAxis));
  if (!yAxis) yAxis = { x: 0, y: 1 };

  const scale = Math.max(
    distance(leftShoulder, rightShoulder),
    distance(leftHip, rightHip),
    distance(shoulderCenter, hipCenter),
    0.001,
  );

  return { center, scale, xAxis, yAxis };
}

export function transformPointsWithReferences<T extends NormalizedLandmark>(
  points: readonly T[],
  sourceReference: PoseReference | null,
  targetReference: PoseReference | null,
): T[] | readonly T[] {
  if (!Array.isArray(points) || !sourceReference || !targetReference) return points;
  return points.map((point) => {
    if (!point) return point;
    const normalizedX = (point.x - sourceReference.center.x) / sourceReference.scale;
    const normalizedY = (point.y - sourceReference.center.y) / sourceReference.scale;
    return {
      ...point,
      x:
        targetReference.center.x +
        (normalizedX * targetReference.xAxis.x + normalizedY * targetReference.yAxis.x) *
          targetReference.scale,
      y:
        targetReference.center.y +
        (normalizedX * targetReference.xAxis.y + normalizedY * targetReference.yAxis.y) *
          targetReference.scale,
    };
  });
}

export interface NormalizedPoint {
  x: number;
  y: number;
  v: number;
}

export function normalizeLandmarks(
  points: readonly (NormalizedLandmark | null | undefined)[] | null | undefined,
): NormalizedPoint[] | null {
  const reference = getPoseReference(points);
  if (!reference) return null;
  return (points as readonly NormalizedLandmark[]).map((point) => ({
    x: (point.x - reference.center.x) / reference.scale,
    y: (point.y - reference.center.y) / reference.scale,
    v: getVisibility(point),
  }));
}

export function getDefaultStageReference(): PoseReference {
  return {
    center: { x: 0.5, y: 0.5 },
    scale: 0.28,
    xAxis: { x: 1, y: 0 },
    yAxis: { x: 0, y: 1 },
  };
}
