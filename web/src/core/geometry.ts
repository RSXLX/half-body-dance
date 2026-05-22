/**
 * Shared geometric helpers for the scoring core.
 *
 * Ported verbatim from pose_viewer.html (distance / getVisibility / getVector /
 * normalizeVector / getPerpendicular) so that score outputs stay identical.
 */

import type { NormalizedLandmark } from './types.js';
import type { Vec2 } from './similarity.js';

export function distance(p1: Vec2 | null | undefined, p2: Vec2 | null | undefined): number {
  if (!p1 || !p2) return 0;
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

export function getVisibility(point: NormalizedLandmark | { v?: number } | null | undefined): number {
  if (!point) return 0;
  const asV = (point as { v?: number }).v;
  if (typeof asV === 'number') return asV;
  const vis = (point as NormalizedLandmark).visibility;
  if (typeof vis === 'number') return vis;
  return 1;
}

export function getVector(
  points: readonly (NormalizedLandmark | null | undefined)[],
  a: number,
  b: number,
): Vec2 | null {
  const pa = points[a];
  const pb = points[b];
  if (!pa || !pb) return null;
  return { x: pb.x - pa.x, y: pb.y - pa.y };
}

export function normalizeVector(vector: Vec2 | null | undefined): Vec2 | null {
  if (!vector) return null;
  const length = Math.hypot(vector.x, vector.y);
  if (!length) return null;
  return { x: vector.x / length, y: vector.y / length };
}

export function getPerpendicular(vector: Vec2): Vec2 {
  return { x: -vector.y, y: vector.x };
}
