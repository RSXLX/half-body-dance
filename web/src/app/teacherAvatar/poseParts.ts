import type { NormalizedLandmark, PoseFrame } from '../../core/types.js';
import { getVisibility } from '../../core/geometry.js';
import { getPoseReference } from '../../core/reference.js';
import type { StageRect } from '../stageRenderer.js';

export type TeacherBodyPartId =
  | 'leftUpperArm'
  | 'leftForearm'
  | 'rightUpperArm'
  | 'rightForearm'
  | 'leftThigh'
  | 'leftCalf'
  | 'rightThigh'
  | 'rightCalf';

export interface TeacherBodyPart {
  id: TeacherBodyPartId;
  from: number;
  to: number;
  startRatio: number;
  endRatio: number;
  curvature: number;
}

export interface TeacherLimbMetric {
  startRadius: number;
  endRadius: number;
  curvature: number;
}

export interface TeacherAvatarMetrics {
  scalePx: number;
  limbs: Record<TeacherBodyPartId, TeacherLimbMetric>;
  joints: {
    shoulder: number;
    elbow: number;
    wrist: number;
    hip: number;
    knee: number;
    ankle: number;
  };
  extremities: {
    hand: number;
    foot: number;
  };
  head: {
    baseRadius: number;
  };
}

export const TEACHER_BODY_PARTS: readonly TeacherBodyPart[] = [
  { id: 'leftUpperArm', from: 11, to: 13, startRatio: 0.12, endRatio: 0.095, curvature: -0.035 },
  { id: 'leftForearm', from: 13, to: 15, startRatio: 0.095, endRatio: 0.07, curvature: 0.032 },
  { id: 'rightUpperArm', from: 12, to: 14, startRatio: 0.12, endRatio: 0.095, curvature: 0.035 },
  { id: 'rightForearm', from: 14, to: 16, startRatio: 0.095, endRatio: 0.07, curvature: -0.032 },
  { id: 'leftThigh', from: 23, to: 25, startRatio: 0.15, endRatio: 0.11, curvature: 0.022 },
  { id: 'leftCalf', from: 25, to: 27, startRatio: 0.11, endRatio: 0.075, curvature: -0.025 },
  { id: 'rightThigh', from: 24, to: 26, startRatio: 0.15, endRatio: 0.11, curvature: -0.022 },
  { id: 'rightCalf', from: 26, to: 28, startRatio: 0.11, endRatio: 0.075, curvature: 0.025 },
];

export function hasRenderablePose(frame: PoseFrame | null | undefined): boolean {
  return Array.isArray(frame?.pose_landmarks) && frame.pose_landmarks.length >= 29;
}

export function findNearestRenderableFrame(
  frames: readonly PoseFrame[] | null | undefined,
  index: number,
): PoseFrame | null {
  if (!frames?.length) return null;
  const start = Math.min(frames.length - 1, Math.max(0, Math.floor(index)));
  if (hasRenderablePose(frames[start])) return frames[start] ?? null;
  for (let offset = 1; offset < frames.length; offset += 1) {
    const forward = start + offset;
    if (forward < frames.length && hasRenderablePose(frames[forward])) return frames[forward] ?? null;
    const backward = start - offset;
    if (backward >= 0 && hasRenderablePose(frames[backward])) return frames[backward] ?? null;
  }
  return frames[start] ?? null;
}

export function getVisibleTeacherBodyParts(
  points: readonly (NormalizedLandmark | null | undefined)[],
  minVisibility = 0.25,
): TeacherBodyPart[] {
  return TEACHER_BODY_PARTS.filter((part) => {
    const from = points[part.from];
    const to = points[part.to];
    return !!from && !!to && getVisibility(from) >= minVisibility && getVisibility(to) >= minVisibility;
  });
}

export function getTeacherAvatarMetrics(
  points: readonly (NormalizedLandmark | null | undefined)[],
  rect: StageRect,
): TeacherAvatarMetrics {
  const reference = getPoseReference(points);
  const scalePx = (reference?.scale || 0.22) * Math.min(rect.width, rect.height);
  const radius = (ratio: number, min: number) => Math.max(min, scalePx * ratio);
  const limbs = {} as Record<TeacherBodyPartId, TeacherLimbMetric>;
  for (const part of TEACHER_BODY_PARTS) {
    limbs[part.id] = {
      startRadius: radius(part.startRatio, part.id.includes('Thigh') ? 5 : 3),
      endRadius: radius(part.endRatio, part.id.includes('Calf') ? 4 : 3),
      curvature: part.curvature,
    };
  }
  const joint = radius(0.075, 3.5);
  return {
    scalePx,
    limbs,
    joints: {
      shoulder: joint * 1.12,
      elbow: joint,
      wrist: joint * 0.62,
      hip: joint * 1.18,
      knee: joint,
      ankle: joint * 0.58,
    },
    extremities: {
      hand: radius(0.075, 3),
      foot: radius(0.09, 4),
    },
    head: {
      baseRadius: radius(0.22, 12),
    },
  };
}
