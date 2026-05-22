/**
 * Arm-focused pose comparison — ported verbatim from pose_viewer.html so the
 * scores stay identical byte-for-byte.
 *
 * Scoring weights:
 *   Arm total:          upperArm 35% + forearm 35% + elbow 20% + wrist 10%
 *   Overall body total: leftArm 50% + rightArm 50%
 *   Segment score:      directionScore 75% + lengthScore 25%
 *
 * Landmark indices (MediaPipe Pose 33-point layout):
 *   11 left shoulder, 13 left elbow, 15 left wrist
 *   12 right shoulder, 14 right elbow, 16 right wrist
 */

import type { NormalizedLandmark } from './types.js';
import { normalizeLandmarks, type NormalizedPoint } from './reference.js';
import { cosineSimilarity2D, type Vec2 } from './similarity.js';
import { getVector, getVisibility } from './geometry.js';

export const ARM_SCORE_SYSTEM_LABEL = '上臂 35% + 前臂 35% + 肘部 20% + 手腕 10%';

export interface ArmSideConfig {
  label: string;
  upperArm: [number, number];
  forearm: [number, number];
  elbow: number;
  wrist: number;
}

export const ARM_SCORING_CONFIG: Readonly<{ left: ArmSideConfig; right: ArmSideConfig }> = {
  left: { label: '左臂', upperArm: [11, 13], forearm: [13, 15], elbow: 13, wrist: 15 },
  right: { label: '右臂', upperArm: [12, 14], forearm: [14, 16], elbow: 14, wrist: 16 },
};

export interface WeightedItem {
  score: number | null | undefined;
  weight: number;
}

export function weightedAverage(items: readonly WeightedItem[] | null | undefined): number | null {
  const validItems = (items ?? []).filter(
    (item): item is { score: number; weight: number } =>
      !!item && typeof item.score === 'number' && Number.isFinite(item.score) && item.weight > 0,
  );
  if (!validItems.length) return null;
  const totalWeight = validItems.reduce((sum, item) => sum + item.weight, 0);
  if (!totalWeight) return null;
  const totalScore = validItems.reduce((sum, item) => sum + item.score * item.weight, 0);
  return Math.round(totalScore / totalWeight);
}

export function scoreArmSegment(
  normalizedUser: readonly NormalizedPoint[],
  normalizedTarget: readonly NormalizedPoint[],
  a: number,
  b: number,
): number | null {
  const userA = normalizedUser[a];
  const userB = normalizedUser[b];
  const targetA = normalizedTarget[a];
  const targetB = normalizedTarget[b];
  if ([userA, userB, targetA, targetB].some((point) => getVisibility(point) < 0.35)) {
    return null;
  }
  const userVector = getVector(normalizedUser, a, b) as Vec2 | null;
  const targetVector = getVector(normalizedTarget, a, b) as Vec2 | null;
  if (!userVector || !targetVector) return null;
  const directionScore = (cosineSimilarity2D(userVector, targetVector) + 1) / 2;
  const userLength = Math.hypot(userVector.x, userVector.y);
  const targetLength = Math.hypot(targetVector.x, targetVector.y);
  const lengthScore = 1 - Math.min(1, Math.abs(userLength - targetLength) / Math.max(targetLength, 0.001));
  return Math.max(0, Math.round((directionScore * 0.75 + lengthScore * 0.25) * 100));
}

export function scoreArmJoint(
  normalizedUser: readonly NormalizedPoint[],
  normalizedTarget: readonly NormalizedPoint[],
  index: number,
  maxGap = 0.9,
): number | null {
  const userPoint = normalizedUser[index];
  const targetPoint = normalizedTarget[index];
  if (!userPoint || !targetPoint) return null;
  if (getVisibility(userPoint) < 0.35 || getVisibility(targetPoint) < 0.35) return null;
  const gap = Math.hypot(userPoint.x - targetPoint.x, userPoint.y - targetPoint.y);
  return Math.max(0, Math.round((1 - Math.min(1, gap / maxGap)) * 100));
}

export interface ArmSideScore {
  label: string;
  score: number | null;
  summary: string;
  components: {
    upperArm: number | null;
    forearm: number | null;
    elbow: number | null;
    wrist: number | null;
  };
}

export function analyzeArmSide(
  normalizedUser: readonly NormalizedPoint[],
  normalizedTarget: readonly NormalizedPoint[],
  config: ArmSideConfig,
): ArmSideScore {
  const upperArm = scoreArmSegment(normalizedUser, normalizedTarget, config.upperArm[0], config.upperArm[1]);
  const forearm = scoreArmSegment(normalizedUser, normalizedTarget, config.forearm[0], config.forearm[1]);
  const elbow = scoreArmJoint(normalizedUser, normalizedTarget, config.elbow);
  const wrist = scoreArmJoint(normalizedUser, normalizedTarget, config.wrist, 0.8);

  const score = weightedAverage([
    { score: upperArm, weight: 0.35 },
    { score: forearm, weight: 0.35 },
    { score: elbow, weight: 0.2 },
    { score: wrist, weight: 0.1 },
  ]);

  return {
    label: config.label,
    score,
    summary: score === null ? '未识别' : `${score}%`,
    components: { upperArm, forearm, elbow, wrist },
  };
}

export interface ArmPoseScore {
  total: number;
  left: ArmSideScore;
  right: ArmSideScore;
  scoringSystem: string;
}

export function compareArmPoses(
  userPose: readonly NormalizedLandmark[] | null | undefined,
  targetPose: readonly NormalizedLandmark[] | null | undefined,
): ArmPoseScore | null {
  const normalizedUser = normalizeLandmarks(userPose);
  const normalizedTarget = normalizeLandmarks(targetPose);
  if (!normalizedUser || !normalizedTarget) return null;

  const left = analyzeArmSide(normalizedUser, normalizedTarget, ARM_SCORING_CONFIG.left);
  const right = analyzeArmSide(normalizedUser, normalizedTarget, ARM_SCORING_CONFIG.right);
  const total = weightedAverage([
    { score: left.score, weight: 0.5 },
    { score: right.score, weight: 0.5 },
  ]);
  if (total === null) return null;
  return { total, left, right, scoringSystem: ARM_SCORE_SYSTEM_LABEL };
}

export function scoreLabelFromValue(score: number): string {
  if (score >= 85) return '高度匹配';
  if (score >= 70) return '基本匹配';
  if (score >= 50) return '部分匹配';
  return '未匹配';
}
