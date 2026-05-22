export { cosineSimilarity, cosineSimilarity2D, cosineSimilarityN } from './similarity.js';
export type { Vec2 } from './similarity.js';

export { distance, getVector, getVisibility, normalizeVector, getPerpendicular } from './geometry.js';

export {
  getPoseReference,
  transformPointsWithReferences,
  normalizeLandmarks,
  getDefaultStageReference,
} from './reference.js';
export type { NormalizedPoint } from './reference.js';

export {
  compareHands,
  findMatchingHand,
  normalizeHandLandmarks,
  getHandScale,
} from './hands.js';
export type { NormalizedHandPoint } from './hands.js';

export {
  ARM_SCORE_SYSTEM_LABEL,
  ARM_SCORING_CONFIG,
  analyzeArmSide,
  compareArmPoses,
  scoreArmJoint,
  scoreArmSegment,
  scoreLabelFromValue,
  weightedAverage,
} from './poseCompare.js';
export type { ArmPoseScore, ArmSideConfig, ArmSideScore, WeightedItem } from './poseCompare.js';

export type {
  NormalizedLandmark,
  HandFrame,
  PoseFrame,
  PoseData,
  PoseReference,
} from './types.js';
