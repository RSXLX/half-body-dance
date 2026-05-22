import { CARDINAL8_VALUES, JOINT_IDS, MOTION_SCHEMA_VERSION, type MotionAnalysis } from './motionTypes.js';
import type { PoseData } from './types.js';

export type MotionFetch = typeof fetch;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isFiniteNonNegativeNumber(value: unknown): value is number {
  return isFiniteNumber(value) && value >= 0;
}

function hasStringFields(value: Record<string, unknown>, fields: string[]): boolean {
  return fields.every((field) => typeof value[field] === 'string');
}

function isValidJointId(value: unknown): boolean {
  return typeof value === 'string' && JOINT_IDS.includes(value as never);
}

function isValidCardinal(value: unknown): boolean {
  return typeof value === 'string' && CARDINAL8_VALUES.includes(value as never);
}

function isValidJointStats(stats: unknown): boolean {
  if (!isPlainObject(stats)) return false;
  return Object.entries(stats).every(([joint, value]) => {
    if (!isValidJointId(joint) || !isPlainObject(value)) return false;
    if (!isFiniteNonNegativeNumber(value.distance) || !isFiniteNonNegativeNumber(value.peakSpeed)) return false;
    if (!isValidCardinal(value.cardinal)) return false;
    if (!isFiniteNonNegativeNumber(value.visibility)) return false;
    if (value.pathLength !== undefined && !isFiniteNonNegativeNumber(value.pathLength)) return false;
    if (value.angleDeg !== undefined && !isFiniteNumber(value.angleDeg)) return false;
    return true;
  });
}

function isValidBeatAction(beat: unknown): boolean {
  if (!isPlainObject(beat)) return false;
  if (!isFiniteNonNegativeNumber(beat.index)) return false;
  if (
    !isFiniteNonNegativeNumber(beat.startTime) ||
    !isFiniteNonNegativeNumber(beat.endTime) ||
    !isFiniteNonNegativeNumber(beat.duration)
  ) {
    return false;
  }
  if (beat.endTime < beat.startTime) return false;
  if (!isValidJointId(beat.primaryJoint) || !isValidCardinal(beat.primaryDirection)) return false;
  if (!hasStringFields(beat, ['emoji', 'label'])) return false;
  if (beat.segmentId !== undefined && typeof beat.segmentId !== 'string') return false;
  if (beat.visibilityWarning !== undefined && typeof beat.visibilityWarning !== 'string') return false;
  return isValidJointStats(beat.jointStats);
}

function isValidLimbBeatSlice(slice: unknown): boolean {
  if (!isPlainObject(slice)) return false;
  if (!isFiniteNonNegativeNumber(slice.beatIndex)) return false;
  if (!isFiniteNonNegativeNumber(slice.startTime) || !isFiniteNonNegativeNumber(slice.endTime)) return false;
  if (slice.endTime < slice.startTime) return false;
  if (!isValidCardinal(slice.cardinal) || typeof slice.emoji !== 'string') return false;
  if (!isFiniteNonNegativeNumber(slice.distance) || !isFiniteNonNegativeNumber(slice.peakSpeed)) return false;
  return true;
}

function isValidLimbs(limbs: unknown): boolean {
  if (limbs === undefined) return true;
  if (!isPlainObject(limbs)) return false;
  return Object.entries(limbs).every(([joint, slices]) => {
    return isValidJointId(joint) && Array.isArray(slices) && slices.every(isValidLimbBeatSlice);
  });
}

function isValidHint(hint: unknown): boolean {
  if (!isPlainObject(hint)) return false;
  if (!isFiniteNumber(hint.triggerTime) || !isFiniteNumber(hint.leadMs)) return false;
  if (!hasStringFields(hint, ['segmentId', 'preview'])) return false;
  return hint.cue === 'beatPrep' || hint.cue === 'directionArrow' || hint.cue === 'sectionStart';
}

function isValidSegment(segment: unknown): boolean {
  if (!isPlainObject(segment)) return false;
  if (!hasStringFields(segment, ['id', 'emoji', 'title', 'description'])) return false;
  if (!isFiniteNonNegativeNumber(segment.index)) return false;
  if (
    !isFiniteNonNegativeNumber(segment.startTime) ||
    !isFiniteNonNegativeNumber(segment.endTime) ||
    !isFiniteNonNegativeNumber(segment.duration)
  ) {
    return false;
  }
  if (segment.endTime < segment.startTime) return false;
  if (!Array.isArray(segment.primaryJoints) || !segment.primaryJoints.every(isValidJointId)) return false;
  if (!isValidCardinal(segment.primaryDirection)) return false;
  if (!Array.isArray(segment.keyFrames) || !segment.keyFrames.every(isFiniteNonNegativeNumber)) return false;
  if (!Array.isArray(segment.tips) || !segment.tips.every((tip) => typeof tip === 'string')) return false;
  if (segment.beatIndices !== undefined && (!Array.isArray(segment.beatIndices) || !segment.beatIndices.every(isFiniteNonNegativeNumber))) return false;
  return segment.difficulty === 1 || segment.difficulty === 2 || segment.difficulty === 3;
}

function isValidSummary(summary: unknown): boolean {
  if (!isPlainObject(summary)) return false;
  if (!Array.isArray(summary.primaryJoints) || !summary.primaryJoints.every(isValidJointId)) return false;
  if (!Array.isArray(summary.dominantDirections)) return false;
  return summary.dominantDirections.every((entry) => {
    return isPlainObject(entry) && isValidCardinal(entry.cardinal) && isFiniteNonNegativeNumber(entry.share);
  });
}

export function isMotionAnalysis(value: unknown): value is MotionAnalysis {
  if (!isPlainObject(value)) return false;
  const candidate = value as Partial<MotionAnalysis>;
  if (candidate.schema_version !== MOTION_SCHEMA_VERSION) return false;
  if (typeof candidate.source_pose !== 'string') return false;
  if (typeof candidate.fps !== 'number' || !Number.isFinite(candidate.fps) || candidate.fps <= 0) return false;
  if (typeof candidate.duration !== 'number' || !Number.isFinite(candidate.duration) || candidate.duration < 0) return false;
  if (typeof candidate.extracted_at !== 'string') return false;
  if (!isPlainObject(candidate.extract_config)) return false;
  if (!Array.isArray(candidate.beats) || !Array.isArray(candidate.segments) || !Array.isArray(candidate.hints)) {
    return false;
  }
  if (!isValidSummary(candidate.summary) || !isPlainObject(candidate.trajectories)) return false;
  return (
    candidate.beats.every(isValidBeatAction) &&
    isValidLimbs(candidate.limbs) &&
    candidate.segments.every(isValidSegment) &&
    candidate.hints.every(isValidHint)
  );
}

export function getSegmentStartTime(segment: unknown): number | null {
  if (!isPlainObject(segment) || !isFiniteNonNegativeNumber(segment.startTime)) return null;
  return segment.startTime;
}

export async function loadMotionFromPath(path: string, fetchImpl: MotionFetch = fetch): Promise<MotionAnalysis | null> {
  try {
    const response = await fetchImpl(path);
    if (!response.ok) return null;
    const json = await response.json();
    return isMotionAnalysis(json) ? json : null;
  } catch (err) {
    console.warn('[motionData] failed to load motion', err);
    return null;
  }
}

export async function analyzeMotionFromPose(
  poseJson: PoseData | Record<string, unknown>,
  fetchImpl: MotionFetch = fetch,
): Promise<MotionAnalysis | null> {
  try {
    const response = await fetchImpl('/api/analyze-motion', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ poseJson }),
    });
    if (!response.ok) return null;
    const payload = await response.json();
    return payload?.ok && isMotionAnalysis(payload.motion) ? payload.motion : null;
  } catch (err) {
    console.warn('[motionData] failed to analyze motion', err);
    return null;
  }
}
