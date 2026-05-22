/**
 * Scoring loop glue — pure functions only.
 *
 * The main practice loop in pose_viewer.html does many things at once (canvas
 * drawing, audio sync, UI state). This module isolates the piece that does not
 * need a DOM: given the current target frame and the user's pose, produce the
 * score + label payload the HUD renders.
 */

import type { NormalizedLandmark, PoseFrame } from '../core/types.js';
import { compareArmPoses, scoreLabelFromValue, type ArmPoseScore } from '../core/poseCompare.js';

export interface ScoringInput {
  userPose: NormalizedLandmark[] | null | undefined;
  targetFrame: PoseFrame | null | undefined;
  /** Whether the user is currently visible to the camera (pose confidence). */
  personVisible: boolean;
  /** Whether the camera pipeline is running. */
  cameraRunning: boolean;
}

export type MatchState =
  | { kind: 'idle'; label: string }
  | { kind: 'no-camera'; label: string }
  | { kind: 'no-person'; label: string }
  | { kind: 'no-target'; label: string }
  | { kind: 'scored'; score: number; armScores: ArmPoseScore; label: string };

export function evaluateFrame(input: ScoringInput): MatchState {
  if (!input.cameraRunning) return { kind: 'no-camera', label: '摄像头未开启' };
  if (!input.targetFrame) return { kind: 'no-target', label: '待机中' };
  if (!input.personVisible || !input.userPose) return { kind: 'no-person', label: '搜索人体中' };
  const targetPose = input.targetFrame.pose_landmarks;
  const armScores = compareArmPoses(input.userPose, targetPose);
  if (!armScores) return { kind: 'no-person', label: '人体不完整' };
  return {
    kind: 'scored',
    score: armScores.total,
    armScores,
    label: scoreLabelFromValue(armScores.total),
  };
}

/** Pick the frame whose `.time` is closest to the given timestamp. */
export function findFrameAtTime(
  frames: readonly PoseFrame[] | null | undefined,
  timeSeconds: number,
): PoseFrame | null {
  if (!Array.isArray(frames) || frames.length === 0) return null;
  if (timeSeconds <= (frames[0]!.time ?? 0)) return frames[0]!;
  const last = frames[frames.length - 1]!;
  if (timeSeconds >= (last.time ?? 0)) return last;
  // binary search
  let lo = 0;
  let hi = frames.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    const t = frames[mid]!.time ?? 0;
    if (t <= timeSeconds) lo = mid;
    else hi = mid;
  }
  const a = frames[lo]!;
  const b = frames[hi]!;
  const da = Math.abs((a.time ?? 0) - timeSeconds);
  const db = Math.abs((b.time ?? 0) - timeSeconds);
  return da <= db ? a : b;
}
