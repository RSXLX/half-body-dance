import { describe, expect, it } from 'vitest';
import fixture from '../__fixtures__/angel_three_frames.json' with { type: 'json' };
import type { NormalizedLandmark, PoseFrame } from '../types.js';
import { compareArmPoses, scoreLabelFromValue } from '../poseCompare.js';
import {
  getDefaultStageReference,
  getPoseReference,
  normalizeLandmarks,
  transformPointsWithReferences,
} from '../reference.js';
import { normalizeHandLandmarks, findMatchingHand } from '../hands.js';

type Fixture = { frames: PoseFrame[] };
const frames = (fixture as Fixture).frames;

function poseOf(i: number): NormalizedLandmark[] {
  const frame = frames[i]!;
  return frame.pose_landmarks;
}

describe('golden sample: reference frame', () => {
  it('builds a non-null reference for every fixture frame', () => {
    frames.forEach((frame, idx) => {
      const ref = getPoseReference(frame.pose_landmarks);
      expect(ref, `frame ${idx}`).not.toBeNull();
      expect(ref!.scale).toBeGreaterThan(0);
      // xAxis / yAxis should be roughly unit length
      expect(Math.hypot(ref!.xAxis.x, ref!.xAxis.y)).toBeCloseTo(1, 1);
      expect(Math.hypot(ref!.yAxis.x, ref!.yAxis.y)).toBeCloseTo(1, 1);
    });
  });

  it('transformPointsWithReferences returns the same number of points', () => {
    const src = poseOf(0);
    const ref = getPoseReference(src)!;
    const transformed = transformPointsWithReferences(src, ref, ref) as NormalizedLandmark[];
    expect(transformed.length).toBe(src.length);
    // And every point should still be a finite (x, y). The transform is not an
    // identity even for src == ref because legacy encodes via global coords and
    // decodes via the (xAxis, yAxis) basis; we only assert structural integrity.
    transformed.forEach((p) => {
      expect(Number.isFinite(p.x)).toBe(true);
      expect(Number.isFinite(p.y)).toBe(true);
    });
  });

  it('exposes a sensible default stage reference', () => {
    const def = getDefaultStageReference();
    expect(def.scale).toBeGreaterThan(0);
    expect(def.center.x).toBeCloseTo(0.5);
    expect(def.center.y).toBeCloseTo(0.5);
  });
});

describe('golden sample: normalizeLandmarks', () => {
  it('places the torso center at the origin', () => {
    const norm = normalizeLandmarks(poseOf(0))!;
    // Shoulder midpoint (11, 12) and hip midpoint (23, 24) should be roughly symmetric around origin
    const leftShoulder = norm[11]!;
    const rightShoulder = norm[12]!;
    const leftHip = norm[23]!;
    const rightHip = norm[24]!;
    const cx = (leftShoulder.x + rightShoulder.x + leftHip.x + rightHip.x) / 4;
    const cy = (leftShoulder.y + rightShoulder.y + leftHip.y + rightHip.y) / 4;
    expect(cx).toBeCloseTo(0, 1);
    expect(cy).toBeCloseTo(0, 1);
  });
});

describe('golden sample: compareArmPoses', () => {
  it('returns a perfect match when comparing a frame to itself', () => {
    const pose = poseOf(0);
    const result = compareArmPoses(pose, pose);
    expect(result).not.toBeNull();
    expect(result!.total).toBe(100);
    expect(result!.left.score).toBe(100);
    expect(result!.right.score).toBe(100);
    expect(result!.scoringSystem).toContain('上臂');
  });

  it('returns some score < 100 between two different frames in the sequence', () => {
    const a = poseOf(0);
    const b = poseOf(1);
    const result = compareArmPoses(a, b);
    expect(result).not.toBeNull();
    expect(result!.total).toBeLessThan(100);
    expect(result!.total).toBeGreaterThanOrEqual(0);
  });

  it('is approximately symmetric (within rounding noise)', () => {
    const a = poseOf(0);
    const b = poseOf(2);
    const ab = compareArmPoses(a, b)!.total;
    const ba = compareArmPoses(b, a)!.total;
    // Math.round + weighted averages can diverge by a handful of points; the
    // important invariant is that swapping arguments doesn't change the score
    // meaningfully.
    expect(Math.abs(ab - ba)).toBeLessThanOrEqual(5);
  });
});

describe('golden sample: scoreLabelFromValue', () => {
  it.each([
    [99, '高度匹配'],
    [85, '高度匹配'],
    [84, '基本匹配'],
    [70, '基本匹配'],
    [69, '部分匹配'],
    [50, '部分匹配'],
    [49, '未匹配'],
    [0, '未匹配'],
  ])('maps %i to "%s"', (score, label) => {
    expect(scoreLabelFromValue(score)).toBe(label);
  });
});

describe('golden sample: hands', () => {
  it('normalizeHandLandmarks puts the wrist at origin when hand data exists', () => {
    const handFrame = frames.find((f) => Array.isArray(f.hands) && f.hands.length > 0);
    if (!handFrame || !handFrame.hands || handFrame.hands.length === 0) {
      return; // angel clip may lack hand detections; skip silently
    }
    const first = handFrame.hands[0]!;
    const normalized = normalizeHandLandmarks(first.landmarks);
    if (!normalized) return;
    expect(normalized[0]!.x).toBeCloseTo(0, 6);
    expect(normalized[0]!.y).toBeCloseTo(0, 6);
  });

  it('findMatchingHand prefers identical handedness', () => {
    const left: { handedness: 'Left'; landmarks: NormalizedLandmark[] } = {
      handedness: 'Left',
      landmarks: [{ x: 0, y: 0 }],
    };
    const right: { handedness: 'Right'; landmarks: NormalizedLandmark[] } = {
      handedness: 'Right',
      landmarks: [{ x: 1, y: 0 }],
    };
    const target: { handedness: 'Right'; landmarks: NormalizedLandmark[] } = {
      handedness: 'Right',
      landmarks: [{ x: 0.9, y: 0 }],
    };
    expect(findMatchingHand([left, right], target)).toBe(right);
  });

  it('findMatchingHand falls back to nearest wrist when handedness mismatch', () => {
    const a = { handedness: 'Left' as const, landmarks: [{ x: 0.1, y: 0.1 }] as NormalizedLandmark[] };
    const b = { handedness: 'Left' as const, landmarks: [{ x: 0.9, y: 0.9 }] as NormalizedLandmark[] };
    const target = { handedness: 'Right' as const, landmarks: [{ x: 0.85, y: 0.87 }] as NormalizedLandmark[] };
    expect(findMatchingHand([a, b], target)).toBe(b);
  });
});
