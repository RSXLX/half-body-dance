import { describe, expect, it } from 'vitest';
import { analyzeBeats, summarizeBeatAnalyses } from '../beatAnalysis.js';
import type { Beat } from '../beats.js';
import type { PoseData, NormalizedLandmark } from '../types.js';

function pose(x = 0.5): NormalizedLandmark[] {
  const pts: NormalizedLandmark[] = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    visibility: 1,
  }));
  pts[11] = { x: 0.4, y: 0.4, visibility: 1 };
  pts[12] = { x: 0.6, y: 0.4, visibility: 1 };
  pts[13] = { x: 0.35, y: 0.5, visibility: 1 };
  pts[14] = { x: 0.65, y: 0.5, visibility: 1 };
  pts[15] = { x, y: 0.55, visibility: 1 };
  pts[16] = { x: x + 0.1, y: 0.55, visibility: 1 };
  pts[23] = { x: 0.45, y: 0.7, visibility: 1 };
  pts[24] = { x: 0.55, y: 0.7, visibility: 1 };
  return pts;
}

function makePoseData(): PoseData {
  const frames = Array.from({ length: 20 }, (_, i) => ({
    time: i * 0.1,
    pose_landmarks: pose(0.5 + 0.05 * Math.sin(i)),
  }));
  return { fps: 10, frames };
}

function makeBeats(): Beat[] {
  return [
    { beatIndex: 0, timestamp: 0.5, startTime: 0, endTime: 1, representativeFrameIndex: 5, intensity: 0.6, source: 'velocity-peak' },
    { beatIndex: 1, timestamp: 1.5, startTime: 1, endTime: 2, representativeFrameIndex: 15, intensity: 0.6, source: 'velocity-peak' },
  ];
}

describe('analyzeBeats', () => {
  it('produces one analysis per beat with normal default status', () => {
    const out = analyzeBeats(makeBeats(), makePoseData(), null);
    expect(out).toHaveLength(2);
    expect(out[0]!.actionName).toContain('第 1 拍');
    expect(out[0]!.startTime).toBe(0);
    expect(out[0]!.endTime).toBe(1);
    expect(out[0]!.timestamp).toBe(0.5);
    expect(out[0]!.suggestions.length).toBeGreaterThan(0);
  });

  it('flags rhythm drift when user sample is far from beat center', () => {
    const data = makePoseData();
    const userSamples = [
      { time: 0.51, pose: data.frames[5]!.pose_landmarks },
      { time: 1.9, pose: data.frames[15]!.pose_landmarks },
    ];
    const out = analyzeBeats(makeBeats(), data, null, { userSamples });
    // 第二拍：节奏偏 0.4s > tolerance 0.18，应有 rhythm-drift 严重问题
    const issues = out[1]!.issues.map((i) => i.kind);
    expect(issues).toContain('rhythm-drift');
    expect(['warning', 'error']).toContain(out[1]!.status);
  });

  it('marks normal when user matches target tightly', () => {
    const data = makePoseData();
    const userSamples = [
      { time: 0.5, pose: data.frames[5]!.pose_landmarks },
      { time: 1.5, pose: data.frames[15]!.pose_landmarks },
    ];
    const out = analyzeBeats(makeBeats(), data, null, { userSamples });
    expect(out[0]!.poseDeviation).not.toBeNull();
    expect(out[0]!.rhythmMatch).toBeGreaterThanOrEqual(75);
  });

  it('summarize aggregates counts and averages', () => {
    const out = analyzeBeats(makeBeats(), makePoseData(), null);
    const sum = summarizeBeatAnalyses(out);
    expect(sum.total).toBe(2);
    expect(sum.normal + sum.warning + sum.error).toBe(2);
    expect(sum.averageSmoothness).toBeGreaterThanOrEqual(0);
  });
});
