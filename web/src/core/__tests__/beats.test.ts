import { describe, expect, it } from 'vitest';
import { detectBeats, beatIndexAtTime } from '../beats.js';
import type { PoseData, NormalizedLandmark } from '../types.js';

function makePose(x: number, y: number): NormalizedLandmark[] {
  const pts: NormalizedLandmark[] = Array.from({ length: 33 }, () => ({
    x: 0.5,
    y: 0.5,
    visibility: 1,
  }));
  // 双腕 (15/16) 注入运动
  pts[15] = { x, y, visibility: 1 };
  pts[16] = { x, y, visibility: 1 };
  // 肩 11/12 / 髋 23/24，让 reference 可以构建
  pts[11] = { x: 0.4, y: 0.4, visibility: 1 };
  pts[12] = { x: 0.6, y: 0.4, visibility: 1 };
  pts[23] = { x: 0.45, y: 0.7, visibility: 1 };
  pts[24] = { x: 0.55, y: 0.7, visibility: 1 };
  return pts;
}

function syntheticPoseData(): PoseData {
  // 60 帧 / 6 秒 / 三个明显速度峰值
  const frames = [];
  for (let i = 0; i < 60; i++) {
    const t = i * 0.1;
    // 三段三角波，峰在 t≈1, 3, 5
    const phase = ((t % 2) - 1);
    const x = 0.5 + 0.3 * (1 - Math.abs(phase));
    frames.push({ time: t, pose_landmarks: makePose(x, 0.5) });
  }
  return { fps: 10, frames };
}

describe('detectBeats', () => {
  it('returns empty when input is missing', () => {
    expect(detectBeats(null)).toEqual([]);
    expect(detectBeats(undefined)).toEqual([]);
    expect(detectBeats({ fps: 30, frames: [] })).toEqual([]);
  });

  it('detects velocity peaks from synthetic wrist motion', () => {
    const beats = detectBeats(syntheticPoseData(), null, { source: 'velocity-peak' });
    expect(beats.length).toBeGreaterThanOrEqual(2);
    beats.forEach((b, i) => {
      expect(b.beatIndex).toBe(i);
      expect(b.endTime).toBeGreaterThan(b.startTime);
      expect(b.timestamp).toBeGreaterThanOrEqual(b.startTime);
      expect(b.timestamp).toBeLessThanOrEqual(b.endTime);
      expect(b.source).toBe('velocity-peak');
    });
  });

  it('falls back to uniform beats when no peaks', () => {
    const flat: PoseData = {
      fps: 10,
      frames: Array.from({ length: 30 }, (_, i) => ({
        time: i * 0.1,
        pose_landmarks: makePose(0.5, 0.5),
      })),
    };
    const beats = detectBeats(flat, null, { source: 'fallback-uniform', fallbackBeats: 6 });
    expect(beats).toHaveLength(6);
    expect(beats[0]!.source).toBe('fallback-uniform');
  });

  it('beatIndexAtTime returns proper index', () => {
    const beats = detectBeats(syntheticPoseData(), null, { source: 'fallback-uniform', fallbackBeats: 4 });
    expect(beatIndexAtTime(beats, beats[1]!.timestamp)).toBe(1);
    expect(beatIndexAtTime([], 1)).toBe(-1);
  });

  it('honours explicit beatTimes from MotionAnalysis', () => {
    const motion = {
      schema_version: 1 as const,
      source_pose: 'x',
      fps: 10,
      duration: 6,
      extracted_at: '',
      extract_config: {
        stride: 1,
        smoothingWindow: 3,
        stillSpeedThreshold: 0.1,
        visibilityThreshold: 0.3,
        minSegmentDuration: 0.5,
        maxSegments: 16,
      },
      trajectories: {},
      segments: [],
      hints: [],
      summary: { primaryJoints: [], dominantDirections: [], beatTimes: [1, 2.5, 4] },
    };
    const beats = detectBeats(syntheticPoseData(), motion as never);
    expect(beats.map((b) => b.timestamp)).toEqual([1, 2.5, 4]);
    expect(beats[0]!.source).toBe('beatTimes');
  });
});
