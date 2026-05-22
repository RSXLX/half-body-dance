import { describe, expect, it } from 'vitest';
import fixture from '../core/__fixtures__/angel_three_frames.json' with { type: 'json' };
import type { PoseFrame } from '../core/types.js';
import { evaluateFrame, findFrameAtTime } from './scoringLoop.js';

type Fixture = { frames: PoseFrame[] };
const frames = (fixture as Fixture).frames;

describe('evaluateFrame', () => {
  const base = frames[0]!;

  it('returns no-camera when camera is off', () => {
    const r = evaluateFrame({
      userPose: null,
      targetFrame: base,
      personVisible: false,
      cameraRunning: false,
    });
    expect(r.kind).toBe('no-camera');
  });

  it('returns no-target when no target frame is passed', () => {
    const r = evaluateFrame({
      userPose: base.pose_landmarks,
      targetFrame: null,
      personVisible: true,
      cameraRunning: true,
    });
    expect(r.kind).toBe('no-target');
  });

  it('returns no-person when person is not visible', () => {
    const r = evaluateFrame({
      userPose: null,
      targetFrame: base,
      personVisible: false,
      cameraRunning: true,
    });
    expect(r.kind).toBe('no-person');
  });

  it('returns scored with total=100 when comparing a frame to itself', () => {
    const r = evaluateFrame({
      userPose: base.pose_landmarks,
      targetFrame: base,
      personVisible: true,
      cameraRunning: true,
    });
    expect(r.kind).toBe('scored');
    if (r.kind === 'scored') {
      expect(r.score).toBe(100);
      expect(r.label).toBe('高度匹配');
    }
  });

  it('returns scored with lower total between different frames', () => {
    const r = evaluateFrame({
      userPose: frames[0]!.pose_landmarks,
      targetFrame: frames[1]!,
      personVisible: true,
      cameraRunning: true,
    });
    if (r.kind === 'scored') {
      expect(r.score).toBeLessThan(100);
    }
  });
});

describe('findFrameAtTime', () => {
  const series: PoseFrame[] = Array.from({ length: 10 }, (_, i) => ({
    time: i * 0.1,
    pose_landmarks: [],
  }));

  it('returns first frame when time precedes range', () => {
    expect(findFrameAtTime(series, -1)).toBe(series[0]);
  });

  it('returns last frame when time exceeds range', () => {
    expect(findFrameAtTime(series, 99)).toBe(series[9]);
  });

  it('picks the closest frame, rounding down on ties', () => {
    // t=0.25 is equidistant from 0.2 and 0.3; the implementation prefers the lower one.
    expect(findFrameAtTime(series, 0.25)).toBe(series[2]);
  });

  it('picks the nearest neighbor for non-tie values', () => {
    expect(findFrameAtTime(series, 0.31)).toBe(series[3]);
    expect(findFrameAtTime(series, 0.79)).toBe(series[8]);
  });

  it('returns null on empty input', () => {
    expect(findFrameAtTime([], 0)).toBeNull();
    expect(findFrameAtTime(null, 0)).toBeNull();
  });
});
