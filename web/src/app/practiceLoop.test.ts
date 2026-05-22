// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest';
import fixture from '../core/__fixtures__/angel_three_frames.json' with { type: 'json' };
import type { PoseFrame } from '../core/types.js';
import type { PoseDetector } from './poseDetector.js';
import { startPracticeLoop } from './practiceLoop.js';

type Fixture = { frames: PoseFrame[] };
const frames = (fixture as Fixture).frames;

function makeFakeVideo(): HTMLVideoElement {
  const v = document.createElement('video');
  Object.defineProperty(v, 'readyState', { value: 4, configurable: true });
  Object.defineProperty(v, 'videoWidth', { value: 720, configurable: true });
  Object.defineProperty(v, 'videoHeight', { value: 1280, configurable: true });
  return v;
}

function makeDetector(returnPose: PoseFrame | null): PoseDetector {
  return {
    detectForVideo() {
      if (!returnPose) return null;
      return {
        landmarks: returnPose.pose_landmarks,
        visibility: 0.95,
      };
    },
    close() {},
  };
}

interface Driver {
  tick: () => void;
  scheduled: number;
  cancelled: number;
  scoreCalls: Array<{ smoothed: number | null; raw: number | null }>;
  setNow(ms: number): void;
}

function buildDriver(initialNow = 0) {
  const driver: Driver = {
    tick: () => {},
    scheduled: 0,
    cancelled: 0,
    scoreCalls: [],
    setNow: (ms) => {
      now = ms;
    },
  };
  let pending: ((now: number) => void) | null = null;
  let now = initialNow;
  driver.tick = () => {
    const fn = pending;
    pending = null;
    if (fn) fn(now);
  };
  return {
    driver,
    api: {
      schedule: (cb: (now: number) => void) => {
        driver.scheduled++;
        pending = cb;
        return driver.scheduled;
      },
      cancel: () => {
        driver.cancelled++;
      },
      now: () => now,
    },
  };
}

describe('startPracticeLoop', () => {
  it('schedules a tick and stops on stop()', () => {
    const { driver, api } = buildDriver();
    const handle = startPracticeLoop({
      detector: makeDetector(null),
      video: makeFakeVideo(),
      getTargetFrame: () => frames[0]!,
      isCameraRunning: () => false,
      onScore: () => {},
      ...api,
    });
    expect(driver.scheduled).toBe(1);
    handle.stop();
    expect(driver.cancelled).toBe(1);
  });

  it('runs detector every Nth tick (detection stride)', () => {
    const { driver, api } = buildDriver();
    const detector = makeDetector(frames[0]!);
    const detectSpy = vi.spyOn(detector, 'detectForVideo');
    const handle = startPracticeLoop({
      detector,
      video: makeFakeVideo(),
      getTargetFrame: () => frames[0]!,
      isCameraRunning: () => true,
      onScore: () => {},
      detectionStride: 3,
      evaluateEveryMs: 0,
      ...api,
    });
    for (let i = 0; i < 9; i++) {
      driver.setNow(i * 16);
      driver.tick();
    }
    handle.stop();
    // ticks 3, 6, 9 satisfy frameCount % 3 === 0 → 3 detections
    expect(detectSpy).toHaveBeenCalledTimes(3);
  });

  it('throttles evaluate by evaluateEveryMs', () => {
    const { driver, api } = buildDriver();
    const calls: Array<{ smoothed: number | null }> = [];
    const handle = startPracticeLoop({
      detector: makeDetector(frames[0]!),
      video: makeFakeVideo(),
      getTargetFrame: () => frames[0]!,
      isCameraRunning: () => true,
      onScore: (s) => calls.push({ smoothed: s.smoothed }),
      detectionStride: 1,
      evaluateEveryMs: 100,
      ...api,
    });
    for (const ts of [0, 30, 60, 99, 100, 150, 200, 250]) {
      driver.setNow(ts);
      driver.tick();
    }
    handle.stop();
    // evaluate at 0, 100, 200 → 3 score callbacks
    expect(calls.length).toBe(3);
  });

  it('produces a smoothed score that approaches 100 against same-frame target', () => {
    const { driver, api } = buildDriver();
    const calls: Array<{ smoothed: number | null }> = [];
    const handle = startPracticeLoop({
      detector: makeDetector(frames[0]!),
      video: makeFakeVideo(),
      getTargetFrame: () => frames[0]!,
      isCameraRunning: () => true,
      onScore: (s) => calls.push({ smoothed: s.smoothed }),
      detectionStride: 1,
      evaluateEveryMs: 0,
      smootherAlpha: 0.5,
      ...api,
    });
    for (let i = 0; i < 10; i++) {
      driver.setNow(i);
      driver.tick();
    }
    handle.stop();
    const last = calls[calls.length - 1]!.smoothed!;
    expect(last).toBeGreaterThan(95);
  });

  it('emits no-camera state when camera is off', () => {
    const { driver, api } = buildDriver();
    const captured: Array<unknown> = [];
    const handle = startPracticeLoop({
      detector: makeDetector(frames[0]!),
      video: makeFakeVideo(),
      getTargetFrame: () => frames[0]!,
      isCameraRunning: () => false,
      onScore: (s) => captured.push(s.match),
      detectionStride: 1,
      evaluateEveryMs: 0,
      ...api,
    });
    driver.setNow(1);
    driver.tick();
    handle.stop();
    const first = captured[0] as { kind: string };
    expect(first.kind).toBe('no-camera');
  });
});
