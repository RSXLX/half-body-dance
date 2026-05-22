import { describe, expect, it } from 'vitest';
import {
  getStageContentRect,
  mapPointToStage,
  resolveCanvasPixelSize,
  resolveStageAspectRatio,
} from './stageRenderer.js';

describe('resolveStageAspectRatio', () => {
  it('prefers actual video dimensions when ready', () => {
    expect(
      resolveStageAspectRatio({
        videoReady: true,
        videoWidth: 1920,
        videoHeight: 1080,
      }),
    ).toBeCloseTo(16 / 9);
  });

  it('falls back to pose metadata when video not ready', () => {
    expect(
      resolveStageAspectRatio({
        videoReady: false,
        videoWidth: 0,
        videoHeight: 0,
        poseVideoWidth: 720,
        poseVideoHeight: 1280,
      }),
    ).toBeCloseTo(720 / 1280);
  });

  it('falls back to 9:16 when nothing known', () => {
    expect(
      resolveStageAspectRatio({
        videoReady: false,
        videoWidth: 0,
        videoHeight: 0,
      }),
    ).toBeCloseTo(9 / 16);
  });
});

describe('getStageContentRect', () => {
  it('letterboxes a wide canvas to a 9:16 aspect', () => {
    const rect = getStageContentRect({ width: 900, height: 800 }, 9 / 16);
    expect(rect.height).toBeCloseTo(800);
    expect(rect.width).toBeCloseTo(800 * (9 / 16));
    expect(rect.x).toBeCloseTo((900 - rect.width) / 2);
    expect(rect.y).toBeCloseTo(0);
  });

  it('pillarboxes a tall canvas to a 16:9 aspect', () => {
    const rect = getStageContentRect({ width: 500, height: 1000 }, 16 / 9);
    expect(rect.width).toBeCloseTo(500);
    expect(rect.height).toBeCloseTo(500 / (16 / 9));
    expect(rect.y).toBeCloseTo((1000 - rect.height) / 2);
  });

  it('returns zeros when canvas is zero-sized', () => {
    expect(getStageContentRect({ width: 0, height: 0 }, 1)).toEqual({
      x: 0,
      y: 0,
      width: 0,
      height: 0,
    });
  });
});

describe('mapPointToStage', () => {
  it('maps center of a 100x100 rect at offset (20,30)', () => {
    expect(mapPointToStage({ x: 0.5, y: 0.5 }, { x: 20, y: 30, width: 100, height: 100 })).toEqual({
      x: 70,
      y: 80,
    });
  });

  it('returns null for null inputs', () => {
    expect(mapPointToStage(null, { x: 0, y: 0, width: 1, height: 1 })).toBeNull();
    expect(mapPointToStage({ x: 0, y: 0 }, null)).toBeNull();
  });
});

describe('resolveCanvasPixelSize', () => {
  it('rounds CSS size * dpr to at least 1', () => {
    expect(resolveCanvasPixelSize({ width: 100, height: 50 }, 2)).toEqual({
      width: 200,
      height: 100,
    });
    expect(resolveCanvasPixelSize({ width: 0, height: 0 }, 2)).toEqual({
      width: 1,
      height: 1,
    });
  });
});
