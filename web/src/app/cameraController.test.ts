import { describe, expect, it, vi } from 'vitest';
import {
  describeCameraError,
  createCameraController,
  getPreferredFrontCameraConstraints,
} from './cameraController.js';

function fakeStream(stopped: string[]) {
  return {
    getTracks: () => [
      {
        stop: () => stopped.push('stopped'),
      },
    ],
  } as unknown as MediaStream;
}

describe('getPreferredFrontCameraConstraints', () => {
  it('prefers the user-facing camera at 1280x720', () => {
    const c = getPreferredFrontCameraConstraints();
    expect(c.facingMode.ideal).toBe('user');
    expect(c.width.ideal).toBe(1280);
    expect(c.height.ideal).toBe(720);
  });
});

describe('describeCameraError', () => {
  it('turns common getUserMedia failures into actionable Chinese hints', () => {
    expect(describeCameraError(new DOMException('denied', 'NotAllowedError'))).toContain('权限');
    expect(describeCameraError(new DOMException('missing', 'NotFoundError'))).toContain('没有检测到摄像头');
    expect(describeCameraError(new DOMException('busy', 'NotReadableError'))).toContain('占用');
    expect(describeCameraError(new DOMException('bad constraint', 'OverconstrainedError'))).toContain('分辨率');
  });

  it('keeps an unknown error message when no known browser name is present', () => {
    expect(describeCameraError(new Error('custom failure'))).toBe('custom failure');
  });
});

describe('createCameraController', () => {
  it('starts the camera and records running state', async () => {
    const stopped: string[] = [];
    const stream = fakeStream(stopped);
    const request = vi.fn<(c: MediaStreamConstraints) => Promise<MediaStream>>(async () => stream);
    const ctrl = createCameraController({ requestUserMedia: request });
    expect(ctrl.running).toBe(false);
    await ctrl.start();
    expect(ctrl.running).toBe(true);
    expect(ctrl.stream).toBe(stream);
    expect(request).toHaveBeenCalledTimes(1);
    const passed = request.mock.calls[0]![0];
    expect(passed.audio).toBe(false);
    expect(passed.video as Record<string, unknown>).toMatchObject({
      facingMode: { ideal: 'user' },
    });
  });

  it('stops tracks and clears state on stop()', async () => {
    const stopped: string[] = [];
    const stream = fakeStream(stopped);
    const ctrl = createCameraController({ requestUserMedia: async () => stream });
    await ctrl.start();
    ctrl.stop();
    expect(stopped).toEqual(['stopped']);
    expect(ctrl.running).toBe(false);
    expect(ctrl.stream).toBeNull();
  });

  it('restarts cleanly — previous stream is stopped before a second start', async () => {
    const stoppedA: string[] = [];
    const stoppedB: string[] = [];
    const streamA = fakeStream(stoppedA);
    const streamB = fakeStream(stoppedB);
    const requests = [streamA, streamB];
    const ctrl = createCameraController({
      requestUserMedia: async () => requests.shift()!,
    });
    await ctrl.start();
    await ctrl.start();
    expect(stoppedA).toEqual(['stopped']);
    expect(ctrl.stream).toBe(streamB);
  });

  it('keeps the latest stream when an earlier overlapping start resolves last', async () => {
    const stoppedA: string[] = [];
    const stoppedB: string[] = [];
    const streamA = fakeStream(stoppedA);
    const streamB = fakeStream(stoppedB);
    const pending: Array<(stream: MediaStream) => void> = [];
    const ctrl = createCameraController({
      requestUserMedia: () => new Promise<MediaStream>((resolve) => pending.push(resolve)),
    });

    const firstStart = ctrl.start();
    const secondStart = ctrl.start();

    pending[1]!(streamB);
    await expect(secondStart).resolves.toBe(streamB);
    expect(ctrl.stream).toBe(streamB);
    expect(stoppedB).toEqual([]);

    pending[0]!(streamA);
    await expect(firstStart).resolves.toBe(streamB);
    expect(ctrl.stream).toBe(streamB);
    expect(stoppedA).toEqual(['stopped']);
    expect(stoppedB).toEqual([]);
  });

  it('rejects a pending start that becomes stale after stop()', async () => {
    const stopped: string[] = [];
    const staleStream = fakeStream(stopped);
    let resolveStart: ((stream: MediaStream) => void) | null = null;
    const ctrl = createCameraController({
      requestUserMedia: () => new Promise<MediaStream>((resolve) => {
        resolveStart = resolve;
      }),
    });

    const start = ctrl.start();
    ctrl.stop();
    resolveStart!(staleStream);

    await expect(start).rejects.toThrow('stale camera start ignored');
    expect(stopped).toEqual(['stopped']);
    expect(ctrl.stream).toBeNull();
    expect(ctrl.running).toBe(false);
  });

  it('surfaces getUserMedia errors', async () => {
    const ctrl = createCameraController({
      requestUserMedia: async () => {
        throw new Error('denied');
      },
    });
    await expect(ctrl.start()).rejects.toThrow('denied');
    expect(ctrl.running).toBe(false);
  });
});
