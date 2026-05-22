/**
 * Practice render loop.
 *
 * Drives the PoseDetector + scoring pipeline at a steady cadence and pushes
 * smoothed scores into a callback the view consumes. All scheduling is
 * injectable so vitest can drive deterministic ticks; in production we use
 * `requestAnimationFrame` + `performance.now`.
 *
 * Key tunables (see docs/optimization-mediapipe-scoring.md):
 *   - detectionStride : run detection every Nth frame; reuse last landmarks
 *   - evaluateEveryMs : minimum interval between evaluateFrame() invocations
 *   - smoother alpha  : EMA reactivity
 */

import type { NormalizedLandmark, PoseFrame } from '../core/types.js';
import type { PoseDetector } from './poseDetector.js';
import { evaluateFrame, findFrameAtTime, type MatchState } from './scoringLoop.js';
import { createScoreSmoother, type ScoreSmoother } from './scoreSmoothing.js';

export interface PracticeLoopOptions {
  detector: PoseDetector;
  video: HTMLVideoElement;
  /** Returns the current target frame at this clock instant, or null. */
  getTargetFrame: () => PoseFrame | null;
  /** Whether the user pose pipeline is considered active. */
  isCameraRunning: () => boolean;
  onScore: (state: { match: MatchState; smoothed: number | null; raw: number | null }) => void;
  /**
   * Called every render tick with the latest detected user landmarks (or null
   * when detection failed/skipped). Lets the view paint at rAF cadence even
   * though scoring runs at evaluateEveryMs.
   */
  onFrame?: (state: { userPose: NormalizedLandmark[] | null; targetFrame: PoseFrame | null; t: number }) => void;
  detectionStride?: number;
  evaluateEveryMs?: number;
  smootherAlpha?: number;
  schedule?: (cb: (now: number) => void) => number;
  cancel?: (handle: number) => void;
  now?: () => number;
}

export interface PracticeLoopHandle {
  stop(): void;
  /** Inspectable for tests. */
  readonly smoother: ScoreSmoother;
}

export function startPracticeLoop(opts: PracticeLoopOptions): PracticeLoopHandle {
  const stride = Math.max(1, Math.floor(opts.detectionStride ?? 2));
  const evaluateEveryMs = Math.max(0, opts.evaluateEveryMs ?? 100);
  const smoother = createScoreSmoother({ alpha: opts.smootherAlpha ?? 0.3 });
  const schedule = opts.schedule ?? ((cb) => requestAnimationFrame(cb));
  const cancel = opts.cancel ?? ((h) => cancelAnimationFrame(h));
  // `opts.now` is reserved for future non-schedule-driven paths; tests pass it
  // for symmetry with other helpers but the current loop uses the timestamp
  // delivered by the scheduler.
  void opts.now;

  let frameCount = 0;
  let lastEvaluateAt: number | null = null;
  let lastLandmarks: NormalizedLandmark[] | null = null;
  let lastVisibility = 0;
  let stopped = false;
  let pending = 0;

  function tick(t: number) {
    if (stopped) return;
    frameCount++;
    const cameraOn = opts.isCameraRunning();
    if (cameraOn && frameCount % stride === 0) {
      try {
        const detection = opts.detector.detectForVideo(opts.video, t);
        if (detection) {
          lastLandmarks = detection.landmarks;
          lastVisibility = detection.visibility;
        } else {
          lastLandmarks = null;
          lastVisibility = 0;
        }
      } catch {
        lastLandmarks = null;
        lastVisibility = 0;
      }
    }

    const target = opts.getTargetFrame();
    if (lastEvaluateAt === null || t - lastEvaluateAt >= evaluateEveryMs) {
      lastEvaluateAt = t;
      const match = evaluateFrame({
        userPose: lastLandmarks,
        targetFrame: target,
        personVisible: lastVisibility >= 0.4,
        cameraRunning: cameraOn,
      });
      const raw = match.kind === 'scored' ? match.score : null;
      const smoothed = smoother.push(raw);
      opts.onScore({ match, smoothed, raw });
    }
    opts.onFrame?.({ userPose: lastLandmarks, targetFrame: target, t });

    pending = schedule(tick);
  }

  pending = schedule(tick);

  return {
    stop() {
      stopped = true;
      cancel(pending);
    },
    smoother,
  };
}

/** Pure helper used by main.ts: convert audio time → target frame. */
export function targetFrameAt(
  frames: readonly PoseFrame[] | null | undefined,
  timeSeconds: number,
): PoseFrame | null {
  return findFrameAtTime(frames, timeSeconds);
}
