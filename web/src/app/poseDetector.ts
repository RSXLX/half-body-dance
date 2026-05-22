/**
 * PoseDetector abstraction.
 *
 * The PracticeView render loop only needs `detectForVideo` + `close`. The
 * production implementation wraps `@mediapipe/tasks-vision`'s PoseLandmarker
 * (Tasks Web — same model family the Python pipeline uses, see
 * docs/optimization-mediapipe-scoring.md §1.1). Tests pass a mock that
 * implements the same interface.
 */

import type { NormalizedLandmark } from '../core/types.js';

export interface PoseDetectionResult {
  landmarks: NormalizedLandmark[];
  visibility: number;
}

export interface PoseDetector {
  detectForVideo(video: HTMLVideoElement, timestampMs: number): PoseDetectionResult | null;
  close(): void;
}

export interface CreatePoseDetectorOptions {
  /** Override the model URL — defaults to MediaPipe's hosted lite model. */
  modelAssetPath?: string;
  /** Override the wasm fileset URL — defaults to jsdelivr. */
  wasmBaseUrl?: string;
  /**
   * Try GPU first, fall back to CPU on failure. Set to "cpu" to skip the GPU
   * attempt (useful on devices that crash on Tasks GPU init).
   */
  delegate?: 'gpu' | 'cpu' | 'auto';
}

/**
 * Lazily import @mediapipe/tasks-vision and create a PoseLandmarker. The
 * dynamic import keeps the WASM + JS bundle out of the initial Vite chunk.
 */
export async function createTasksPoseDetector(
  options: CreatePoseDetectorOptions = {},
): Promise<PoseDetector> {
  const mod = await import('@mediapipe/tasks-vision');
  const wasmBase =
    options.wasmBaseUrl ?? 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
  const modelPath =
    options.modelAssetPath ??
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

  const fileset = await mod.FilesetResolver.forVisionTasks(wasmBase);

  const tryDelegate = async (delegate: 'GPU' | 'CPU') => {
    return mod.PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelPath, delegate },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.45,
      minPosePresenceConfidence: 0.45,
      minTrackingConfidence: 0.35,
    });
  };

  let landmarker: Awaited<ReturnType<typeof tryDelegate>>;
  const want = options.delegate ?? 'auto';
  if (want === 'cpu') {
    landmarker = await tryDelegate('CPU');
  } else {
    try {
      landmarker = await tryDelegate('GPU');
    } catch (gpuError) {
      if (want === 'gpu') throw gpuError;
      // eslint-disable-next-line no-console
      console.warn('[poseDetector] GPU delegate failed, falling back to CPU', gpuError);
      landmarker = await tryDelegate('CPU');
    }
  }

  return {
    detectForVideo(video: HTMLVideoElement, timestampMs: number): PoseDetectionResult | null {
      if (video.readyState < 2 || video.videoWidth === 0) return null;
      const result = landmarker.detectForVideo(video, timestampMs);
      const first = result.landmarks?.[0];
      if (!first || !first.length) return null;
      const visAvg =
        first.reduce((sum, p) => sum + (p.visibility ?? 1), 0) / first.length;
      return {
        landmarks: first.map((p) => ({
          x: p.x,
          y: p.y,
          z: p.z,
          visibility: p.visibility,
        })),
        visibility: visAvg,
      };
    },
    close() {
      landmarker.close();
    },
  };
}

/**
 * No-op detector for environments without a camera or WASM support. Always
 * returns null, never throws. Used as a graceful fallback in main.ts.
 */
export function createNoopPoseDetector(): PoseDetector {
  return {
    detectForVideo: () => null,
    close: () => {},
  };
}
