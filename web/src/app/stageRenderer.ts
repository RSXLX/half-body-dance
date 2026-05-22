/**
 * Stage coordinate helpers — pure math separated from canvas drawing.
 *
 * The practice stage renders the teacher skeleton + user pose into a canvas
 * that may be a different size than the source video. These helpers convert
 * between the normalized [0,1] landmark space and canvas pixel coordinates.
 *
 * All functions are pure; the caller owns the CanvasRenderingContext2D.
 */

export interface CanvasSize {
  width: number;
  height: number;
}

export interface StageRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Vec2Point {
  x: number;
  y: number;
}

/**
 * Compute the aspect ratio of the actual video content being shown.
 * Callers pass in measurements they already have; we don't touch DOM here.
 */
export function resolveStageAspectRatio(args: {
  videoReady: boolean;
  videoWidth: number;
  videoHeight: number;
  poseVideoWidth?: number | null;
  poseVideoHeight?: number | null;
  fallback?: number;
}): number {
  if (args.videoReady && args.videoWidth > 0 && args.videoHeight > 0) {
    return args.videoWidth / args.videoHeight;
  }
  const pw = Number(args.poseVideoWidth) || 0;
  const ph = Number(args.poseVideoHeight) || 0;
  if (pw > 0 && ph > 0) return pw / ph;
  return args.fallback ?? 9 / 16;
}

/**
 * Given a canvas size and desired aspect ratio, returns a letterboxed content
 * rectangle centered within the canvas.
 */
export function getStageContentRect(canvas: CanvasSize, aspectRatio: number): StageRect {
  const canvasWidth = canvas.width || 0;
  const canvasHeight = canvas.height || 0;
  if (!canvasWidth || !canvasHeight) {
    return { x: 0, y: 0, width: canvasWidth, height: canvasHeight };
  }
  const widthFromHeight = canvasHeight * aspectRatio;
  let width = canvasWidth;
  let height = canvasHeight;
  if (widthFromHeight <= canvasWidth) {
    width = widthFromHeight;
  } else {
    height = canvasWidth / aspectRatio;
  }
  return {
    x: (canvasWidth - width) / 2,
    y: (canvasHeight - height) / 2,
    width,
    height,
  };
}

/** Map a normalized (0..1) landmark to pixel coordinates within `rect`. */
export function mapPointToStage(
  point: Vec2Point | null | undefined,
  rect: StageRect | null | undefined,
): Vec2Point | null {
  if (!point || !rect) return null;
  return {
    x: rect.x + point.x * rect.width,
    y: rect.y + point.y * rect.height,
  };
}

/**
 * Compute the backing-store resolution for a canvas given its CSS size and
 * the device pixel ratio. Pure; caller applies the result to canvas.width/height.
 */
export function resolveCanvasPixelSize(css: CanvasSize, dpr = 1): CanvasSize {
  return {
    width: Math.max(1, Math.round((css.width || 0) * dpr)),
    height: Math.max(1, Math.round((css.height || 0) * dpr)),
  };
}
