/**
 * Score smoothing — one-pole IIR (EMA).
 *
 *   s_t = alpha * input + (1 - alpha) * s_{t-1}
 *
 * `alpha` close to 1 reacts fast (= raw); close to 0 is sticky. The default
 * 0.3 takes ~3 frames to follow a step change to within 5%, which feels stable
 * on the score dial without lagging perceptibly.
 *
 * `null` resets the smoother — useful when the user hides from the camera.
 * The smoother stays in floating point; round at the UI boundary only.
 */

export interface ScoreSmoother {
  push(value: number | null): number | null;
  reset(): void;
  readonly value: number | null;
}

export interface ScoreSmootherOptions {
  /** EMA alpha. Default 0.3. */
  alpha?: number;
  /** Initial value; default null (no smoothing until first non-null input). */
  initial?: number | null;
}

export function createScoreSmoother(options: ScoreSmootherOptions = {}): ScoreSmoother {
  const alpha = clamp01(options.alpha ?? 0.3);
  let current: number | null = options.initial ?? null;

  return {
    push(value: number | null): number | null {
      if (value === null || !Number.isFinite(value)) {
        // Treat as a soft reset: keep the last visible value but stop updating.
        return current;
      }
      if (current === null) {
        current = value;
      } else {
        current = alpha * value + (1 - alpha) * current;
      }
      return current;
    },
    reset(): void {
      current = options.initial ?? null;
    },
    get value(): number | null {
      return current;
    },
  };
}

function clamp01(n: number): number {
  if (!Number.isFinite(n)) return 0.3;
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

/** Helper: render a smoothed score for the HUD. */
export function formatSmoothedScore(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return '--';
  return `${Math.round(value)}%`;
}
