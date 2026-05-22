/**
 * Animated counter for a numeric label. Pure-ish: side effect is invoking
 * `setText` with intermediate string values. The clock and rAF are injected so
 * tests can drive the animation deterministically.
 */

export interface AnimateScoreOptions {
  durationMs?: number;
  /** Use to inject deterministic time/RAF in tests. */
  now?: () => number;
  schedule?: (cb: (now: number) => void) => void;
}

export function animateScoreText(
  setText: (value: string) => void,
  targetText: string,
  options: AnimateScoreOptions = {},
): void {
  const numericMatch = String(targetText).match(/-?\d+(?:\.\d+)?/);
  if (!numericMatch) {
    setText(targetText);
    return;
  }
  const target = Number.parseFloat(numericMatch[0]);
  if (!Number.isFinite(target)) {
    setText(targetText);
    return;
  }
  const suffix = String(targetText).replace(/-?\d+(?:\.\d+)?/, '').trim();
  const duration = options.durationMs ?? 700;
  const now = options.now ?? (() => performance.now());
  const schedule = options.schedule ?? ((cb) => requestAnimationFrame(cb));
  const start = now();

  const isInteger = Number.isInteger(target);

  function step(currentTime: number) {
    const elapsed = currentTime - start;
    const t = Math.min(1, Math.max(0, elapsed / duration));
    const eased = 1 - Math.pow(1 - t, 3);
    const v = target * eased;
    if (t >= 1) {
      setText(targetText);
      return;
    }
    if (suffix) {
      setText(`${Math.round(v)}${suffix}`);
    } else if (isInteger) {
      setText(String(Math.round(v)));
    } else {
      setText(v.toFixed(1));
    }
    schedule(step);
  }

  schedule(step);
}
