import { describe, expect, it } from 'vitest';
import { animateScoreText } from './animateScore.js';

describe('animateScoreText', () => {
  it('passes through non-numeric target immediately', () => {
    const seen: string[] = [];
    animateScoreText((v) => seen.push(v), '--', {
      now: () => 0,
      schedule: () => {},
    });
    expect(seen).toEqual(['--']);
  });

  it('interpolates to the target over duration and ends on the exact label', () => {
    const seen: string[] = [];
    const pending: Array<(now: number) => void> = [];
    let clock = 0;
    animateScoreText((v) => seen.push(v), '82%', {
      durationMs: 100,
      now: () => clock,
      schedule: (cb) => pending.push(cb),
    });
    // tick roughly every 25ms
    const ticks = [0, 25, 50, 75, 100];
    ticks.forEach((t) => {
      clock = t;
      const next = pending.shift();
      if (next) next(t);
    });
    expect(seen.length).toBeGreaterThan(1);
    expect(seen[seen.length - 1]).toBe('82%');
    // Monotonic non-decreasing integer progress on the way up
    const numeric = seen.map((s) => Number.parseInt(s, 10));
    for (let i = 1; i < numeric.length; i++) {
      expect(numeric[i]!).toBeGreaterThanOrEqual(numeric[i - 1]!);
    }
  });

  it('keeps one decimal when the target is fractional', () => {
    const seen: string[] = [];
    const pending: Array<(now: number) => void> = [];
    let clock = 0;
    animateScoreText((v) => seen.push(v), '9.5', {
      durationMs: 100,
      now: () => clock,
      schedule: (cb) => pending.push(cb),
    });
    [0, 50].forEach((t) => {
      clock = t;
      const next = pending.shift();
      if (next) next(t);
    });
    // Any intermediate value should have a decimal
    const intermediate = seen.slice(0, -1);
    if (intermediate.length > 0) {
      expect(intermediate.some((s) => s.includes('.'))).toBe(true);
    }
  });
});
